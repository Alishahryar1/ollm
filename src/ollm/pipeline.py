import os
import time
import threading
import queue
import torch
from typing import Optional, Dict, Any, Tuple

class LayerPipeline:
    """
    Double-buffered 3-stage pipeline for layer loading with CYCLIC continuation.
    This implementation pre-allocates two sets of buffers (CPU and GPU) and cycles
    between them to eliminate memory allocation overhead during inference.

    - Stage 1: GPU computes on Buffer A.
    - Stage 2: CPU->GPU transfers the next layer into Buffer B (async).
    - Stage 3: SSD->CPU loads the layer after next into CPU Buffer A (async).
    """

    def __init__(self, loader, stats=None, num_layers=32):
        self.loader = loader
        self.stats = stats
        self.num_layers = num_layers

        # --- New Double-Buffer Management ---
        self.layer_structure = self._get_layer_structure()
        
        # Pre-allocate two sets of buffers (CPU Pinned and GPU)
        print("[Pipeline] Allocating double buffers...")
        self.cpu_buffers = [self._allocate_buffers(device='cpu', pinned=True) for _ in range(2)]
        self.gpu_buffers = [self._allocate_buffers(device=self.loader.device, pinned=False) for _ in range(2)]
        print("[Pipeline] Buffer allocation complete.")

        # Tracks which logical layer is in which physical buffer.
        # Key: buffer_idx (0 or 1), Value: (layer_idx, generation)
        self.buffer_status: Dict[int, Tuple[int, int]] = {}
        
        # Tracks which buffer the GPU is currently computing on.
        self.compute_buffer_idx = 0

        # --- Async Infrastructure ---
        self.disk_to_cpu_queue = queue.Queue(maxsize=3)
        self.disk_loader_thread = None
        self.stop_thread = threading.Event()
        
        # Events now track the readiness of a CPU buffer for transfer.
        self.cpu_buffer_ready_events = {0: threading.Event(), 1: threading.Event()}
        
        self.transfer_stream = torch.cuda.Stream()
        
        # --- State Tracking ---
        self.current_generation = 0
        self._layers_processed = 0
        self.cyclic_enabled = True
        
        print(f"[Pipeline] Initialized with {num_layers} layers and double buffers (cyclic mode).")

    def _get_layer_structure(self) -> Dict[str, Tuple[torch.Size, torch.dtype]]:
        """Extracts tensor names, shapes, and dtypes from layer 0 manifest."""
        base = "model.layers.0."
        self.loader.preload_layer_safetensors(base)
        temp_dict = self.loader.load_dict_from_disk(base, device='cpu')
        structure = {name: (tensor.shape, tensor.dtype) for name, tensor in temp_dict.items()}
        return structure

    def _allocate_buffers(self, device: str, pinned: bool) -> Dict[str, torch.Tensor]:
        """Allocates a dictionary of empty tensors based on the layer structure."""
        buffers = {}
        for name, (shape, dtype) in self.layer_structure.items():
            tensor = torch.empty(shape, dtype=dtype, device=device)
            if pinned:
                tensor = tensor.pin_memory()
            buffers[name] = tensor
        return buffers

    def start(self):
        """Start the disk loading thread."""
        if self.disk_loader_thread is None or not self.disk_loader_thread.is_alive():
            self.stop_thread.clear()
            self.disk_loader_thread = threading.Thread(target=self._disk_loader_worker, daemon=True)
            self.disk_loader_thread.start()
            print("[Pipeline] Disk loader thread started.")

    def stop(self):
        """Stop the disk loading thread."""
        self.stop_thread.set()
        try:
            self.disk_to_cpu_queue.put_nowait(None)
        except queue.Full:
            pass
        if self.disk_loader_thread:
            self.disk_loader_thread.join(timeout=2.0)
        print("[Pipeline] Disk loader thread stopped.")

    def _disk_loader_worker(self):
        """Background thread that loads layers from disk into a pre-allocated CPU buffer."""
        while not self.stop_thread.is_set():
            try:
                task = self.disk_to_cpu_queue.get(timeout=0.1)
                if task is None:
                    break
                
                layer_idx, generation, target_buffer_idx, event = task
                base = f"model.layers.{layer_idx}."
                
                t1 = time.perf_counter()
                self.loader.preload_layer_safetensors(base)
                
                # Load data temporarily to a standard CPU tensor
                layer_dict_temp = self.loader.load_dict_from_disk(base, device='cpu')
                
                # Copy data into the pre-allocated pinned buffer
                target_cpu_buffer = self.cpu_buffers[target_buffer_idx]
                for k, temp_tensor in layer_dict_temp.items():
                    target_cpu_buffer[k].copy_(temp_tensor)
                
                self.buffer_status[target_buffer_idx] = (layer_idx, generation)
                event.set()
                
                if self.stats:
                    self.stats.set("disk_to_cpu", t1)
                
                gen_str = f"[gen{generation}]" if generation > 0 else ""
                print(f"[Pipeline] Layer {layer_idx}{gen_str} loaded to CPU buffer {target_buffer_idx} ({time.perf_counter()-t1:.3f}s)")
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Pipeline] Error in disk loader: {e}")

    def _schedule_disk_load(self, layer_idx: int, generation: int, target_buffer_idx: int):
        """Schedule a layer to be loaded from disk into a specific CPU buffer."""
        if layer_idx >= self.num_layers:
            return
        
        event = self.cpu_buffer_ready_events[target_buffer_idx]
        event.clear() # Mark this buffer as not ready
        
        try:
            self.disk_to_cpu_queue.put_nowait((layer_idx, generation, target_buffer_idx, event))
            gen_str = f"[gen{generation}]" if generation > 0 else ""
            print(f"[Pipeline] Scheduled disk load for layer {layer_idx}{gen_str} into buffer {target_buffer_idx}")
        except queue.Full:
            print(f"[Pipeline] Disk load queue full, skipping layer {layer_idx}")

    def _transfer_cpu_to_gpu(self, buffer_idx: int):
        """Transfer data from CPU buffer[idx] to GPU buffer[idx] asynchronously."""
        t1 = time.perf_counter()
        cpu_dict = self.cpu_buffers[buffer_idx]
        gpu_dict = self.gpu_buffers[buffer_idx]
        
        with torch.cuda.stream(self.transfer_stream):
            for k, cpu_tensor in cpu_dict.items():
                gpu_dict[k].copy_(cpu_tensor, non_blocking=True)
        
        if self.stats:
            self.stats.set("cpu_to_gpu", t1)
        
        layer_idx, generation = self.buffer_status.get(buffer_idx, (-1, -1))
        gen_str = f"[gen{generation}]" if generation > 0 else ""
        print(f"[Pipeline] Layer {layer_idx}{gen_str} transfer CPU->GPU initiated for buffer {buffer_idx} ({time.perf_counter()-t1:.3f}s)")

    def initialize_pipeline(self, start_layer: int = 0):
        """Initialize the pipeline by pre-loading the first few layers into buffers."""
        print(f"[Pipeline] Initializing from layer {start_layer}")
        self.start()
        self.reset_generation_counter()
        
        # Synchronously load and transfer the first two layers to prime the pipeline
        for i in range(2):
            layer_idx = start_layer + i
            if layer_idx < self.num_layers:
                base = f"model.layers.{layer_idx}."
                self.loader.preload_layer_safetensors(base)
                
                # Load to CPU buffer i, then transfer to GPU buffer i
                temp_dict = self.loader.load_dict_from_disk(base, 'cpu')
                for k, t in temp_dict.items():
                    self.cpu_buffers[i][k].copy_(t)
                
                self.buffer_status[i] = (layer_idx, 0)
                self._transfer_cpu_to_gpu(i)
                print(f"[Pipeline] Layer {layer_idx} pre-loaded to GPU buffer {i}")
        
        # Wait for initial transfers to complete before starting generation
        self.transfer_stream.synchronize()
        
        # Schedule async loads for the next two layers
        if start_layer + 2 < self.num_layers:
            self._schedule_disk_load(start_layer + 2, 0, 0) # Load L2 into CPU buffer 0
        if start_layer + 3 < self.num_layers:
            self._schedule_disk_load(start_layer + 3, 0, 1) # Load L3 into CPU buffer 1

    def get_layer_and_advance(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """Get layer weights for execution and advance the pipeline using double buffering."""
        t_total = time.perf_counter()
        
        # The buffer for the current computation is self.compute_buffer_idx
        # The buffer for the next layer's transfer is 1 - self.compute_buffer_idx
        current_buffer_idx = self.compute_buffer_idx
        next_buffer_idx = 1 - current_buffer_idx
        
        # Wait for the transfer of the current layer's data to be complete
        torch.cuda.current_stream().wait_stream(self.transfer_stream)
        
        if layer_idx == 0 and self._layers_processed > 0:
            self.current_generation += 1
            print(f"[Pipeline] ★ Starting generation {self.current_generation} (cyclic continuation)")
        
        # --- Stage 1: Get current layer for GPU execution ---
        current_gen = self.current_generation
        expected_key = (layer_idx, current_gen)
        if self.buffer_status.get(current_buffer_idx) != expected_key:
            print(f"[Pipeline] WARNING: Buffer {current_buffer_idx} mismatch! Expected {expected_key}, found {self.buffer_status.get(current_buffer_idx)}. Sync loading.")
            base = f"model.layers.{layer_idx}."
            sync_dict = self.loader.load_dict_to_cuda(base)
            for k, t in sync_dict.items(): self.gpu_buffers[current_buffer_idx][k].copy_(t)
            self.buffer_status[current_buffer_idx] = expected_key

        current_dict = self.gpu_buffers[current_buffer_idx]
        self._layers_processed += 1
        
        # --- Stage 2: Transfer next layer (N+1) to GPU ---
        next_layer_idx = layer_idx + 1
        if next_layer_idx < self.num_layers:
            # Wait for the CPU buffer for the next layer to be ready
            if not self.cpu_buffer_ready_events[next_buffer_idx].wait(timeout=10.0):
                print(f"[Pipeline] Timeout waiting for CPU buffer {next_buffer_idx} for layer {next_layer_idx}. May cause stalls.")
            self._transfer_cpu_to_gpu(next_buffer_idx)

        # --- Stage 3: Load layer (N+2) from Disk to CPU ---
        next_next_layer_idx = layer_idx + 2
        if next_next_layer_idx < self.num_layers:
            # Schedule load into the buffer we just finished computing on
            self._schedule_disk_load(next_next_layer_idx, current_gen, current_buffer_idx)
        
        # --- Cyclic Continuation Logic ---
        if self.cyclic_enabled and layer_idx == self.num_layers - 1:
            print(f"[Pipeline] ★ Last layer of gen{current_gen}, preparing gen{current_gen + 1}")
            next_gen = current_gen + 1
            # Schedule load for L0 of next gen into the buffer for N+1
            self._schedule_disk_load(0, next_gen, next_buffer_idx)
            # Schedule load for L1 of next gen into the buffer for N+2
            self._schedule_disk_load(1, next_gen, current_buffer_idx)

        # --- Advance Buffer Pointer ---
        # The next computation will use the buffer we just started transferring into
        self.compute_buffer_idx = next_buffer_idx
        
        if self.stats:
            self.stats.set("pipeline_advance", t_total)
        
        return current_dict

    def reset_generation_counter(self):
        """Reset generation counter and buffer states for a new sequence."""
        self.current_generation = 0
        self._layers_processed = 0
        self.compute_buffer_idx = 0
        self.buffer_status.clear()
        self.cpu_buffer_ready_events[0].clear()
        self.cpu_buffer_ready_events[1].clear()
        print("[Pipeline] Generation counter and buffer states reset.")

    def cleanup(self):
        """Clean up resources."""
        self.stop()
        # Buffers are managed by the object's lifecycle, no manual deletion needed
        print("[Pipeline] Cleanup complete.")