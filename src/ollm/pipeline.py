# FILE: pipeline.py

import os
import time
import threading
import queue
import torch
from typing import Optional, Dict, Any, Tuple

class LayerPipeline:
    """
    Generalized, multi-buffered M-stage pipeline for layer loading with CYCLIC continuation.
    This implementation pre-allocates a configurable number of CPU (m) and GPU (n) buffers
    and cycles between them to hide I/O and transfer latency.

    - `m` CPU buffers are used for asynchronous disk-to-CPU loading.
    - `n` GPU buffers are used for asynchronous CPU-to-GPU transfers.

    The pipeline operates on a lookahead principle:
    - While the GPU computes on layer `L` (using GPU buffer `L % n`),
    - The pipeline schedules the transfer of layers `L+1`, `L+2`, ... up to `L+n-1`.
    - Simultaneously, it schedules disk loading for layers `L+n`, `L+n+1`, ... up to `L+n+m-1`.
    """

    def __init__(self, loader, stats=None, num_layers=32, num_cpu_buffers=3, num_gpu_buffers=2):
        self.loader = loader
        self.stats = stats
        self.num_layers = num_layers
        self.num_cpu_buffers = num_cpu_buffers
        self.num_gpu_buffers = num_gpu_buffers

        if not (self.num_cpu_buffers > 0 and self.num_gpu_buffers > 0):
            raise ValueError("Number of CPU and GPU buffers must be greater than 0.")
        if self.num_gpu_buffers > self.num_cpu_buffers:
            raise ValueError(f"num_gpu_buffers ({self.num_gpu_buffers}) cannot be greater than num_cpu_buffers ({self.num_cpu_buffers}).")

        # --- Buffer Management ---
        self.layer_structure = self._get_layer_structure()
        
        print(f"[Pipeline] Allocating {self.num_cpu_buffers} CPU and {self.num_gpu_buffers} GPU buffers...")
        self.cpu_buffers = [self._allocate_buffers(device='cpu', pinned=True) for _ in range(self.num_cpu_buffers)]
        self.gpu_buffers = [self._allocate_buffers(device=self.loader.device, pinned=False) for _ in range(self.num_gpu_buffers)]
        print("[Pipeline] Buffer allocation complete.")

        # State tracking for buffer content: buffer_idx -> (layer_idx, generation)
        self.cpu_buffer_content: Dict[int, Tuple[int, int]] = {}
        self.gpu_buffer_content: Dict[int, Tuple[int, int]] = {}
        
        # --- Async Infrastructure ---
        self.disk_to_cpu_queue = queue.Queue(maxsize=self.num_cpu_buffers * 2)
        self.disk_loader_thread = None
        self.stop_thread = threading.Event()
        
        self.cpu_buffer_ready_events = {i: threading.Event() for i in range(self.num_cpu_buffers)}
        
        self.transfer_stream = torch.cuda.Stream()
        
        # --- State Tracking ---
        self.current_generation = 0
        self._layers_processed = 0
        
        print(f"[Pipeline] Initialized with {num_layers} layers, {self.num_cpu_buffers} CPU buffers, {self.num_gpu_buffers} GPU buffers (cyclic mode).")

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
        """Stop the disk loading thread gracefully."""
        if self.disk_loader_thread is None:
            return
            
        self.stop_thread.set()
        # Unblock the worker thread if it's waiting on the queue
        try:
            self.disk_to_cpu_queue.put_nowait(None)
        except queue.Full:
            # If the queue is full, the worker is busy and will see the stop_thread event soon.
            pass
        
        if self.disk_loader_thread.is_alive():
            self.disk_loader_thread.join(timeout=5.0) # Wait for the thread to finish
            if self.disk_loader_thread.is_alive():
                print("[Pipeline] WARNING: Disk loader thread did not stop within timeout.")
        
        self.disk_loader_thread = None
        print("[Pipeline] Disk loader thread stopped.")


    def _disk_loader_worker(self):
        """Background thread that loads layers from disk into a pre-allocated CPU buffer."""
        while not self.stop_thread.is_set():
            try:
                task = self.disk_to_cpu_queue.get(timeout=0.1)
                if task is None: # Sentinel value to exit
                    break
                
                layer_idx, generation, target_buffer_idx, event = task
                base = f"model.layers.{layer_idx}."
                
                t1 = time.perf_counter()
                self.loader.preload_layer_safetensors(base)
                
                layer_dict_temp = self.loader.load_dict_from_disk(base, device='cpu')
                
                target_cpu_buffer = self.cpu_buffers[target_buffer_idx]
                for k, temp_tensor in layer_dict_temp.items():
                    target_cpu_buffer[k].copy_(temp_tensor)
                
                self.cpu_buffer_content[target_buffer_idx] = (layer_idx, generation)
                event.set()
                
                if self.stats:
                    self.stats.set("disk_to_cpu", t1)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Pipeline] Error in disk loader: {e}")
                self.stop_thread.set() # Stop on error

    def _schedule_disk_load(self, layer_idx: int, generation: int):
        """Schedule a layer to be loaded from disk into its designated CPU buffer."""
        target_buffer_idx = layer_idx % self.num_cpu_buffers
        event = self.cpu_buffer_ready_events[target_buffer_idx]
        
        if self.cpu_buffer_content.get(target_buffer_idx) == (layer_idx, generation):
            return

        event.clear()
        try:
            self.disk_to_cpu_queue.put_nowait((layer_idx, generation, target_buffer_idx, event))
        except queue.Full:
            print(f"[Pipeline] Disk load queue full, skipping layer {layer_idx}")

    def _transfer_cpu_to_gpu(self, source_cpu_idx: int, target_gpu_idx: int):
        """Transfer data from a specific CPU buffer to a specific GPU buffer asynchronously."""
        t1 = time.perf_counter()
        cpu_dict = self.cpu_buffers[source_cpu_idx]
        gpu_dict = self.gpu_buffers[target_gpu_idx]
        
        with torch.cuda.stream(self.transfer_stream):
            for k, cpu_tensor in cpu_dict.items():
                gpu_dict[k].copy_(cpu_tensor, non_blocking=True)
        
        self.gpu_buffer_content[target_gpu_idx] = self.cpu_buffer_content.get(source_cpu_idx, (-1, -1))
        
        if self.stats:
            self.stats.set("cpu_to_gpu", t1)

    def initialize_pipeline(self, start_layer: int = 0):
        """Robustly initialize the pipeline by fully populating all buffers."""
        print(f"[Pipeline] Initializing from layer {start_layer}")
        self.start()
        self.reset_generation_counter()
        
        # 1. Schedule disk loads to fill all CPU buffers
        for i in range(self.num_cpu_buffers):
            layer_idx = start_layer + i
            if layer_idx < self.num_layers:
                self._schedule_disk_load(layer_idx, 0)

        # 2. Wait for CPU loads and schedule transfers to fill all GPU buffers
        for i in range(self.num_gpu_buffers):
            layer_idx = start_layer + i
            if layer_idx < self.num_layers:
                cpu_idx = layer_idx % self.num_cpu_buffers
                gpu_idx = layer_idx % self.num_gpu_buffers
                if not self.cpu_buffer_ready_events[cpu_idx].wait(timeout=20.0):
                     raise TimeoutError(f"Timeout waiting for initial CPU load of layer {layer_idx} into buffer {cpu_idx}")
                self._transfer_cpu_to_gpu(cpu_idx, gpu_idx)
                print(f"[Pipeline] Layer {layer_idx} pre-loaded to GPU buffer {gpu_idx}")

        self.transfer_stream.synchronize()
        print("[Pipeline] Initial pre-loading and transfers complete.")

    def get_layer_and_advance(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """Get layer weights for execution and advance the pipeline."""
        t_total = time.perf_counter()
        
        torch.cuda.current_stream().wait_stream(self.transfer_stream)
        
        if layer_idx == 0 and self._layers_processed > 0:
            self.current_generation += 1
            print(f"[Pipeline] ★ Starting generation {self.current_generation} (cyclic continuation)")
        
        current_gen = self.current_generation
        
        # --- Stage 1: Get current layer for GPU execution ---
        current_gpu_idx = layer_idx % self.num_gpu_buffers
        expected_key = (layer_idx, current_gen)
        
        if self.gpu_buffer_content.get(current_gpu_idx) != expected_key:
            print(f"[Pipeline] WARNING: Buffer {current_gpu_idx} mismatch! Expected {expected_key}, found {self.gpu_buffer_content.get(current_gpu_idx)}. Sync loading.")
            base = f"model.layers.{layer_idx}."
            sync_dict = self.loader.load_dict_to_cuda(base)
            for k, t in sync_dict.items(): self.gpu_buffers[current_gpu_idx][k].copy_(t)
            self.gpu_buffer_content[current_gpu_idx] = expected_key

        current_dict = self.gpu_buffers[current_gpu_idx]
        self._layers_processed += 1
        
        # --- FIX: Unified scheduling logic using modulo ---
        # This logic seamlessly handles both regular lookahead and cyclic continuation.

        # --- Stage 2: Schedule transfer for the *next* layer to be computed ---
        next_layer_to_compute_idx = (layer_idx + 1) % self.num_layers
        next_compute_gen = current_gen + 1 if layer_idx == self.num_layers - 1 else current_gen
        
        source_cpu_idx = next_layer_to_compute_idx % self.num_cpu_buffers
        target_gpu_idx = next_layer_to_compute_idx % self.num_gpu_buffers
        
        if not self.cpu_buffer_ready_events[source_cpu_idx].wait(timeout=10.0):
            print(f"[Pipeline] Timeout waiting for CPU buffer {source_cpu_idx} for layer {next_layer_to_compute_idx}. May cause stalls.")
        
        # Only transfer if the GPU buffer doesn't already have the correct content
        if self.gpu_buffer_content.get(target_gpu_idx) != (next_layer_to_compute_idx, next_compute_gen):
            self._transfer_cpu_to_gpu(source_cpu_idx, target_gpu_idx)

        # --- Stage 3: Schedule disk load for a future layer to keep the CPU buffers full ---
        layer_to_load_idx = (layer_idx + self.num_gpu_buffers) % self.num_layers
        load_gen = current_gen + 1 if layer_idx + self.num_gpu_buffers >= self.num_layers else current_gen
        self._schedule_disk_load(layer_to_load_idx, load_gen)
        
        if self.stats:
            self.stats.set("pipeline_advance", t_total)
        
        return current_dict

    def reset_generation_counter(self):
        """Reset generation counter and buffer states for a new sequence."""
        self.current_generation = 0
        self._layers_processed = 0
        self.cpu_buffer_content.clear()
        self.gpu_buffer_content.clear()
        for event in self.cpu_buffer_ready_events.values():
            event.clear()
        print("[Pipeline] Generation counter and buffer states reset.")

    def cleanup(self):
        """Clean up resources."""
        self.stop()
        print("[Pipeline] Cleanup complete.")