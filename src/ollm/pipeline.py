import os
import time
import threading
import queue
import torch
from typing import Optional, Dict, Any
from datetime import datetime

class LayerPipeline:
    """
    3-stage pipeline for layer loading with CYCLIC continuation:
    - Pipeline continues loading layers for next token while current token executes
    - When layer N finishes, pipeline starts loading layer 0 for next iteration
    - Creates perpetual pipeline that never stops between tokens
    
    Stage 1: GPU executes current layer
    Stage 2: CPU->GPU transfers next layer (async)
    Stage 3: SSD->CPU loads layer after next (async in thread)
    """
    
    def __init__(self, loader, stats=None, num_layers=32):
        self.loader = loader
        self.stats = stats
        self.num_layers = num_layers
        
        # Pipeline buffers - now support multiple "generations" of layers
        # Key format: (layer_idx, generation)
        self.gpu_layers = {}
        self.cpu_layers = {}
        
        # Track which generation we're on (increments after each full pass)
        self.current_generation = 0
        
        # Async loading infrastructure
        self.disk_to_cpu_queue = queue.Queue(maxsize=3)
        self.disk_loader_thread = None
        self.stop_thread = threading.Event()
        
        # Threading events for synchronization (per layer, reused cyclically)
        self.cpu_layer_ready_events = [threading.Event() for _ in range(num_layers)]
        
        # Pinned memory for faster CPU->GPU transfers
        self.use_pinned_memory = True
        
        # CUDA stream for async transfers
        self.transfer_stream = torch.cuda.Stream()
        
        # Cyclic pipeline tracking
        self.cyclic_enabled = True
        self.next_gen_layer_idx = 0  # Track which layer of next generation to load
        
        print(f"[Pipeline] Initialized with {num_layers} layers (cyclic mode enabled)")
    
    def start(self):
        """Start the disk loading thread"""
        if self.disk_loader_thread is None or not self.disk_loader_thread.is_alive():
            self.stop_thread.clear()
            self.disk_loader_thread = threading.Thread(target=self._disk_loader_worker, daemon=True)
            self.disk_loader_thread.start()
            print("[Pipeline] Disk loader thread started")
    
    def stop(self):
        """Stop the disk loading thread"""
        self.stop_thread.set()
        try:
            self.disk_to_cpu_queue.put_nowait(None)
        except queue.Full:
            pass
        if self.disk_loader_thread:
            self.disk_loader_thread.join(timeout=2.0)
        print("[Pipeline] Disk loader thread stopped")
    
    def _disk_loader_worker(self):
        """Background thread that loads layers from disk to CPU"""
        while not self.stop_thread.is_set():
            try:
                task = self.disk_to_cpu_queue.get(timeout=0.1)
                if task is None:
                    break
                
                layer_idx, generation = task
                base = f"model.layers.{layer_idx}."
                
                t1 = time.perf_counter()
                self.loader.preload_layer_safetensors(base)
                layer_dict = self.loader.load_dict_from_disk(base, device='cpu')
                
                if self.use_pinned_memory:
                    pinned_dict = {}
                    for key, tensor in layer_dict.items():
                        try:
                            pinned_tensor = torch.empty_like(tensor, pin_memory=True)
                            pinned_tensor.copy_(tensor)
                            pinned_dict[key] = pinned_tensor
                        except Exception:
                            pinned_dict[key] = tensor
                    layer_dict = pinned_dict
                
                # Store with generation tag
                self.cpu_layers[(layer_idx, generation)] = layer_dict
                
                # Signal that this layer is ready
                self.cpu_layer_ready_events[layer_idx].set()
                
                if self.stats:
                    self.stats.set("disk_to_cpu", t1)
                
                gen_str = f"[gen{generation}]" if generation > 0 else ""
                print(f"[Pipeline] Layer {layer_idx}{gen_str} loaded to CPU ({time.perf_counter()-t1:.3f}s)")
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Pipeline] Error in disk loader: {e}")
    
    def _schedule_disk_load(self, layer_idx: int, generation: int):
        """Schedule a layer to be loaded from disk to CPU"""
        if layer_idx >= self.num_layers:
            return
        
        key = (layer_idx, generation)
        if key in self.cpu_layers or key in self.gpu_layers:
            return
        
        try:
            # Reset the event before scheduling
            self.cpu_layer_ready_events[layer_idx].clear()
            self.disk_to_cpu_queue.put_nowait((layer_idx, generation))
            gen_str = f"[gen{generation}]" if generation > 0 else ""
            print(f"[Pipeline] Scheduled disk load for layer {layer_idx}{gen_str}")
        except queue.Full:
            print(f"[Pipeline] Disk load queue full, skipping layer {layer_idx}")
    
    def _transfer_cpu_to_gpu(self, layer_idx: int, generation: int) -> Optional[Dict[str, torch.Tensor]]:
        """Transfer a layer from CPU to GPU asynchronously"""
        key = (layer_idx, generation)
        if key not in self.cpu_layers:
            return None
        
        t1 = time.perf_counter()
        
        cpu_dict = self.cpu_layers[key]
        gpu_dict = {}
        
        with torch.cuda.stream(self.transfer_stream):
            for k, tensor in cpu_dict.items():
                gpu_dict[k] = tensor.to(self.loader.device, non_blocking=True)
        
        if self.stats:
            self.stats.set("cpu_to_gpu", t1)
        
        gen_str = f"[gen{generation}]" if generation > 0 else ""
        print(f"[Pipeline] Layer {layer_idx}{gen_str} transfer CPU->GPU initiated ({time.perf_counter()-t1:.3f}s)")
        return gpu_dict
    
    def initialize_pipeline(self, start_layer: int = 0):
        """Initialize the pipeline by pre-loading first 3 layers"""
        print(f"[Pipeline] Initializing from layer {start_layer}")
        self.start()
        self.current_generation = 0
        
        # Load first 2 layers directly to GPU
        for i in range(2):
            layer_idx = start_layer + i
            if layer_idx < self.num_layers:
                base = f"model.layers.{layer_idx}."
                self.loader.preload_layer_safetensors(base)
                self.gpu_layers[(layer_idx, 0)] = self.loader.load_dict_to_cuda(base)
                print(f"[Pipeline] Layer {layer_idx} pre-loaded to GPU")
        
        # Load third layer to CPU synchronously
        if start_layer + 2 < self.num_layers:
            layer_idx = start_layer + 2
            self.disk_to_cpu_queue.put((layer_idx, 0))
            self.cpu_layer_ready_events[layer_idx].wait(timeout=10.0)
            if not self.cpu_layer_ready_events[layer_idx].is_set():
                print(f"[Pipeline] WARNING: Initial CPU load for layer {layer_idx} timed out!")
            else:
                print(f"[Pipeline] Layer {layer_idx} pre-loaded to CPU")
        
        # Schedule fourth layer for background loading
        if start_layer + 3 < self.num_layers:
            self._schedule_disk_load(start_layer + 3, 0)
        
        # Initialize next generation tracking
        self.next_gen_layer_idx = 0
    
    def get_layer_and_advance(self, layer_idx: int) -> Optional[Dict[str, torch.Tensor]]:
        """
        Get layer weights for execution and advance the pipeline.
        
        CYCLIC FEATURE: When we reach the last layers, we start pre-loading
        the first layers of the next generation (next token).
        """
        t_total = time.perf_counter()
        
        # Wait for any pending transfers to complete
        torch.cuda.current_stream().wait_stream(self.transfer_stream)
        
        # Determine current generation based on layer_idx wrapping
        # If layer_idx == 0, we might be starting a new generation
        if layer_idx == 0 and hasattr(self, '_layers_processed') and self._layers_processed > 0:
            self.current_generation += 1
            print(f"[Pipeline] ★ Starting generation {self.current_generation} (cyclic continuation)")
        
        if not hasattr(self, '_layers_processed'):
            self._layers_processed = 0
        
        current_gen = self.current_generation
        key = (layer_idx, current_gen)
        
        # Get current layer from GPU
        if key not in self.gpu_layers:
            print(f"[Pipeline] WARNING: Layer {layer_idx}[gen{current_gen}] not in GPU, loading synchronously")
            base = f"model.layers.{layer_idx}."
            self.loader.preload_layer_safetensors(base)
            layer_dict = self.loader.load_dict_to_cuda(base)
            self.gpu_layers[key] = layer_dict
        
        current_dict = self.gpu_layers[key]
        self._layers_processed += 1
        
        # Advance pipeline for next layer in current generation
        next_layer = layer_idx + 1
        next_next_layer = layer_idx + 2
        
        # Stage 2: Start CPU->GPU transfer for next layer in current generation
        if next_layer < self.num_layers:
            next_key = (next_layer, current_gen)
            if next_key not in self.gpu_layers:
                max_wait = 10.0
                is_ready = self.cpu_layer_ready_events[next_layer].wait(timeout=max_wait)
                
                if is_ready and next_key in self.cpu_layers:
                    gpu_dict = self._transfer_cpu_to_gpu(next_layer, current_gen)
                    if gpu_dict:
                        self.gpu_layers[next_key] = gpu_dict
                else:
                    # Fallback
                    print(f"[Pipeline] Layer {next_layer} not ready, loading directly to GPU")
                    base = f"model.layers.{next_layer}."
                    self.loader.preload_layer_safetensors(base)
                    self.gpu_layers[next_key] = self.loader.load_dict_to_cuda(base)
        
        # Stage 3: Schedule disk->CPU load for layer after next
        if next_next_layer < self.num_layers:
            self._schedule_disk_load(next_next_layer, current_gen)
        
        # ★ CYCLIC CONTINUATION: Start loading next generation layers
        if self.cyclic_enabled:
            # When we're near the end of current generation, start loading next generation
            # Start when we're at layer (num_layers - 3) or later
            if layer_idx >= self.num_layers - 3:
                next_gen = current_gen + 1
                
                # Calculate which layer of next generation to pre-load
                # We want to stay ~3 layers ahead
                layers_from_end = self.num_layers - layer_idx
                next_gen_target = max(0, 3 - layers_from_end)
                
                # Schedule next generation layers
                for next_gen_layer in range(self.next_gen_layer_idx, next_gen_target + 1):
                    if next_gen_layer < self.num_layers:
                        next_gen_key = (next_gen_layer, next_gen)
                        if next_gen_key not in self.cpu_layers and next_gen_key not in self.gpu_layers:
                            self._schedule_disk_load(next_gen_layer, next_gen)
                            self.next_gen_layer_idx = next_gen_layer + 1
                
                # If we're at the last layer, prepare first 2 layers of next gen for GPU
                if layer_idx == self.num_layers - 1:
                    print(f"[Pipeline] ★ Last layer of gen{current_gen}, preparing gen{next_gen}")
                    self.next_gen_layer_idx = 0  # Reset for next cycle
        
        # Cleanup: Free old layers
        # Keep current and next layer, clean up layer_idx - 2
        old_layer = layer_idx - 2
        if old_layer >= 0:
            old_key = (old_layer, current_gen)
            if old_key in self.gpu_layers:
                del self.gpu_layers[old_key]
            if old_key in self.cpu_layers:
                del self.cpu_layers[old_key]
        
        # Also cleanup very old generations (gen < current_gen - 1)
        if current_gen > 1:
            very_old_gen = current_gen - 2
            keys_to_delete = [k for k in list(self.gpu_layers.keys()) + list(self.cpu_layers.keys()) 
                             if k[1] <= very_old_gen]
            for k in keys_to_delete:
                if k in self.gpu_layers:
                    del self.gpu_layers[k]
                if k in self.cpu_layers:
                    del self.cpu_layers[k]
        
        if self.stats:
            self.stats.set("pipeline_advance", t_total)
        
        return current_dict
    
    def reset_generation_counter(self):
        """Reset generation counter (useful when starting new sequence)"""
        self.current_generation = 0
        self.next_gen_layer_idx = 0
        self._layers_processed = 0
        print("[Pipeline] Generation counter reset")
    
    def cleanup(self):
        """Clean up resources"""
        self.stop()
        self.gpu_layers.clear()
        self.cpu_layers.clear()
        for event in self.cpu_layer_ready_events:
            event.clear()
        print("[Pipeline] Cleanup complete")