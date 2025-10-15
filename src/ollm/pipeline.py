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
        
        # Use a dictionary for events to avoid race conditions between generations.
        self.cpu_layer_ready_events = {}
        self._events_lock = threading.Lock()
        
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
                
                layer_idx, generation, event = task
                key = (layer_idx, generation)
                base = f"model.layers.{layer_idx}."
                
                t1 = time.perf_counter()
                self.loader.preload_layer_safetensors(base)
                layer_dict = self.loader.load_dict_from_disk(base, device='cpu')
                
                if self.use_pinned_memory:
                    pinned_dict = {}
                    for k, tensor in layer_dict.items():
                        try:
                            pinned_tensor = torch.empty_like(tensor, pin_memory=True)
                            pinned_tensor.copy_(tensor)
                            pinned_dict[k] = pinned_tensor
                        except Exception:
                            pinned_dict[k] = tensor
                    layer_dict = pinned_dict
                
                self.cpu_layers[key] = layer_dict
                event.set()
                
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
        
        with self._events_lock:
            if key in self.cpu_layer_ready_events:
                return
            event = threading.Event()
            self.cpu_layer_ready_events[key] = event
        
        try:
            self.disk_to_cpu_queue.put_nowait((layer_idx, generation, event))
            gen_str = f"[gen{generation}]" if generation > 0 else ""
            print(f"[Pipeline] Scheduled disk load for layer {layer_idx}{gen_str}")
        except queue.Full:
            print(f"[Pipeline] Disk load queue full, skipping layer {layer_idx}")
            with self._events_lock:
                if key in self.cpu_layer_ready_events:
                    del self.cpu_layer_ready_events[key]
    
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
        self.reset_generation_counter()
        
        for i in range(2):
            layer_idx = start_layer + i
            if layer_idx < self.num_layers:
                base = f"model.layers.{layer_idx}."
                self.loader.preload_layer_safetensors(base)
                self.gpu_layers[(layer_idx, 0)] = self.loader.load_dict_to_cuda(base)
                print(f"[Pipeline] Layer {layer_idx} pre-loaded to GPU")
        
        if start_layer + 2 < self.num_layers:
            layer_idx = start_layer + 2
            key = (layer_idx, 0)
            self._schedule_disk_load(layer_idx, 0)
            
            with self._events_lock:
                event = self.cpu_layer_ready_events.get(key)
            
            if event:
                if not event.wait(timeout=15.0):
                    raise RuntimeError(f"Pipeline initialization failed: Timed out waiting for layer {layer_idx} to load to CPU.")
                print(f"[Pipeline] Layer {layer_idx} pre-loaded to CPU")
            else:
                raise RuntimeError(f"Failed to schedule initial load for layer {layer_idx}")

        if start_layer + 3 < self.num_layers:
            self._schedule_disk_load(start_layer + 3, 0)
        
    def get_layer_and_advance(self, layer_idx: int) -> Optional[Dict[str, torch.Tensor]]:
        """
        Get layer weights for execution and advance the pipeline.
        """
        t_total = time.perf_counter()
        
        torch.cuda.current_stream().wait_stream(self.transfer_stream)
        
        if layer_idx == 0 and self._layers_processed > 0:
            self.current_generation += 1
            print(f"[Pipeline] ★ Starting generation {self.current_generation} (cyclic continuation)")
        
        current_gen = self.current_generation
        key = (layer_idx, current_gen)
        
        if key not in self.gpu_layers:
            print(f"[Pipeline] WARNING: Layer {layer_idx}[gen{current_gen}] not in GPU, loading synchronously")
            base = f"model.layers.{layer_idx}."
            self.loader.preload_layer_safetensors(base)
            self.gpu_layers[key] = self.loader.load_dict_to_cuda(base)
        
        current_dict = self.gpu_layers[key]
        self._layers_processed += 1
        
        next_layer = layer_idx + 1
        next_next_layer = layer_idx + 2
        
        if next_layer < self.num_layers:
            next_key = (next_layer, current_gen)
            if next_key not in self.gpu_layers:
                with self._events_lock:
                    event = self.cpu_layer_ready_events.get(next_key)
                
                if event and event.wait(timeout=10.0):
                    if next_key in self.cpu_layers:
                        gpu_dict = self._transfer_cpu_to_gpu(next_layer, current_gen)
                        if gpu_dict: self.gpu_layers[next_key] = gpu_dict
                else:
                    print(f"[Pipeline] Layer {next_layer} not ready, loading directly to GPU")
                    base = f"model.layers.{next_layer}."
                    self.loader.preload_layer_safetensors(base)
                    self.gpu_layers[next_key] = self.loader.load_dict_to_cuda(base)
        
        if next_next_layer < self.num_layers:
            self._schedule_disk_load(next_next_layer, current_gen)
        
        if self.cyclic_enabled and layer_idx >= self.num_layers - 3:
            next_gen = current_gen + 1
            layers_from_end = self.num_layers - layer_idx
            next_gen_target = max(0, 3 - layers_from_end)
            
            for next_gen_layer in range(self.next_gen_layer_idx, next_gen_target + 1):
                if next_gen_layer < self.num_layers:
                    self._schedule_disk_load(next_gen_layer, next_gen)
                    self.next_gen_layer_idx = next_gen_layer + 1
            
            # --- START: FIX FOR CYCLIC GPU TRANSFER ---
            if layer_idx == self.num_layers - 1:
                print(f"[Pipeline] ★ Last layer of gen{current_gen}, preparing gen{next_gen}")
                self.next_gen_layer_idx = 0
                
                # Proactively transfer the first two layers of the next generation to the GPU
                # to ensure they are ready when the next token processing starts.
                for i in range(2):
                    next_gen_layer_idx = i
                    next_gen_key = (next_gen_layer_idx, next_gen)

                    if next_gen_key in self.gpu_layers: continue

                    with self._events_lock:
                        event = self.cpu_layer_ready_events.get(next_gen_key)

                    if event and event.wait(timeout=10.0):
                        if next_gen_key in self.cpu_layers:
                            gpu_dict = self._transfer_cpu_to_gpu(next_gen_layer_idx, next_gen)
                            if gpu_dict:
                                self.gpu_layers[next_gen_key] = gpu_dict
                                print(f"[Pipeline] ★ Proactively transferred layer {next_gen_layer_idx}[gen{next_gen}] to GPU")
                        else:
                            print(f"[Pipeline] WARNING: Event for {next_gen_key} was set, but layer not in CPU cache.")
                    else:
                        print(f"[Pipeline] WARNING: Could not proactively transfer layer {next_gen_key} to GPU. Timed out waiting for CPU.")
            # --- END: FIX FOR CYCLIC GPU TRANSFER ---

        # Cleanup old layers
        old_layer = layer_idx - 2
        if old_layer >= 0:
            old_key = (old_layer, current_gen)
            if old_key in self.gpu_layers: del self.gpu_layers[old_key]
            if old_key in self.cpu_layers: del self.cpu_layers[old_key]
            with self._events_lock:
                if old_key in self.cpu_layer_ready_events: del self.cpu_layer_ready_events[old_key]
        
        if current_gen > 1:
            very_old_gen = current_gen - 2
            keys_to_delete = [k for k in list(self.gpu_layers.keys()) + list(self.cpu_layers.keys()) 
                             if k[1] <= very_old_gen]
            for k in keys_to_delete:
                if k in self.gpu_layers: del self.gpu_layers[k]
                if k in self.cpu_layers: del self.cpu_layers[k]
            with self._events_lock:
                event_keys_to_delete = [k for k in self.cpu_layer_ready_events if k[1] <= very_old_gen]
                for k in event_keys_to_delete:
                    del self.cpu_layer_ready_events[k]
        
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
        with self._events_lock:
            self.cpu_layer_ready_events.clear()
        print("[Pipeline] Cleanup complete")