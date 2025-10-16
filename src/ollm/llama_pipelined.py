# llama3-1B/3B/8B-chat with pipelined loading and cyclic continuation

import time
from datetime import datetime
import torch
from torch import nn
from typing import Optional, Tuple, Union, Dict, Any, Unpack
from .utils import _walk_to_parent, _assign_tensor_to_module, _set_meta_placeholder
from .pipeline import LayerPipeline

# shared objects
loader, stats, pipeline = None, None, None

#======== rewriting core classes (tested on transformers==4.52.3) ==============
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb, eager_attention_forward, LlamaForCausalLM, 
    LlamaAttention, LlamaMLP, LlamaDecoderLayer, LlamaModel, LlamaConfig, 
    create_causal_mask, Cache
)
from transformers.modeling_outputs import BaseModelOutputWithPast

class MyLlamaMLP(LlamaMLP):
    def forward(self, x):
        chunk_size, chunks = 16384, []
        x = x.squeeze(0)
        for i in range(0, x.shape[0], chunk_size):
            gate_chunk = self.act_fn(self.gate_proj(x[i:i+chunk_size]))
            up_chunk = self.up_proj(x[i:i+chunk_size])
            out_chunk = self.down_proj(gate_chunk * up_chunk)
            chunks.append(out_chunk)
        down_proj = torch.cat(chunks, dim=0).unsqueeze(0)
        return down_proj


class PipelinedLoaderLayer:
    """Uses the pipeline to load layer weights"""
    
    def _load_layer_weights_pipelined(self):
        """Load layer weights using the pipeline"""
        global pipeline
        
        if pipeline is None:
            # Fallback to non-pipelined loading
            t1 = time.perf_counter()
            base = f"model.layers.{self.layer_idx}."
            loader.preload_layer_safetensors(base)
            d = loader.load_dict_to_cuda(base)
            if stats:
                stats.set("layer_load", t1)
        else:
            # Use pipeline to get layer weights
            d = pipeline.get_layer_and_advance(self.layer_idx)
        
        # Assign tensors to module
        for attr_path, tensor in d.items():
            parent, leaf = _walk_to_parent(self, attr_path)
            _assign_tensor_to_module(parent, leaf, tensor)
    
    def _unload_layer_weights(self):
        """Replace loaded weights with meta placeholders"""
        base = f"model.layers.{self.layer_idx}."
        for attr_path in loader.manifest[base]:
            parent, leaf = _walk_to_parent(self, attr_path)
            _set_meta_placeholder(parent, leaf)


class MyLlamaDecoderLayer(LlamaDecoderLayer, PipelinedLoaderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        self.layer_idx = layer_idx
        super().__init__(config, layer_idx)
    
    def forward(self, *args, **kwargs):
        self._load_layer_weights_pipelined()
        out = super().forward(*args, **kwargs)
        self._unload_layer_weights()
        return out


class MyLlamaModel(LlamaModel):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList()
        for layer_idx in range(config.num_hidden_layers):
            self.layers.append(MyLlamaDecoderLayer(config, layer_idx))
            self.layers[-1]._unload_layer_weights()
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        
        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)
        
        if use_cache and past_key_values is None:
            from transformers import DynamicCache
            past_key_values = DynamicCache()
        
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        
        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        #============= pipelined execution ==============
        # Store original device of embed/lm_head
        embed_device = self.embed_tokens.weight.device
        lm_head_device = self.parent_lm_head.weight.device
        
        # Only move to CPU if they're on GPU (to free VRAM during layer execution)
        if embed_device.type == 'cuda':
            self.embed_tokens.cpu()
        if lm_head_device.type == 'cuda':
            self.parent_lm_head.cpu()
        
        # Initialize pipeline before first layer
        global pipeline
        if pipeline is not None:
            if not hasattr(pipeline, '_initialized'):
                pipeline.initialize_pipeline(start_layer=0)
                pipeline._initialized = True
            
            # If this is a new sequence (prefill with many tokens), reset generation counter
            # Detect prefill: large cache_position or no past_key_values
            is_prefill = (cache_position.numel() > 1) or (past_key_values is None or past_key_values.get_seq_length() == 0)
            # FIX: Use getattr for safer access on the first run
            if is_prefill and getattr(pipeline, '_last_was_decode', False):
                # Transition from decode to prefill means new sequence
                pipeline.reset_generation_counter()
                print("[Llama] New sequence detected, reset pipeline generation counter")
            pipeline._last_was_decode = not is_prefill
        
        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        
        hidden_states = self.norm(hidden_states)
        
        # Move embed/lm_head back to GPU where hidden_states is
        self.embed_tokens.to(hidden_states.device)
        self.parent_lm_head.to(hidden_states.device)
        
        if stats:
            print("./LlamaPipelined.forward.", datetime.now().strftime("%H:%M:%S"), 
                  stats.print_and_clean() if stats else "")
        #================================================
        
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


# Monkey-patch
import transformers.models.llama.modeling_llama as llama_modeling
llama_modeling.LlamaMLP = MyLlamaMLP
llama_modeling.LlamaDecoderLayer = MyLlamaDecoderLayer
llama_modeling.LlamaModel = MyLlamaModel


class MyLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model.parent_lm_head = self.lm_head
        self.num_hidden_layers = config.num_hidden_layers
    
    def generate(self, **args):
        with torch.no_grad():
            return super().generate(**args)
    
    # CHANGE: Modified to accept buffer counts
    def enable_pipeline(self, enable: bool = True, num_cpu_buffers: int = 2, num_gpu_buffers: int = 2):
        """Enable or disable pipelined loading with configurable buffer counts."""
        global pipeline, loader, stats
        
        if enable:
            if pipeline is None:
                pipeline = LayerPipeline(
                    loader=loader,
                    stats=stats,
                    num_layers=self.num_hidden_layers,
                    num_cpu_buffers=num_cpu_buffers,
                    num_gpu_buffers=num_gpu_buffers
                )
                print(f"[Llama] Pipeline enabled (cyclic mode, {num_cpu_buffers} CPU / {num_gpu_buffers} GPU buffers)")
            else:
                print("[Llama] Pipeline already enabled")
        else:
            self.cleanup_pipeline()
    
    # FIX: Added an explicit cleanup method
    def cleanup_pipeline(self):
        """
        Explicitly stop and clean up the pipeline resources.
        This is the reliable way to prevent zombie threads.
        """
        global pipeline
        if pipeline is not None:
            pipeline.cleanup()
            pipeline = None
            print("[Llama] Pipeline cleaned up and disabled.")

    def __del__(self):
        """
        Attempt to cleanup pipeline on deletion.
        NOTE: This is not reliable. Always call the explicit .cleanup() method
        on the InferencePipelined object before your program exits.
        """
        self.cleanup_pipeline()