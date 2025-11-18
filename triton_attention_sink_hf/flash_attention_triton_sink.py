"""
HuggingFace-compatible attention implementation using triton-flash-attn-sink kernel.
This follows the HF attention implementation interface.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn


class TritonFlashAttentionSink:
    """
    HuggingFace-compatible attention implementation using triton-flash-attn-sink.
    
    Can be used with:
        model = AutoModel.from_pretrained(
            "model_name",
            attn_implementation="path/to/triton_attention_sink_hf"
        )
    
    Or in VERL config:
        override_config:
          attn_implementation: "path/to/triton_attention_sink_hf"
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize the attention implementation."""
        self._attention_fn = None
        self._config = {}
        
        # Extract configuration from kwargs if provided
        self.enable_learned_sinks = kwargs.pop("enable_learned_sinks", False)
        self.bandwidth = kwargs.pop("bandwidth", 0)
        self.sink_init_value = kwargs.pop("sink_init_value", 0.0)
        
        print(f"[TritonFlashAttentionSink] Initialized with "
              f"learned_sinks={self.enable_learned_sinks}, bandwidth={self.bandwidth}")
    
    def _get_attention_fn(self):
        """Lazy load the triton attention function."""
        if self._attention_fn is None:
            try:
                from triton_flash_attn_sink import attention
                self._attention_fn = attention
            except ImportError as e:
                raise ImportError(
                    "triton_flash_attn_sink is not installed. Install with:\n"
                    "git clone https://huggingface.co/medmekk/triton-flash-attn-sink-clone\n"
                    "cd triton-flash-attn-sink-clone\n"
                    "pip install -e build/torch-universal"
                ) from e
        return self._attention_fn
    
    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass compatible with HuggingFace attention interface.
        
        Args:
            query: (batch, seq_len, num_heads, head_dim)
            key: (batch, seq_len, num_key_value_heads, head_dim)
            value: (batch, seq_len, num_key_value_heads, head_dim)
            attention_mask: Optional attention mask
            
        Returns:
            Attention output (batch, seq_len, num_heads, head_dim)
        """
        attention_fn = self._get_attention_fn()
        
        # Transpose to (batch, num_heads, seq_len, head_dim) for triton
        q = query.transpose(1, 2).contiguous()
        k = key.transpose(1, 2).contiguous()
        v = value.transpose(1, 2).contiguous()
        
        # Handle GQA/MQA: repeat key/value heads if needed
        if k.shape[1] != q.shape[1]:
            n_rep = q.shape[1] // k.shape[1]
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)
        
        # Compute softmax scale
        head_dim = q.shape[-1]
        softmax_scale = kwargs.get("softmax_scale", 1.0 / (head_dim ** 0.5))
        
        # Get sinks (None for now, can be extended to support learned sinks)
        sinks = None
        if self.enable_learned_sinks:
            # TODO: Implement learned sinks management
            # For now, we use None (no sinks)
            pass
        
        # Determine causality
        is_causal = kwargs.get("is_causal", True)
        
        # Call triton kernel (must use positional args, not kwargs)
        attn_output = attention_fn(
            q, k, v, sinks,
            is_causal,  # causal
            softmax_scale,  # sm_scale
            self.bandwidth,  # bandwidth
        )
        
        # Transpose back to (batch, seq_len, num_heads, head_dim)
        return attn_output.transpose(1, 2).contiguous()


# HuggingFace looks for this function signature
def flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    query_length: int = None,
    dropout: float = 0.0,
    softmax_scale: Optional[float] = None,
    is_causal: bool = True,
    **kwargs,
) -> torch.Tensor:
    """
    HuggingFace-compatible flash attention forward function.
    This is the interface that HF transformers expects.
    """
    # Create an instance (or use a cached one)
    impl = TritonFlashAttentionSink()
    
    return impl(
        query=query_states,
        key=key_states,
        value=value_states,
        attention_mask=attention_mask,
        softmax_scale=softmax_scale,
        is_causal=is_causal,
        **kwargs,
    )


# Export the main function for HF compatibility
__all__ = ["TritonFlashAttentionSink", "flash_attention_forward"]

