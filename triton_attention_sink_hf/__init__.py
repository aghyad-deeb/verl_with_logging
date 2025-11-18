"""
HuggingFace-compatible Triton Flash Attention with Attention Sinks
Upload to HF Hub and use with: attn_implementation="your-username/triton-flash-attn-sink"
"""

__version__ = "0.1.0"

# This is what HF transformers looks for
from .flash_attention_triton_sink import flash_attention_forward

__all__ = ["flash_attention_forward"]

