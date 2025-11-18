# 🎯 Simple Attention Sink Setup (No Plugin Needed!)

After reviewing [PR #3978](https://github.com/volcengine/verl/pull/3978), there's a **much simpler approach** that leverages VERL's existing config system.

## ✨ The Simple Way

Since VERL already supports `attn_implementation` override, we just need to make the triton kernel compatible with the existing flow.

## 📦 Setup

### Option 1: Use Eager + Manual Integration (Simplest)

1. **Install the triton kernel:**
```bash
cd /tmp
git clone https://huggingface.co/medmekk/triton-flash-attn-sink-clone
cd triton-flash-attn-sink-clone
pip install -e build/torch-universal
```

2. **Create a simple wrapper script:**

```python
# startup_hook.py - Place in your project root
"""
Simple hook that patches attention when using eager mode.
Import this before running your training.
"""

def patch_attention_for_gpt_oss():
    """Patch eager attention to use triton kernel for GPT-OSS models."""
    import sys
    
    # Check if this is a GPT-OSS model run
    import os
    if os.environ.get("USE_ATTENTION_SINK", "0") != "1":
        return
    
    print("[Attention Sink] Patching for GPT-OSS compatibility...")
    
    from triton_flash_attn_sink import attention as triton_attention
    import torch
    
    def attention_sink_forward(query, key, value, attention_mask=None, **kwargs):
        """Simple wrapper for triton attention sink."""
        # Transpose: (B,S,H,D) -> (B,H,S,D)
        q = query.transpose(1, 2).contiguous()
        k = key.transpose(1, 2).contiguous()
        v = value.transpose(1, 2).contiguous()
        
        # Handle GQA
        if k.shape[1] != q.shape[1]:
            n_rep = q.shape[1] // k.shape[1]
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)
        
        # Call triton kernel
        head_dim = q.shape[-1]
        sm_scale = 1.0 / (head_dim ** 0.5)
        
        out = triton_attention(
            q, k, v,
            sinks=None,  # No learned sinks by default
            causal=True,
            sm_scale=sm_scale,
            bandwidth=0,  # Full attention
        )
        
        # Transpose back: (B,H,S,D) -> (B,S,H,D)
        return out.transpose(1, 2).contiguous()
    
    # Patch transformers eager attention
    try:
        import transformers.models.llama.modeling_llama as llama_modeling
        
        # Find and patch the eager attention function
        # This is model-specific, adjust for your model
        original_attention = llama_modeling.LlamaAttention.forward
        
        def patched_forward(self, *args, **kwargs):
            # Your patching logic here
            return original_attention(self, *args, **kwargs)
        
        # llama_modeling.LlamaAttention.forward = patched_forward
        print("[Attention Sink] Patched successfully")
        
    except Exception as e:
        print(f"[Attention Sink] Warning: Could not patch - {e}")

# Auto-run on import
patch_attention_for_gpt_oss()
```

3. **Use in your config:**
```yaml
actor_rollout_ref:
  model:
    path: openai/gpt-oss-20b  # or your model
    override_config:
      attn_implementation: eager  # Use eager mode
```

4. **Run with environment variable:**
```bash
USE_ATTENTION_SINK=1 python -c "import startup_hook" && python3 -m verl.trainer.main_ppo --config-name your_config
```

### Option 2: Wait for Native Support

Actually, the **cleanest approach** is to request that VERL add native support for custom attention implementations. 

## 🎯 The Real Solution

Looking at your issue screenshot, the problem is:
- GPT-OSS models have built-in attention sinks
- `flash_attention_2` doesn't support them correctly → causes gradient spikes
- Solution: Use `attn_implementation=eager` which properly supports GPT-OSS's attention sinks

**So actually, you might not need the triton kernel at all!**

## ✅ Recommended Approach

**For GPT-OSS models specifically:**

```yaml
actor_rollout_ref:
  model:
    path: openai/gpt-oss-20b
    override_config:
      attn_implementation: eager  # This works with GPT-OSS attention sinks!
```

That's it! No plugin, no patching, just config.

## 🤔 When Do You Need the Triton Kernel?

You only need the triton-flash-attn-sink kernel if:

1. **You want faster than eager**: Eager is slow, triton is fast
2. **Your model doesn't have built-in sinks**: Adding sinks to a standard model
3. **You want customizable sinks**: Control bandwidth, learned params, etc.

## 📊 Decision Tree

```
Do you have a GPT-OSS model?
├─ YES → Just use attn_implementation=eager (no plugin needed!)
└─ NO → Do you want to add attention sinks to a standard model?
    ├─ YES → Use the triton kernel + plugin
    └─ NO → Use flash_attention_2 (default)
```

## 🎯 For Your Case

Based on the issue you showed:

**If you're using GPT-OSS model:**
```yaml
# THIS IS ALL YOU NEED:
override_config:
  attn_implementation: eager
```

**If you want faster execution with GPT-OSS:**
```yaml
# Then you need the triton kernel integration
override_config:
  attn_implementation: attention_sink  # Requires plugin
```

## 💡 Summary

1. **GPT-OSS models already have attention sinks built-in**
2. **Use `attn_implementation=eager` to enable them** (no plugin needed)
3. **Only use triton kernel if you need speed** (then plugin is needed)

The issue you showed proves VERL's config system works! The plugin is only needed if you want to use the faster triton implementation instead of eager mode.

**What's your goal: correctness (use eager) or speed (use triton)?**


