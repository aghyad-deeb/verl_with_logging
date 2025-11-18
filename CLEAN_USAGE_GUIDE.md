# ✨ Clean Usage Guide: No Code Changes, No Runtime Patching!

Thanks to HuggingFace's support for custom attention implementations, we can use the triton attention sink kernel with **just a config change**.

## 🎯 The Clean Way

HuggingFace transformers allows:
```python
attn_implementation="path/to/custom/attention"  # Local path
attn_implementation="username/repo-name"         # HF Hub
```

We leverage this to use triton-flash-attn-sink without any code modifications!

## 📦 Quick Setup

### 1. Install Triton Kernel

```bash
cd /tmp
git clone https://huggingface.co/medmekk/triton-flash-attn-sink-clone
cd triton-flash-attn-sink-clone
pip install -e build/torch-universal
```

### 2. Verify Installation

```bash
python -c "from triton_flash_attn_sink import attention; print('✓ Ready!')"
```

### 3. Use in VERL Config

```yaml
actor_rollout_ref:
  model:
    path: meta-llama/Llama-2-7b-hf
    override_config:
      # Use the HF-compatible wrapper we created
      attn_implementation: /data2/Users/aghyad/verl_copy/verl_with_logging/triton_attention_sink_hf
```

That's it! **No plugin, no runtime patching, no code changes!**

## 🎬 Complete Example

### Config File: `config/ppo_triton_sink.yaml`

```yaml
defaults:
  - ppo_trainer
  - _self_

actor_rollout_ref:
  model:
    path: meta-llama/Llama-2-7b-hf
    override_config:
      attn_implementation: /data2/Users/aghyad/verl_copy/verl_with_logging/triton_attention_sink_hf

trainer:
  experiment_name: ppo_with_triton_attention_sink
  total_epochs: 10
  project_name: my_project

data:
  train_files: data/train.parquet
  val_files: data/val.parquet
```

### Run Training:

```bash
python3 -m verl.trainer.main_ppo --config-name ppo_triton_sink
```

## 🔧 Using Relative Paths

If you prefer relative paths:

```yaml
# From your project directory:
override_config:
  attn_implementation: ./triton_attention_sink_hf
```

## ☁️ Optional: Upload to HuggingFace Hub

For easier sharing and deployment:

```bash
# 1. Install HF CLI if not already
pip install huggingface_hub

# 2. Login
huggingface-cli login

# 3. Upload
cd /data2/Users/aghyad/verl_copy/verl_with_logging
huggingface-cli upload your-username/triton-flash-attn-sink ./triton_attention_sink_hf
```

Then use:
```yaml
override_config:
  attn_implementation: your-username/triton-flash-attn-sink
```

## ✅ Verify It's Working

Look for this in your training logs:

```
[TritonFlashAttentionSink] Initialized with learned_sinks=False, bandwidth=0
```

## 📊 Comparison: Old vs New Approach

### ❌ Old Approach (Plugin with Runtime Patching):
```
Install plugin → Runtime monkey patching → Config change → Training
- Complex setup
- Runtime modifications
- Hard to debug
```

### ✅ New Approach (HF-Compatible):
```
Install kernel → Config change → Training
- Simple setup
- No runtime patching
- Standard HF mechanism
```

## 🎯 Why This is Better

1. **✅ No code changes** - VERL stays pristine
2. **✅ No runtime patching** - Uses standard HF mechanism  
3. **✅ Easy to disable** - Just change config back
4. **✅ Standard pattern** - Same as kernels-community/vllm-flash-attn3
5. **✅ Shareable** - Upload to HF Hub for others to use

## 🔄 Switching Between Implementations

```yaml
# Standard flash attention
override_config:
  attn_implementation: flash_attention_2

# Eager (for debugging)
override_config:
  attn_implementation: eager

# Triton attention sink
override_config:
  attn_implementation: /path/to/triton_attention_sink_hf

# Another custom kernel from HF Hub
override_config:
  attn_implementation: kernels-community/vllm-flash-attn3
```

All work the same way - just config changes!

## 🐛 Troubleshooting

**Module not found:**
- Use absolute path or ensure relative path is correct from where you run the command

**Triton kernel not found:**
```bash
cd /tmp/triton-flash-attn-sink-clone
pip install -e build/torch-universal
```

**Still using flash_attention_2?**
- Check your config is being loaded correctly
- Verify the path in override_config is correct

## 📚 Summary

You were absolutely right to question the plugin approach! Using HF's built-in support for custom attention implementations via `attn_implementation` paths is:

- **Cleaner** - No runtime patching
- **Simpler** - Just config changes
- **Standard** - Same as other custom kernels
- **Maintainable** - Clear separation of concerns

This is the proper way to do it! 🎉


