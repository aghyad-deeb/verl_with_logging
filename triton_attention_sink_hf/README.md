# Triton Flash Attention Sink - HuggingFace Compatible

A HuggingFace-compatible attention implementation using the triton-flash-attn-sink kernel.

## 🚀 Usage

### Option 1: Local Path (Recommended for Development)

```yaml
# In your VERL config:
actor_rollout_ref:
  model:
    override_config:
      attn_implementation: /absolute/path/to/triton_attention_sink_hf
```

or with relative path from your project:

```yaml
actor_rollout_ref:
  model:
    override_config:
      attn_implementation: ./triton_attention_sink_hf
```

### Option 2: Upload to HuggingFace Hub

```bash
# 1. Upload this directory to HuggingFace
huggingface-cli upload your-username/triton-flash-attn-sink ./triton_attention_sink_hf

# 2. Then use in config:
```

```yaml
actor_rollout_ref:
  model:
    override_config:
      attn_implementation: your-username/triton-flash-attn-sink
```

### Option 3: Direct Python Usage

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    attn_implementation="/path/to/triton_attention_sink_hf",
    torch_dtype=torch.bfloat16,
)
```

## 📋 Prerequisites

Install the triton kernel first:

```bash
cd /tmp
git clone https://huggingface.co/medmekk/triton-flash-attn-sink-clone
cd triton-flash-attn-sink-clone
pip install -e build/torch-universal
```

## ✅ Verify Installation

```python
python -c "from triton_flash_attn_sink import attention; print('✓ Kernel installed')"
python -c "from triton_attention_sink_hf import TritonFlashAttentionSink; print('✓ HF wrapper ready')"
```

## 🎯 Complete VERL Example

```yaml
# config/ppo_with_attention_sink.yaml
defaults:
  - ppo_trainer
  - _self_

actor_rollout_ref:
  model:
    path: meta-llama/Llama-2-7b-hf
    override_config:
      # Use absolute or relative path
      attn_implementation: /data2/Users/aghyad/verl_copy/verl_with_logging/triton_attention_sink_hf
      # Or if uploaded to HF Hub:
      # attn_implementation: your-username/triton-flash-attn-sink

trainer:
  experiment_name: ppo_with_triton_sink
  total_epochs: 10
```

Run:
```bash
python3 -m verl.trainer.main_ppo --config-name ppo_with_attention_sink
```

## 🔧 Configuration Options

Currently supported via environment variables or config:

- `bandwidth`: Local attention window (0 = full attention)
- `enable_learned_sinks`: Use learned attention sinks (experimental)

## 📊 Performance

- ✅ Triton-optimized kernel
- ✅ Supports GQA/MQA
- ✅ Memory efficient
- ✅ Compatible with FSDP

## 🐛 Troubleshooting

**Error: "triton_flash_attn_sink is not installed"**
```bash
git clone https://huggingface.co/medmekk/triton-flash-attn-sink-clone
cd triton-flash-attn-sink-clone
pip install -e build/torch-universal
```

**Error: "Cannot find attention implementation"**
- Make sure you use absolute path or correct relative path
- Or upload to HuggingFace Hub and use `username/repo`

**Want to verify it's working?**
Check logs for:
```
[TritonFlashAttentionSink] Initialized with learned_sinks=False, bandwidth=0
```

## 📚 Technical Details

This implementation:
- Follows HF's attention implementation interface
- Wraps the triton-flash-attn-sink kernel
- Handles tensor shape conversions
- Supports GQA/MQA automatically
- Compatible with VERL's existing infrastructure

No runtime patching or code modifications needed!


