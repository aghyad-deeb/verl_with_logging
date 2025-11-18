# 🚀 Attention Sink Quick Start (No VERL Code Changes!)

Use Triton Flash Attention with Attention Sinks in VERL **without modifying any VERL code**.

## 📦 Installation (3 steps)

### 1. Install the Triton Kernel

```bash
cd /tmp
git clone https://huggingface.co/medmekk/triton-flash-attn-sink-clone
cd triton-flash-attn-sink-clone
pip install -e build/torch-universal
```

### 2. Install the Plugin

```bash
cd verl_attention_sink_plugin
pip install -e .
```

### 3. Verify

```bash
python -c "from triton_flash_attn_sink import attention; import verl_attention_sink; print('✅ Ready!')"
```

## 🎯 Usage (Just Change Config!)

### Option A: YAML Config

```yaml
actor_rollout_ref:
  model:
    override_config:
      attn_implementation: attention_sink
```

### Option B: Command Line

```bash
python3 -m verl.trainer.main_ppo \
    +actor_rollout_ref.model.override_config.attn_implementation=attention_sink
```

That's it! **Zero code changes to VERL!**

## 📝 Example Configs

### Standard Training
```yaml
actor_rollout_ref:
  model:
    path: meta-llama/Llama-2-7b-hf
    override_config:
      attn_implementation: attention_sink
```

### Long-Context (Local Attention)
```yaml
actor_rollout_ref:
  model:
    override_config:
      attn_implementation: attention_sink
      attention_sink_bandwidth: 4096  # 4k token window
```

### Learned Sinks
```yaml
actor_rollout_ref:
  model:
    override_config:
      attn_implementation: attention_sink
      attention_sink_learned_sinks: true  # Trainable
```

## ✅ Verify It's Working

Look for these in your training logs:

```
[AttentionSink Plugin] Successfully patched VERL for attention_sink support
[AttentionSink Plugin] Initialized with learned_sinks=False, bandwidth=0
[AttentionSink Plugin] Patched _flash_attention_forward in ...
```

## 🔧 Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `attn_implementation` | `flash_attention_2` | Set to `attention_sink` |
| `attention_sink_learned_sinks` | `false` | Use trainable sinks |
| `attention_sink_bandwidth` | `0` | Local window (0=full) |
| `attention_sink_init_value` | `0.0` | Initial sink value |

## 📚 Full Documentation

See `verl_attention_sink_plugin/README.md` for:
- Advanced configuration
- Use cases
- Troubleshooting
- Technical details

## 🐛 Troubleshooting

**Plugin not loading?**
```python
# Add to your training script
import verl_attention_sink  # Force load before running
```

**Disable the plugin?**
```bash
export VERL_DISABLE_ATTENTION_SINK_PATCH=1
```

## 🎓 References

- [StreamingLLM Paper](https://arxiv.org/abs/2309.17453)
- [Kernel Repo](https://huggingface.co/medmekk/triton-flash-attn-sink-clone)
- Plugin: `verl_attention_sink_plugin/`


