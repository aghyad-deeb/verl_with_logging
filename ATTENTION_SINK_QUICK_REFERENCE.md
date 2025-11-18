# Attention Sink Implementation - Quick Reference

## Core Files and Line Numbers

### 1. Parameter Definition
**File**: `/data2/Users/aghyad/verl_copy/verl_with_logging/venv/lib/python3.10/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py`
- **Line 298**: `self.sinks = nn.Parameter(torch.empty(config.num_attention_heads))`
- **Line 433**: Initialization: `module.sinks.data.normal_(mean=0.0, std=std)`

### 2. Eager Attention (Main Implementation)
**File**: `transformers/models/gpt_oss/modeling_gpt_oss.py`
- **Lines 241-270**: Complete `eager_attention_forward` function
- **Line 258**: Reshape: `sinks.reshape(1, -1, 1, 1).expand(...)`
- **Line 259**: Concat: `torch.cat([attn_weights, sinks], dim=-1)`
- **Line 264**: Stability: `combined_logits - combined_logits.max(...)`
- **Line 265**: Softmax: `F.softmax(combined_logits, ...)`
- **Line 266**: Drop sink: `probs[..., :-1]`

### 3. Flash Attention Support
**File**: `transformers/modeling_flash_attention_utils.py`
- **Line 463**: Parameter definition in `_process_flash_attention_kwargs`
- **Line 493**: Documentation of s_aux
- **Lines 523-524**: Conditional pass-through

### 4. Flex Attention (LSE-based)
**File**: `transformers/integrations/flex_attention.py`
- **Lines 235-339**: Complete `flex_attention_forward` function
- **Lines 275-276**: Comment about why score_mod cannot be used
- **Lines 319-333**: LSE renormalization implementation
  - Line 322: `sinks.view(1, -1, 1, 1).expand(...)`
  - Line 328: `torch.logsumexp(...)`
  - Line 331: `torch.exp(lse_expanded - combined_lse)`
  - Line 332: `attention_output * renorm_factor`

### 5. Eager Paged Attention
**File**: `transformers/integrations/eager_paged.py`
- **Lines 54-62**: Sink handling with `hasattr` check
- **Line 54**: `if hasattr(module, "sinks"):`
- **Line 56**: Reshape and expand
- **Line 57**: Concatenate with attention weights
- **Lines 59-62**: Softmax with stability and drop sink

### 6. Flash Paged Attention
**File**: `transformers/integrations/flash_paged.py`
- **Line 79**: `custom_kwargs = {"s_aux": kwargs.get("s_aux")} if "s_aux" in kwargs else {}`
- **Line 92**: `**custom_kwargs` passed to flash_attn_varlen_func

### 7. Configuration
**File**: `transformers/models/gpt_oss/configuration_gpt_oss.py`
- **Line 39**: Tensor parallelism: `"layers.*.self_attn.sinks": "local_rowwise"`
- **Lines 102-105**: Layer type configuration

---

## Algorithm Overview

### Eager Attention Algorithm
```
1. Compute attention logits: Q @ K^T * scale
2. Add causal mask
3. Reshape sinks from [H] to [1, H, 1, 1]
4. Expand to [B, H, S_q, 1]
5. Concatenate: [B, H, S_q, S_k + 1]
6. Subtract max: numerical stability
7. Softmax: F.softmax(..., dim=-1)
8. Remove last column (sink): [..., :-1] = [B, H, S_q, S_k]
9. Multiply with values
```

### Flex Attention Algorithm (LSE-based)
```
1. Compute attention with flex_attention normally
2. Get LSE (log-sum-exp) from flex_attention output
3. View sinks as [1, H, 1, 1], expand to [B, H, S_q, 1]
4. Compute combined_lse = logsumexp([lse, sinks])
5. Compute renorm_factor = exp(lse - combined_lse)
6. Multiply attention_output by renorm_factor
```

---

## Key Design Decisions

| Decision | Reasoning | Files |
|----------|-----------|-------|
| **[H] shape** | One sink per head | modeling_gpt_oss.py:298 |
| **nn.Parameter** | Make learnable | modeling_gpt_oss.py:298 |
| **Normal init** | Standard weight init | modeling_gpt_oss.py:433 |
| **s_aux name** | Standard across backends | All attention files |
| **Concat + drop** | Simple and stable | eager implementations |
| **LSE trick** | Numerically stable | flex_attention.py:328-332 |
| **local_rowwise TP** | No TP split needed | configuration_gpt_oss.py:39 |
| **hasattr check** | Backward compat | eager_paged.py:54 |

---

## Integration Points

### Where s_aux Parameter is Passed

1. **GptOssAttention.forward** (line 337 of modeling_gpt_oss.py)
   ```python
   s_aux=self.sinks
   ```

2. **ALL_ATTENTION_FUNCTIONS routing** (lines 283-327 of modeling_gpt_oss.py)
   ```python
   attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
   attn_output, attn_weights = attention_interface(..., s_aux=self.sinks, ...)
   ```

### Backend-Specific Handling

| Backend | File | Handling |
|---------|------|----------|
| eager | modeling_gpt_oss.py | Direct concat + softmax |
| flash | modeling_flash_attention_utils.py | Pass to s_aux kwarg |
| flex | integrations/flex_attention.py | LSE renormalization |
| eager_paged | integrations/eager_paged.py | hasattr + concat |
| flash_paged | integrations/flash_paged.py | Extract and pass |

---

## Common Issues and Fixes

### Issue: Sinks not initialized
**Fix**: Ensure `_init_weights` is called - check modeling_gpt_oss.py:433

### Issue: Wrong shape for concat
**Fix**: Must reshape to [1, H, 1, 1] then expand - line 258 of modeling_gpt_oss.py

### Issue: Numerical overflow
**Fix**: Subtract max before softmax - line 264 of modeling_gpt_oss.py

### Issue: Model missing sinks
**Fix**: Use hasattr check - line 54 of eager_paged.py

### Issue: Flex attention returns wrong shape
**Fix**: Sinks only applied when return_lse=True - line 293 of flex_attention.py

---

## Testing Checklist

- [ ] Sinks shape: should be [num_attention_heads]
- [ ] Forward pass: should handle sinks with eager attention
- [ ] Flash attention: check if s_aux is passed through
- [ ] Flex attention: verify LSE renormalization
- [ ] Paged attention: verify hasattr check works
- [ ] Gradient flow: ensure sinks receive gradients
- [ ] Numerical stability: check no NaN/inf in BF16
- [ ] Long sequences: verify bounded cache with sinks
- [ ] Tensor parallelism: verify local_rowwise strategy

---

## References

**Paper**: "Attention Sink: A Simple One-Token Fix for Infinite Attention in Transformers"
- Designed for long-context inference
- Maintains fixed KV cache size
- Learns to absorb context patterns

**Model**: GPT-OSS (MoE mixture of experts model)
- 36 layers with mixed full/sliding attention
- 64 attention heads
- Uses attention sinks for efficiency

---

## Absolute File Paths

```
/data2/Users/aghyad/verl_copy/verl_with_logging/venv/lib/python3.10/site-packages/transformers/
├── models/
│   └── gpt_oss/
│       ├── modeling_gpt_oss.py (main implementation)
│       ├── modular_gpt_oss.py (source for code generation)
│       └── configuration_gpt_oss.py (TP configuration)
├── modeling_flash_attention_utils.py (s_aux parameter def)
├── cache_utils.py (cache integration)
└── integrations/
    ├── flex_attention.py (LSE implementation)
    ├── eager_paged.py (paged eager)
    └── flash_paged.py (paged flash)
```

---

## Key Metrics

| Metric | Value |
|--------|-------|
| Sinks per model | num_hidden_layers * num_attention_heads |
| GPT-OSS model | 36 * 64 = 2,304 sink parameters |
| Memory overhead | Negligible (2K-4K parameters) |
| Computation overhead | 1 extra softmax dimension |
| Initialization std | model.config.initializer_range |

