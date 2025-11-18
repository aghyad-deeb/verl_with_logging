# Transformers Library: Attention Sink Implementation Investigation

## Executive Summary

The transformers library implements **Attention Sinks** as a technique to maintain a fixed KV cache size in transformer models by using auxiliary learnable parameters that absorb attention divergence from long-range context. This investigation reveals the complete implementation of attention sinks across multiple attention backends in GPT-OSS and related models.

---

## 1. Core Attention Sink Concept

### Definition
Attention sinks are learnable parameters that represent auxiliary "sink" tokens added to the attention computation. They help prevent context-independent attention patterns that can cause instability in long-context sequences while maintaining bounded cache sizes.

### Key Characteristics
- Shape: `[num_attention_heads]` - one sink parameter per attention head
- Type: `nn.Parameter` with learnable weights
- Initialization: Normal distribution with model's initializer_range
- Purpose: Added as auxiliary attention scores to stabilize attention distributions

---

## 2. GPT-OSS Model Implementation

### 2.1 Attention Module with Sinks

**File**: `/data2/Users/aghyad/verl_copy/verl_with_logging/venv/lib/python3.10/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py`

#### Sink Parameter Definition (Lines 273-299)

```python
class GptOssAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: GptOssConfig, layer_idx: int):
        super().__init__()
        # ... other initialization ...
        self.sinks = nn.Parameter(torch.empty(config.num_attention_heads))
```

**Key Features**:
- Line 298: `self.sinks = nn.Parameter(torch.empty(config.num_attention_heads))`
- One sink parameter per attention head
- Initialized with model's initializer (lines 432-433 in `_init_weights`)

#### Forward Pass (Lines 300-343)

```python
@deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
def forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[TransformersKwargs],
) -> tuple[torch.Tensor, torch.Tensor]:
    # ... projection code ...
    
    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        s_aux=self.sinks,  # Pass sinks as s_aux parameter
        **kwargs,
    )
```

**Key Points** (Line 337):
- Sinks passed as `s_aux` parameter to attention interface
- Compatible with all attention implementations (eager, flash, flex)

### 2.2 Weight Initialization

**File**: `/data2/Users/aghyad/verl_copy/verl_with_logging/venv/lib/python3.10/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py` (Lines 413-436)

```python
def _init_weights(self, module):
    std = self.config.initializer_range
    # ... other modules ...
    elif isinstance(module, GptOssAttention):
        module.sinks.data.normal_(mean=0.0, std=std)
```

**Initialization**:
- Normal distribution initialization with model's `initializer_range`
- Applied during model weight initialization

---

## 3. Eager Attention Implementation with Sinks

### 3.1 Eager Forward Function

**File**: `/data2/Users/aghyad/verl_copy/verl_with_logging/venv/lib/python3.10/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py` (Lines 241-270)

```python
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # ATTENTION SINK IMPLEMENTATION (Lines 258-267)
    sinks = module.sinks.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
    combined_logits = torch.cat([attn_weights, sinks], dim=-1)

    # Numerical stability: prevent overflow in BF16/FP16
    combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
    probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
    scores = probs[..., :-1]  # Drop the sink after softmax
    attn_weights = nn.functional.dropout(scores, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights
```

**Algorithm Breakdown**:

1. **Reshape Sinks** (Line 258):
   - Input: `[num_heads]`
   - Reshape: `[1, num_heads, 1, 1]`
   - Expand: `[batch, num_heads, seq_len_q, 1]`

2. **Concatenate** (Line 259):
   - `attn_weights`: `[batch, num_heads, seq_len_q, seq_len_k]`
   - `sinks`: `[batch, num_heads, seq_len_q, 1]`
   - Result: `[batch, num_heads, seq_len_q, seq_len_k + 1]`

3. **Numerical Stability** (Line 264):
   - Subtract max for numerical stability before softmax
   - Important for BF16/FP16 training

4. **Softmax** (Line 265):
   - Applied to all positions including the sink

5. **Drop Sink** (Line 266):
   - Remove last dimension after softmax: `probs[..., :-1]`
   - Sink was used only for normalization

---

## 4. Flash Attention Support

### 4.1 Flash Attention Parameters

**File**: `/data2/Users/aghyad/verl_copy/verl_with_logging/venv/lib/python3.10/site-packages/transformers/modeling_flash_attention_utils.py` (Lines 448-526)

```python
def _process_flash_attention_kwargs(
    query_length: int,
    key_length: int,
    is_causal: bool,
    dropout: float = 0.0,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
    use_top_left_mask: bool = False,
    softcap: Optional[float] = None,
    deterministic: Optional[bool] = None,
    s_aux: Optional[torch.Tensor] = None,
    supports_mapping: Optional[dict[str, bool]] = None,
    **kwargs,
):
    """Process kwargs for flash attention call.
    
    Args:
        s_aux (`torch.Tensor`, *optional*):
            Attention sink auxiliary that adds a `bias` to the attention 
            calculation via an additional head.
    """
    flash_kwargs = {
        "causal": is_causal and not (use_top_left_mask and query_length == 1),
        "softmax_scale": softmax_scale,
    }

    # ... other kwargs processing ...

    # Only within kernel implementation atm
    if supports_mapping["s_aux"] and s_aux is not None:
        flash_kwargs["s_aux"] = s_aux

    return flash_kwargs
```

**Key Points**:
- Line 463: `s_aux` parameter definition
- Line 493: Documentation explaining s_aux as auxiliary head for bias
- Lines 523-524: Conditional inclusion if supported by flash attention version

---

## 5. Flex Attention Implementation

### 5.1 Flex Attention Forward with Sinks

**File**: `/data2/Users/aghyad/verl_copy/verl_with_logging/venv/lib/python3.10/site-packages/transformers/integrations/flex_attention.py` (Lines 235-339)

```python
def flex_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Union[torch.Tensor, "BlockMask"],
    scaling: Optional[float] = None,
    softcap: Optional[float] = None,
    head_mask: Optional[torch.Tensor] = None,
    s_aux: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    # ... mask and score_mod setup ...

    def score_mod(score, batch_idx, head_idx, q_idx, kv_idx):
        if softcap is not None:
            score = softcap * torch.tanh(score / softcap)
        if score_mask is not None:
            score = score + score_mask[batch_idx][0][q_idx][kv_idx]
        if head_mask is not None:
            score = score + head_mask[batch_idx][head_idx][0][0]
        # Note: attention sinks cannot be correctly implemented in score_mod
        # because it requires operating on the full attention matrix before softmax.
        return score

    # ... flex_attention call ...

    if return_lse:
        attention_output, lse = flex_attention_output
        lse = lse.to(value.dtype)

        if s_aux is not None:
            # Apply attention sinks by renormalizing using LSE
            batch_size, num_heads, seq_len_q, _ = attention_output.shape
            sinks = s_aux.view(1, -1, 1, 1).expand(batch_size, num_heads, seq_len_q, 1)

            # We need to compute the normalization that includes the sinks
            # log(sum(exp(scores)) + exp(sink)) = log(exp(lse) + exp(sink))
            lse_expanded = lse.unsqueeze(-1)  # [batch, num_heads, seq_len, 1]
            combined_lse = torch.logsumexp(torch.cat([lse_expanded, sinks], dim=-1), dim=-1, keepdim=True)

            # Use new_norm / old_norm = exp(combined_lse - lse) to compute renorm
            renorm_factor = torch.exp(lse_expanded - combined_lse)
            attention_output = attention_output * renorm_factor
    
    attention_output = attention_output.transpose(1, 2).contiguous()
    return attention_output, lse
```

**Sink Implementation Strategy for Flex Attention** (Lines 319-333):

1. **Cannot use score_mod** (Lines 275-276):
   - Sinks need full matrix before softmax
   - score_mod operates on individual scores

2. **Post-computation Renormalization**:
   - Use Log-Sum-Exp (LSE) trick for stability
   - Compute: `combined_lse = log(exp(lse) + exp(sink))`
   - Renormalization factor: `exp(lse_expanded - combined_lse)`
   - Apply to attention output: `output * renorm_factor`

3. **Why LSE Trick?**:
   - Maintains numerical stability
   - Avoids explicit exp/softmax recomputation
   - Mathematically equivalent to including sink in softmax

---

## 6. Eager Paged Attention Integration

### 6.1 Eager Paged Attention Forward

**File**: `/data2/Users/aghyad/verl_copy/verl_with_logging/venv/lib/python3.10/site-packages/transformers/integrations/eager_paged.py` (Lines 19-70)

```python
def eager_paged_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    **kwargs,
):
    # ... cache and mask setup ...

    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if causal_mask is not None:
        attn_weights = attn_weights + causal_mask

    # Handle attention sinks if the model has them
    if hasattr(module, "sinks"):
        # Retrieve the sink and add it to the attention weights
        sinks = module.sinks.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
        attn_weights = torch.cat([attn_weights, sinks], dim=-1)
        # Normalize the attention weights for better numerical stability
        attn_weights = attn_weights - attn_weights.max(dim=-1, keepdim=True).values
        # Apply softmax and drop the sink
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = attn_weights[..., :-1]
    else:
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights
```

**Implementation**:
- Lines 54-62: Conditional sink handling
- Same pattern as eager attention: reshape, concat, softmax, drop sink
- Uses `hasattr` to check for sink support
- Better numerical stability (explicit float32 softmax)

---

## 7. Flash Paged Attention

### 7.1 Flash Paged Forward

**File**: `/data2/Users/aghyad/verl_copy/verl_with_logging/venv/lib/python3.10/site-packages/transformers/integrations/flash_paged.py` (Lines 26-96)

```python
def paged_attention_forward(
    module: torch.nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    cache: PagedAttentionCache = None,
    cu_seq_lens_q=None,
    cu_seq_lens_k=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    implementation=None,
    **kwargs,
) -> torch.Tensor:
    # ... setup code ...

    custom_kwargs = {"s_aux": kwargs.get("s_aux")} if "s_aux" in kwargs else {}

    attn_output = flash_attn_varlen_func(
        q.transpose(1, 2).squeeze(0).contiguous(),
        k.contiguous(),
        v.contiguous(),
        cu_seq_lens_q.to(torch.int32),
        cu_seq_lens_k.to(torch.int32).clone(),
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale=module.scaling,
        causal=True,
        window_size=sliding_window,
        **custom_kwargs,
    )
```

**Key Feature** (Line 79):
- Extracts `s_aux` from kwargs if present
- Passes directly to flash_attn_varlen_func
- Allows version-dependent support

---

## 8. Configuration and Layer Type Management

### 8.1 GPT-OSS Configuration

**File**: `/data2/Users/aghyad/verl_copy/verl_with_logging/venv/lib/python3.10/site-packages/transformers/models/gpt_oss/configuration_gpt_oss.py`

```python
class GptOssConfig(PretrainedConfig):
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.self_attn.sinks": "local_rowwise",  # Line 39
        "layers.*.mlp.experts": "gather",
        # ... other params ...
    }

    def __init__(
        self,
        num_hidden_layers: int = 36,
        # ... config params ...
        layer_types=None,
        **kwargs,
    ):
        # ... initialization ...
        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention" if bool((i + 1) % 2) else "full_attention" 
                for i in range(self.num_hidden_layers)
            ]
```

**Key Features**:
- Line 39: Sinks use `"local_rowwise"` tensor parallelism strategy
- Supports mixed layer types (sliding and full attention)

---

## 9. Sink Parameter Characteristics

### 9.1 Tensor Parallelism Strategy

**File**: `/data2/Users/aghyad/verl_copy/verl_with_logging/venv/lib/python3.10/site-packages/transformers/models/gpt_oss/configuration_gpt_oss.py` (Line 39)

```python
"layers.*.self_attn.sinks": "local_rowwise",
```

This means:
- **local_rowwise**: Each TP rank keeps the full sink parameters locally
- Not split across TP ranks like other attention parameters
- Each rank uses all sink values in its attention computation

---

## 10. Cache Management with Attention Sinks

### 10.1 Cache Layer Mixin

**File**: `/data2/Users/aghyad/verl_copy/verl_with_logging/venv/lib/python3.10/site-packages/transformers/cache_utils.py`

The cache system works independently of attention sinks:
- Sinks are added at attention computation time
- Cache stores original KV states
- Sinks don't affect cache shape or structure
- Attention sinks + sliding window cache = bounded memory

**Example Usage**:
```python
past_key_values = DynamicCache(config=model.config)
outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)
```

---

## 11. Attention Sink Flow Diagram

```
Input Tokens
    |
    v
Query, Key, Value Projections
    |
    v
Query @ Key.T * scaling  -> [batch, heads, seq_q, seq_k]
    |
    v
Add Causal Mask (if applicable)
    |
    +---> Concat with Sinks -> [batch, heads, seq_q, seq_k + 1]
    |                              (add auxiliary column)
    |
    v
Subtract Max (stability)
    |
    v
Softmax over [seq_k + 1]
    |
    v
Drop Last Position (sink only for normalization)
    |                -> [batch, heads, seq_q, seq_k]
    v
Attention Weights @ Value
    |
    v
Output
```

---

## 12. Numerical Stability Considerations

### 12.1 Sink Value Considerations

From the code (eager_attention_forward):

```python
# Prevent overflow in BF16/FP16 when training with bsz>1
combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
```

**Why Important**:
1. Sinks can absorb very different attention scales
2. Adding them to logits can cause numerical issues
3. Max subtraction prevents overflow/underflow
4. Maintains numerical stability in low-precision training

### 12.2 LSE-based Renormalization (Flex Attention)

```python
combined_lse = torch.logsumexp(torch.cat([lse_expanded, sinks], dim=-1), dim=-1, keepdim=True)
renorm_factor = torch.exp(lse_expanded - combined_lse)
attention_output = attention_output * renorm_factor
```

**Advantages**:
- Log-sum-exp is numerically stable
- Avoids explicit recomputation of softmax
- Maintains precision in final outputs

---

## 13. Key Design Patterns

### 13.1 Modularity

1. **Attention Backend Agnostic**: Same sink values work with eager, flash, flex
2. **Passed as `s_aux` Parameter**: Standard name across all implementations
3. **Conditionally Applied**: `hasattr(module, "sinks")` check
4. **Backend-Specific Logic**: Different computation strategies per backend

### 13.2 Initialization and Optimization

1. **Learnable Parameters**: `nn.Parameter` allows gradient updates
2. **Normal Distribution**: Initialized with model's `initializer_range`
3. **Per-Head Granularity**: One sink per attention head
4. **TP-Aware**: Uses `local_rowwise` for tensor parallelism

---

## 14. Summary Table

| Aspect | Value | File | Line |
|--------|-------|------|------|
| Parameter Shape | `[num_attention_heads]` | modeling_gpt_oss.py | 298 |
| Type | `nn.Parameter` | modeling_gpt_oss.py | 298 |
| Eager Implementation | Concat + Softmax | modeling_gpt_oss.py | 258-267 |
| Flash Support | Via s_aux parameter | modeling_flash_attention_utils.py | 523-524 |
| Flex Implementation | LSE renormalization | flex_attention.py | 319-333 |
| Paged Eager | hasattr check | eager_paged.py | 54-62 |
| Paged Flash | Extract s_aux | flash_paged.py | 79 |
| TP Strategy | local_rowwise | configuration_gpt_oss.py | 39 |
| Initialization | Normal distribution | modeling_gpt_oss.py | 433 |

---

## 15. Research Context

Attention sinks implement the concept from **"Attention Sink: A Simple One-Token Fix for Infinite Attention in Transformers"** paper:
- Designed for inference with long sequences
- Maintains bounded KV cache
- Learns to "absorb" context-independent attention patterns
- Compatible with sliding window and other cache optimization techniques

---

## Conclusion

The transformers library implements attention sinks as a sophisticated yet elegant solution to handle long-context sequences:

1. **Simple Parameter**: Just `[num_heads]` learnable values
2. **Backend Agnostic**: Works with eager, flash, and flex attention
3. **Numerically Stable**: Uses careful max-subtraction and LSE tricks
4. **Fully Integrated**: Supports mixed layer types, TP, and all cache strategies
5. **Production Ready**: Already used in GPT-OSS model

The implementation demonstrates careful engineering for both correctness and efficiency across multiple attention computation backends.

