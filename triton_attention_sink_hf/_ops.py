"""Operations interface for triton attention sink."""

def attention(*args, **kwargs):
    """Forward to the actual triton kernel."""
    from triton_flash_attn_sink import attention as triton_attention
    return triton_attention(*args, **kwargs)


