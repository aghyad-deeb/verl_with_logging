.. _attention-sink-integration:

Triton Flash Attention with Attention Sinks
============================================

Last updated: 11/10/2025.

This guide explains how to integrate and use the Triton Flash Attention kernel with attention sinks
in VERL for improved long-context generation and streaming LLM capabilities.

Overview
--------

The attention sink mechanism (from `StreamingLLM <https://arxiv.org/abs/2309.17453>`_) keeps the first 
few tokens always in the attention window, which helps maintain stable attention patterns during 
long-context generation. This implementation uses an optimized Triton kernel that combines:

- **Flash Attention v2**: Fast and memory-efficient attention computation
- **Attention Sinks**: Learned or fixed sink values that remain in attention
- **Banded Attention**: Optional local attention with configurable bandwidth

Installation
------------

Step 1: Clone the Kernel Repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

    # Clone the triton-flash-attn-sink kernel
    cd /tmp
    git clone https://huggingface.co/medmekk/triton-flash-attn-sink-clone
    cd triton-flash-attn-sink-clone

Step 2: Install the Package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

    # Install as a Python package
    pip install -e build/torch-universal

Step 3: Verify Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

    python -c "from triton_flash_attn_sink import attention; print('Successfully installed!')"

If you see "Successfully installed!", you're ready to use attention sinks in VERL!

Configuration
-------------

Basic Configuration
~~~~~~~~~~~~~~~~~~~

Attention sinks are enabled by setting ``attn_implementation`` to ``attention_sink`` in the model's 
``override_config``. This integrates cleanly with VERL's existing attention implementation system.

**YAML Configuration:**

.. code:: yaml

    actor_rollout_ref:
      model:
        path: meta-llama/Llama-2-7b-hf
        override_config:
          attn_implementation: attention_sink
          # Optional parameters:
          attention_sink_learned_sinks: false
          attention_sink_bandwidth: 0
          attention_sink_init_value: 0.0

**Command-line Override:**

.. code:: bash

    python3 -m verl.trainer.main_ppo \
        +actor_rollout_ref.model.override_config.attn_implementation=attention_sink \
        +actor_rollout_ref.model.override_config.attention_sink_bandwidth=0 \
        [other parameters...]

Configuration Options
~~~~~~~~~~~~~~~~~~~~~

``attn_implementation`` (str, default: "flash_attention_2")
    Set to ``"attention_sink"`` to enable the attention sink kernel.

``attention_sink_learned_sinks`` (bool, default: false)
    Whether to use learned attention sink values (per-head trainable parameters).
    
    - ``true``: Each attention head has a learnable sink value that is optimized during training
    - ``false``: Uses fixed sink initialization value

``attention_sink_bandwidth`` (int, default: 0)
    Local attention bandwidth. Controls the attention window size.
    
    - ``0``: Full causal attention (standard behavior)
    - ``> 0``: Local attention with specified bandwidth (e.g., 2048, 4096)
    
    For long-context streaming: use values like 2048-4096
    For standard training: keep at 0

``attention_sink_init_value`` (float, default: 0.0)
    Initial value for attention sinks in log-space. Only used when ``attention_sink_learned_sinks=true``.

Use Cases
---------

Standard PPO Training with Attention Sinks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For normal PPO training with attention sink optimization:

.. code:: yaml

    actor_rollout_ref:
      model:
        override_config:
          attn_implementation: attention_sink
          attention_sink_learned_sinks: false  # Fixed sinks
          attention_sink_bandwidth: 0          # Full attention

Long-Context Streaming
~~~~~~~~~~~~~~~~~~~~~~

For long-context generation with local attention:

.. code:: yaml

    actor_rollout_ref:
      model:
        override_config:
          attn_implementation: attention_sink
          attention_sink_learned_sinks: false
          attention_sink_bandwidth: 4096  # Local attention window

Learned Attention Sinks
~~~~~~~~~~~~~~~~~~~~~~~~

To train models with learned attention sink parameters:

.. code:: yaml

    actor_rollout_ref:
      model:
        override_config:
          attn_implementation: attention_sink
          attention_sink_learned_sinks: true  # Trainable per-head sinks
          attention_sink_bandwidth: 0
          attention_sink_init_value: 0.0      # Initial sink value

Applying to Different Components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can configure attention sinks separately for actor, rollout, reference, and critic models:

.. code:: yaml

    actor_rollout_ref:
      model:
        override_config:
          attn_implementation: attention_sink
          attention_sink_bandwidth: 0

    critic:
      model:
        override_config:
          attn_implementation: attention_sink
          attention_sink_bandwidth: 2048  # Different bandwidth for critic

Complete Example
----------------

Here's a complete example configuration file:

.. code:: yaml

    defaults:
      - ppo_trainer
      - _self_

    actor_rollout_ref:
      model:
        path: meta-llama/Llama-2-7b-hf
        override_config:
          attn_implementation: attention_sink
          attention_sink_learned_sinks: false
          attention_sink_bandwidth: 0
          attention_sink_init_value: 0.0

    trainer:
      experiment_name: ppo_with_attention_sinks
      total_epochs: 20
      project_name: my_project

    data:
      train_files: data/train.parquet
      val_files: data/val.parquet

Running the Training:

.. code:: bash

    python3 -m verl.trainer.main_ppo --config-name my_attention_sink_config

Performance Considerations
--------------------------

Memory Usage
~~~~~~~~~~~~

The attention sink kernel has similar memory characteristics to standard Flash Attention 2:

- Full attention (bandwidth=0): Same as flash_attention_2
- Local attention (bandwidth>0): Reduced memory for very long sequences

Throughput
~~~~~~~~~~

- Comparable performance to flash_attention_2 for standard sequence lengths
- Improved throughput for long-context generation with local attention

When to Use
~~~~~~~~~~~

**Use attention sinks when:**

- Training models for long-context applications
- Implementing streaming LLM capabilities
- Need stable attention patterns across long sequences

**Use standard flash attention when:**

- Standard sequence lengths (<4096 tokens)
- Maximum throughput is critical
- Don't need streaming capabilities

Troubleshooting
---------------

ImportError: triton_flash_attn_sink not found
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Solution:** Install the kernel following the installation steps above.

.. code:: bash

    cd /tmp
    git clone https://huggingface.co/medmekk/triton-flash-attn-sink-clone
    cd triton-flash-attn-sink-clone
    pip install -e build/torch-universal

Kernel compilation errors
~~~~~~~~~~~~~~~~~~~~~~~~~

**Solution:** Ensure you have compatible versions:

- CUDA 11.8+ or 12.0+
- PyTorch 2.0+
- Triton 2.0+

.. code:: bash

    pip install triton>=2.0.0
    pip install torch>=2.0.0

Model fails to load
~~~~~~~~~~~~~~~~~~~

**Solution:** Make sure your model supports flash attention. Check that:

1. Model has ``attn_implementation`` parameter support
2. Model architecture is compatible (most decoder-only models work)

Performance degradation
~~~~~~~~~~~~~~~~~~~~~~~

**Solution:** Try adjusting configuration:

- Set ``bandwidth=0`` for full attention if you don't need local attention
- Disable learned sinks if you don't need trainable parameters
- Verify your hardware supports the kernel efficiently

References
----------

- `StreamingLLM Paper <https://arxiv.org/abs/2309.17453>`_
- `Flash Attention v2 Paper <https://tridao.me/publications/flash2/flash2.pdf>`_
- `Triton Language Documentation <https://triton-lang.org/>`_
- `Kernel Repository <https://huggingface.co/medmekk/triton-flash-attn-sink-clone>`_

Advanced Topics
---------------

Custom Sink Values
~~~~~~~~~~~~~~~~~~

If you want to experiment with different sink initialization strategies:

.. code:: python

    # In your custom training script
    from verl.models.transformers.attention_sink import create_attention_sink_forward
    
    attention_fn = create_attention_sink_forward(
        num_attention_heads=32,
        num_key_value_heads=32,
        enable_learned_sinks=True,
        bandwidth=2048,
        sink_init_value=-2.0,  # Custom init value
    )

Combining with Other Optimizations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Attention sinks can be combined with:

- Gradient checkpointing
- LoRA/QLoRA
- Sequence parallelism (Ulysses)
- FSDP

Example:

.. code:: yaml

    actor_rollout_ref:
      model:
        override_config:
          attn_implementation: attention_sink
        enable_gradient_checkpointing: true
        lora_options:
          enable: true
          r: 16

Monitoring and Debugging
~~~~~~~~~~~~~~~~~~~~~~~~~

To verify attention sinks are being used, check the training logs:

.. code:: text

    Initialized AttentionSinkWrapper with 32 heads, learned_sinks=False, bandwidth=0
    Monkey patch _flash_attention_forward with AttentionSink in transformers.models.llama.modeling_llama

This confirms the kernel is active.

