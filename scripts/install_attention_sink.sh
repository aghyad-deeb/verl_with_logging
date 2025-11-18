#!/bin/bash
# Installation script for triton-flash-attn-sink kernel
# Usage: bash scripts/install_attention_sink.sh

set -e

echo "======================================"
echo "Installing Triton Flash Attention Sink"
echo "======================================"
echo ""

# Check if running in the correct directory
if [ ! -f "setup.py" ]; then
    echo "Error: Please run this script from the VERL root directory"
    exit 1
fi

# Check Python version
echo "Checking Python version..."
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "Python version: $python_version"

if [ "$(printf '%s\n' "3.8" "$python_version" | sort -V | head -n1)" != "3.8" ]; then
    echo "Error: Python 3.8+ is required"
    exit 1
fi

# Check for required dependencies
echo ""
echo "Checking dependencies..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')" || {
    echo "Error: PyTorch is not installed"
    exit 1
}

python3 -c "import triton; print(f'Triton version: {triton.__version__}')" || {
    echo "Warning: Triton is not installed. Installing..."
    pip install triton>=2.0.0
}

# Clone the repository
echo ""
echo "Cloning triton-flash-attn-sink repository..."
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

git clone https://huggingface.co/medmekk/triton-flash-attn-sink-clone
cd triton-flash-attn-sink-clone

# Install the package
echo ""
echo "Installing the kernel package..."
pip install -e build/torch-universal

# Verify installation
echo ""
echo "Verifying installation..."
python3 -c "from triton_flash_attn_sink import attention; print('✓ Installation successful!')" || {
    echo "✗ Installation failed"
    exit 1
}

# Cleanup
cd -
rm -rf "$TEMP_DIR"

echo ""
echo "======================================"
echo "Installation complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo "1. Add attention_sink_options to your config file:"
echo ""
echo "   actor_rollout_ref:"
echo "     model:"
echo "       attention_sink_options:"
echo "         enable: true"
echo "         enable_learned_sinks: false"
echo "         bandwidth: 0"
echo ""
echo "2. Run your training:"
echo "   python3 -m verl.trainer.main_ppo --config-name your_config"
echo ""
echo "For more information, see: docs/advance/attention_sink_integration.rst"
echo ""


