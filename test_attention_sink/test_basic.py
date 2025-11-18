"""
Basic test to verify triton attention sink HF-compatible wrapper works.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_import():
    """Test 1: Can we import the module?"""
    print("=" * 60)
    print("TEST 1: Import Test")
    print("=" * 60)
    
    try:
        from triton_attention_sink_hf import TritonFlashAttentionSink
        print("✅ Successfully imported TritonFlashAttentionSink")
        return True
    except Exception as e:
        print(f"❌ Failed to import: {e}")
        return False


def test_triton_kernel():
    """Test 2: Is the triton kernel installed?"""
    print("\n" + "=" * 60)
    print("TEST 2: Triton Kernel Installation")
    print("=" * 60)
    
    try:
        from triton_flash_attn_sink import attention
        print("✅ Triton kernel is installed")
        return True
    except ImportError as e:
        print(f"❌ Triton kernel not installed: {e}")
        print("\nInstall with:")
        print("  cd /tmp")
        print("  git clone https://huggingface.co/medmekk/triton-flash-attn-sink-clone")
        print("  cd triton-flash-attn-sink-clone")
        print("  pip install -e build/torch-universal")
        return False


def test_basic_forward():
    """Test 3: Can we run a basic forward pass?"""
    print("\n" + "=" * 60)
    print("TEST 3: Basic Forward Pass")
    print("=" * 60)
    
    try:
        import torch
        from triton_attention_sink_hf import TritonFlashAttentionSink
        
        # Create test inputs
        batch_size = 2
        seq_len = 128
        num_heads = 8
        head_dim = 64
        
        print(f"Creating test tensors:")
        print(f"  Batch: {batch_size}, SeqLen: {seq_len}, Heads: {num_heads}, HeadDim: {head_dim}")
        
        query = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.float16)
        key = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.float16)
        value = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.float16)
        
        # Create attention instance
        attn = TritonFlashAttentionSink()
        
        # Forward pass
        print("\nRunning forward pass...")
        output = attn(query, key, value)
        
        # Check output
        assert output.shape == query.shape, f"Shape mismatch: {output.shape} != {query.shape}"
        assert output.dtype == query.dtype, f"Dtype mismatch: {output.dtype} != {query.dtype}"
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"
        
        print(f"✅ Forward pass successful!")
        print(f"  Output shape: {output.shape}")
        print(f"  Output dtype: {output.dtype}")
        print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gqa():
    """Test 4: Does GQA (Grouped Query Attention) work?"""
    print("\n" + "=" * 60)
    print("TEST 4: Grouped Query Attention (GQA)")
    print("=" * 60)
    
    try:
        import torch
        from triton_attention_sink_hf import TritonFlashAttentionSink
        
        batch_size = 2
        seq_len = 128
        num_q_heads = 32
        num_kv_heads = 8  # GQA: fewer KV heads
        head_dim = 64
        
        print(f"Testing GQA:")
        print(f"  Query heads: {num_q_heads}, KV heads: {num_kv_heads}")
        
        query = torch.randn(batch_size, seq_len, num_q_heads, head_dim, device='cuda', dtype=torch.float16)
        key = torch.randn(batch_size, seq_len, num_kv_heads, head_dim, device='cuda', dtype=torch.float16)
        value = torch.randn(batch_size, seq_len, num_kv_heads, head_dim, device='cuda', dtype=torch.float16)
        
        attn = TritonFlashAttentionSink()
        output = attn(query, key, value)
        
        assert output.shape == query.shape, f"Shape mismatch in GQA"
        
        print(f"✅ GQA test successful!")
        print(f"  Output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ GQA test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_path_format():
    """Test 5: Test the path format that will be used in config"""
    print("\n" + "=" * 60)
    print("TEST 5: Path Format Test")
    print("=" * 60)
    
    # Get the absolute path
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file))
    attn_path = os.path.join(project_root, "triton_attention_sink_hf")
    
    print(f"Testing path that will be used in config:")
    print(f"  Path: {attn_path}")
    
    # Check if path exists
    if os.path.exists(attn_path):
        print(f"✅ Path exists")
        
        # Check if __init__.py exists
        init_file = os.path.join(attn_path, "__init__.py")
        if os.path.exists(init_file):
            print(f"✅ __init__.py found")
        else:
            print(f"❌ __init__.py not found")
            return False
        
        # Check if main module exists
        main_file = os.path.join(attn_path, "flash_attention_triton_sink.py")
        if os.path.exists(main_file):
            print(f"✅ flash_attention_triton_sink.py found")
        else:
            print(f"❌ flash_attention_triton_sink.py not found")
            return False
        
        print(f"\nYou can use this in your VERL config:")
        print(f"  override_config:")
        print(f"    attn_implementation: {attn_path}")
        
        return True
    else:
        print(f"❌ Path does not exist: {attn_path}")
        return False


def run_all_tests():
    """Run all tests and report results"""
    print("\n" + "🧪" * 30)
    print("TRITON ATTENTION SINK - TEST SUITE")
    print("🧪" * 30 + "\n")
    
    tests = [
        ("Import Test", test_import),
        ("Triton Kernel Check", test_triton_kernel),
        ("Basic Forward Pass", test_basic_forward),
        ("GQA Support", test_gqa),
        ("Path Format", test_path_format),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Ready to use in VERL!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix before using.")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)


