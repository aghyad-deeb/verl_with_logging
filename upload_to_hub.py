"""
Upload triton_flash_attn_sink to HuggingFace Hub.
Matches the structure of kernels-community/vllm-flash-attn3.
"""

from huggingface_hub import HfApi, create_repo
import os

def upload_to_hub(repo_id: str, token: str = None):
    """
    Upload the triton_flash_attn_sink package to HuggingFace Hub.
    
    Args:
        repo_id: Repository ID in format "username/repo-name"
        token: HuggingFace token (optional if logged in via CLI)
    """
    print(f"Uploading to: {repo_id}")
    print(f"Local path: triton_flash_attn_sink/")
    
    api = HfApi(token=token)
    
    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, repo_type="model", exist_ok=True, token=token)
        print(f"✅ Repository created/verified: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"⚠️  Could not create repo: {e}")
    
    # Upload the directory
    print("\nUploading files...")
    api.upload_folder(
        folder_path="triton_flash_attn_sink",
        repo_id=repo_id,
        repo_type="model",
        token=token,
    )
    
    print(f"\n✅ Upload complete!")
    print(f"\nYou can now use it in VERL:")
    print(f"  override_config:")
    print(f"    attn_implementation: {repo_id}")
    print(f"\nView at: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("Upload Triton Flash Attention Sink to HuggingFace Hub")
    print("=" * 60)
    print()
    
    # Get repo ID from user
    if len(sys.argv) > 1:
        repo_id = sys.argv[1]
    else:
        print("Usage: python upload_to_hub.py YOUR-USERNAME/triton-flash-attn-sink")
        print()
        print("Example: python upload_to_hub.py aghyad/triton-flash-attn-sink")
        print()
        repo_id = input("Enter repository ID (username/repo-name): ").strip()
    
    if not repo_id or '/' not in repo_id:
        print("❌ Invalid repository ID. Must be in format: username/repo-name")
        sys.exit(1)
    
    print()
    print("Make sure you're logged in to HuggingFace:")
    print("  huggingface-cli login")
    print()
    
    confirm = input(f"Upload to {repo_id}? (yes/no): ").strip().lower()
    if confirm != 'yes':
        print("Cancelled.")
        sys.exit(0)
    
    try:
        upload_to_hub(repo_id)
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        print("\nMake sure you:")
        print("  1. Installed huggingface_hub: pip install huggingface_hub")
        print("  2. Logged in: huggingface-cli login")
        print("  3. Have permissions to create repos")
        sys.exit(1)
