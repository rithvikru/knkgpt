#!/usr/bin/env python3
"""
Verify that the environment is set up correctly for training.
"""
import sys
import os

def check_import(module_name, package_name=None):
    """Check if a module can be imported."""
    if package_name is None:
        package_name = module_name
    try:
        __import__(module_name)
        print(f"✅ {package_name} is installed")
        return True
    except ImportError:
        print(f"❌ {package_name} is NOT installed")
        return False

def check_cuda():
    """Check CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"✅ CUDA is available with {gpu_count} GPU(s)")
            for i in range(gpu_count):
                print(f"   - GPU {i}: {torch.cuda.get_device_name(i)}")
            return True
        else:
            print("⚠️  CUDA is not available, will use CPU")
            return True
    except:
        return False

def check_data():
    """Check if data files exist."""
    data_file = "./data/n_2.jsonl"
    if os.path.exists(data_file):
        print(f"✅ Data file found: {data_file}")
        return True
    else:
        print(f"❌ Data file NOT found: {data_file}")
        print("   Please download the dataset or create it")
        return False

def check_tokenizer():
    """Check if custom tokenizer works."""
    try:
        from data.knights_knaves.tokenizer import KnightsKnavesTokenizer
        tokenizer = KnightsKnavesTokenizer()
        test_text = "says 0 (isKnight 0)"
        tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(tokens)
        print(f"✅ Custom tokenizer works")
        print(f"   Test: '{test_text}' -> {tokens} -> '{decoded}'")
        return True
    except Exception as e:
        print(f"❌ Custom tokenizer error: {e}")
        return False

def check_distributed():
    """Check if distributed training environment variables are set."""
    if 'RANK' in os.environ:
        print(f"✅ Distributed training detected:")
        print(f"   - RANK: {os.environ.get('RANK')}")
        print(f"   - WORLD_SIZE: {os.environ.get('WORLD_SIZE')}")
        print(f"   - LOCAL_RANK: {os.environ.get('LOCAL_RANK', 'Not set')}")
    else:
        print("ℹ️  Single GPU/CPU training mode")
    return True

def main():
    print("🔍 Verifying KnightKnaves GPT setup...\n")
    
    all_good = True
    
    # Check Python version
    python_version = sys.version_info
    if python_version >= (3, 9):
        print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        print(f"❌ Python {python_version.major}.{python_version.minor} (need >= 3.9)")
        all_good = False
    
    # Check required packages
    print("\n📦 Checking dependencies:")
    all_good &= check_import("torch")
    all_good &= check_import("numpy")
    all_good &= check_import("tqdm")
    all_good &= check_import("wandb")
    all_good &= check_import("jsonlines")
    
    # Check CUDA
    print("\n🖥️  Checking compute:")
    all_good &= check_cuda()
    
    # Check distributed
    print("\n🌐 Checking distributed setup:")
    check_distributed()
    
    # Check data
    print("\n📊 Checking data:")
    all_good &= check_data()
    
    # Check custom code
    print("\n🧩 Checking custom modules:")
    all_good &= check_tokenizer()
    
    # Summary
    print("\n" + "="*50)
    if all_good:
        print("✅ All checks passed! You're ready to train.")
        print("\nNext steps:")
        print("  1. Run training: python run_training.py")
        print("  2. Or debug mode: python run_training.py --config debug")
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
