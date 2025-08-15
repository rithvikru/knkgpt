# Fix for uv Build Error

The error you encountered is due to the pyproject.toml configuration expecting a specific package structure. Here's how to fix it:

## Quick Fix (Recommended)

Use the simple setup that bypasses pyproject.toml:

```bash
# Remove existing virtual environment
rm -rf .venv

# Run simple setup
make setup-simple

# Or manually:
uv venv --python 3.12
uv pip install -r requirements.txt
```

## Understanding the Error

The error occurred because:
1. The pyproject.toml defined the package name as "knkgpt"
2. Hatchling (the build backend) expected a directory named "knkgpt" to exist
3. Our project structure has "data" and "mingpt" directories instead

## Solutions

### Solution 1: Use requirements.txt mode (Simplest)
This is what the updated launch_training.sh now does:
```bash
uv venv
uv pip install -r requirements.txt
```

### Solution 2: Fix pyproject.toml
Already implemented - the pyproject.toml now correctly specifies:
```toml
[tool.hatch.build.targets.wheel]
packages = ["data", "mingpt"]
```

## Running Training After Fix

Once dependencies are installed:

```bash
# For distributed training (8 GPUs)
./launch_training.sh

# For single GPU testing
uv run python train_gpt_knights_knaves.py --max-games 100000
```

## Benefits of Simple Mode

- No build system complexity
- Faster installation
- Same functionality
- Compatible with all uv features

The training script will now work correctly!