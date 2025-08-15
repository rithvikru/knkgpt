#!/bin/bash
# Simple setup script using uv in pip mode (no pyproject.toml required)

set -e

echo "🚀 Setting up Knights and Knaves GPT project with uv (simple mode)..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Add to PATH for current session
    export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "📋 uv version: $(uv --version)"

# Create virtual environment with Python 3.12
echo "🐍 Creating virtual environment with Python 3.12..."
uv venv --python 3.12

# Install dependencies from requirements.txt
echo "📚 Installing dependencies..."
uv pip install -r requirements.txt

echo "✅ Setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To start training, run:"
echo "  ./launch_training.sh"