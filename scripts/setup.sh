#!/bin/bash
# Setup script for the project using uv

set -e

echo "🚀 Setting up Knights and Knaves GPT project with uv..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Add to PATH for current session
    export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "📋 uv version: $(uv --version)"

# Install Python if needed
echo "🐍 Setting up Python 3.10..."
uv python install 3.10

# Sync dependencies
echo "📚 Installing dependencies..."
uv sync

# Generate lock file
echo "🔒 Creating lock file..."
uv lock

echo "✅ Setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To start training, run:"
echo "  ./launch_training.sh"