.PHONY: help setup install sync update clean format lint test train train-single

help:
	@echo "Knights and Knaves GPT - Available commands:"
	@echo "  make setup        - Install uv and set up the project"
	@echo "  make install      - Install dependencies with uv"
	@echo "  make sync         - Sync dependencies with uv"
	@echo "  make update       - Update dependencies to latest versions"
	@echo "  make clean        - Remove virtual environment and cache files"
	@echo "  make format       - Format code with black and ruff"
	@echo "  make lint         - Run linting checks"
	@echo "  make train        - Run distributed training (8 GPUs)"
	@echo "  make train-single - Run single GPU training for testing"

setup:
	@bash scripts/setup.sh

install:
	uv sync

sync:
	uv sync

update:
	uv lock --update-all
	uv sync

clean:
	rm -rf .venv
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

format:
	uv run black .
	uv run ruff check --fix .

lint:
	uv run black --check .
	uv run ruff check .
	uv run mypy .

train:
	./launch_training.sh

train-single:
	bash scripts/run_single_gpu.sh

# Development helpers
dev-install:
	uv sync --all-extras

notebook:
	uv run jupyter notebook train_gpt_knights_knaves.ipynb

wandb-login:
	uv run wandb login