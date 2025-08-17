"""
Weights & Biases logging utilities.
"""
import wandb
from typing import Dict, Any, Optional
import os


def init_wandb(
    project: str = "knkgpt",
    name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    tags: Optional[list] = None,
    notes: Optional[str] = None,
):
    """Initialize wandb run."""
    run = wandb.init(
        project=project,
        name=name,
        config=config,
        tags=tags,
        notes=notes,
        reinit=True,
    )
    
    # Log code
    wandb.run.log_code(".")
    
    return run


def log_metrics(metrics: Dict[str, float], step: int):
    """Log metrics to wandb."""
    wandb.log(metrics, step=step)


def log_puzzle_examples(examples: list, step: int):
    """Log puzzle examples to wandb as a table."""
    columns = ["Puzzle", "True Solution", "Predicted Solution", "Correct"]
    data = []
    
    for ex in examples:
        data.append([
            ex['puzzle'][:200] + "..." if len(ex['puzzle']) > 200 else ex['puzzle'],
            ex['true_solution'],
            ex['pred_solution'],
            "✓" if ex['correct'] else "✗"
        ])
    
    table = wandb.Table(columns=columns, data=data)
    wandb.log({"puzzle_examples": table}, step=step)


def save_model_artifact(
    model_path: str,
    name: str,
    type: str = "model",
    metadata: Optional[Dict[str, Any]] = None
):
    """Save model as wandb artifact."""
    artifact = wandb.Artifact(
        name=name,
        type=type,
        metadata=metadata
    )
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)


def finish_run():
    """Finish wandb run."""
    wandb.finish()
