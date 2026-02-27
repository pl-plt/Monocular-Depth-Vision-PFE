"""
logging_utils.py — Intégration Weights & Biases (W&B) et TensorBoard.

Fournit un wrapper unifié pour logger les métriques,
les images et les hyperparamètres pendant l'entraînement.

Ref: Phase 4.2 de la roadmap
"""

import os
from typing import Dict, Optional, Any
from pathlib import Path


class LoggerWrapper:
    """
    Wrapper unifié pour W&B et TensorBoard.

    Args:
        backend: "wandb" ou "tensorboard".
        project_name: Nom du projet (pour W&B).
        run_name: Nom du run.
        log_dir: Répertoire pour TensorBoard.
        config: Hyperparamètres à logger.
    """

    def __init__(
        self,
        backend: str = "tensorboard",
        project_name: str = "depth-anything-v2-pfe",
        run_name: Optional[str] = None,
        log_dir: str = "outputs/logs",
        config: Optional[dict] = None,
    ):
        self.backend = backend
        self.writer = None

        if backend == "wandb":
            self._init_wandb(project_name, run_name, config)
        elif backend == "tensorboard":
            self._init_tensorboard(log_dir)

    def _init_wandb(self, project_name, run_name, config):
        """Initialise Weights & Biases."""
        try:
            import wandb
            self.writer = wandb.init(
                project=project_name,
                name=run_name,
                config=config,
            )
        except ImportError:
            print("W&B non installé. Fallback vers TensorBoard.")
            self._init_tensorboard("outputs/logs")

    def _init_tensorboard(self, log_dir):
        """Initialise TensorBoard."""
        try:
            from torch.utils.tensorboard import SummaryWriter
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)
        except ImportError:
            print("TensorBoard non installé. Logging désactivé.")

    def log_scalar(self, tag: str, value: float, step: int):
        """Log une valeur scalaire."""
        if self.writer is None:
            return

        if self.backend == "wandb":
            import wandb
            wandb.log({tag: value}, step=step)
        else:
            self.writer.add_scalar(tag, value, step)

    def log_scalars(self, metrics: Dict[str, float], step: int):
        """Log plusieurs scalaires."""
        for tag, value in metrics.items():
            self.log_scalar(tag, value, step)

    def log_image(self, tag: str, image, step: int):
        """Log une image."""
        if self.writer is None:
            return

        if self.backend == "wandb":
            import wandb
            wandb.log({tag: wandb.Image(image)}, step=step)
        else:
            import torch
            if not isinstance(image, torch.Tensor):
                import numpy as np
                image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            self.writer.add_image(tag, image, step)

    def close(self):
        """Ferme le logger."""
        if self.writer is not None:
            if self.backend == "wandb":
                import wandb
                wandb.finish()
            else:
                self.writer.close()


def setup_logger(
    backend: str = "tensorboard",
    config: Optional[dict] = None,
    **kwargs,
) -> LoggerWrapper:
    """
    Factory pour créer un logger.

    Args:
        backend: "wandb" ou "tensorboard".
        config: Hyperparamètres à logger.

    Returns:
        LoggerWrapper configuré.
    """
    return LoggerWrapper(backend=backend, config=config, **kwargs)
