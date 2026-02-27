"""
checkpoint.py — Gestion des checkpoints de modèle.

Fonctionnalités :
- Sauvegarde automatique
- Sélection du meilleur modèle
- Reprise après interruption
- Nettoyage des anciens checkpoints

Ref: Phase 4.2 de la roadmap
"""

import os
import torch
from pathlib import Path
from typing import Optional, Dict


class CheckpointManager:
    """
    Gestionnaire de checkpoints pour l'entraînement.

    Args:
        checkpoint_dir: Répertoire de sauvegarde.
        max_checkpoints: Nombre max de checkpoints conservés.
    """

    def __init__(
        self,
        checkpoint_dir: str = "outputs/checkpoints",
        max_checkpoints: int = 5,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.checkpoints = []

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        scheduler=None,
        is_best: bool = False,
        extra: Optional[dict] = None,
    ) -> str:
        """
        Sauvegarde un checkpoint.

        Args:
            model: Modèle à sauvegarder.
            optimizer: Optimiseur.
            epoch: Numéro d'epoch.
            metrics: Métriques courantes.
            scheduler: Scheduler LR (optionnel).
            is_best: Si True, sauvegarde aussi comme best_model.pt.
            extra: Données supplémentaires (optionnel).

        Returns:
            Chemin du checkpoint sauvegardé.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        }

        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        if extra is not None:
            checkpoint.update(extra)

        # Sauvegarde
        path = self.checkpoint_dir / f"checkpoint_epoch_{epoch+1:03d}.pt"
        torch.save(checkpoint, path)
        self.checkpoints.append(str(path))

        # Meilleur modèle
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)

        # Nettoyage des anciens checkpoints
        self._cleanup()

        return str(path)

    def load(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler=None,
        device: str = "cpu",
    ) -> dict:
        """
        Charge un checkpoint.

        Args:
            checkpoint_path: Chemin du checkpoint.
            model: Modèle à restaurer.
            optimizer: Optimiseur à restaurer (optionnel).
            scheduler: Scheduler à restaurer (optionnel).
            device: Device cible.

        Returns:
            Dict du checkpoint (epoch, metrics, etc.).
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        return checkpoint

    def load_best(self, model: torch.nn.Module, device: str = "cpu") -> dict:
        """Charge le meilleur modèle."""
        best_path = self.checkpoint_dir / "best_model.pt"
        if not best_path.exists():
            raise FileNotFoundError(f"Pas de best_model.pt dans {self.checkpoint_dir}")
        return self.load(str(best_path), model, device=device)

    def _cleanup(self):
        """Supprime les checkpoints les plus anciens au-delà de max_checkpoints."""
        while len(self.checkpoints) > self.max_checkpoints:
            old_path = self.checkpoints.pop(0)
            if os.path.exists(old_path) and "best_model" not in old_path:
                os.remove(old_path)

    def list_checkpoints(self) -> list:
        """Liste les checkpoints disponibles."""
        return sorted(self.checkpoint_dir.glob("checkpoint_*.pt"))
