"""
trainer.py — Boucle d'entraînement principale du Student.

Gère :
- L'entraînement sur données pseudo-labélisées
- Le monitoring (loss, métriques, learning rate)
- Les checkpoints (sauvegarde automatique)
- L'early stopping
- Le gradient clipping

Hyperparamètres recommandés (Phase 4.2 de la roadmap) :
    batch_size = 16
    learning_rate = 1e-4
    epochs = 20
    optimizer = AdamW (weight_decay=0.01)
    scheduler = CosineAnnealingLR

Ref: Phase 4.2-4.3 de la roadmap
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Dict
from pathlib import Path


class Trainer:
    """
    Entraîneur principal pour le modèle Student.

    Args:
        model: Modèle Student à entraîner.
        criterion: Fonction de perte (DepthAnythingLoss).
        train_loader: DataLoader d'entraînement.
        val_loader: DataLoader de validation.
        config: Dictionnaire de configuration (hyperparamètres).
        device: Device de calcul.
        output_dir: Répertoire de sortie (checkpoints, logs).
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[dict] = None,
        device: str = "cuda",
        output_dir: str = "outputs",
    ):
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = Path(output_dir)

        # Configuration par défaut
        self.config = {
            "epochs": 20,
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "batch_size": 16,
            "gradient_clip_max_norm": 1.0,
            "checkpoint_interval": 2,       # Sauvegarder tous les N epochs
            "early_stopping_patience": 5,   # Arrêter si val loss stagne
            "log_interval": 50,             # Logger tous les N batches
        }
        if config:
            self.config.update(config)

        # Optimiseur
        self.optimizer = AdamW(
            model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )

        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config["epochs"],
        )

        # État de l'entraînement
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0
        self.training_history = []

        # Créer les répertoires de sortie
        (self.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "logs").mkdir(parents=True, exist_ok=True)

        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.output_dir / "logs"))
        print(f"  TensorBoard logs : {self.output_dir / 'logs'}")

        # Compteur global de steps pour TensorBoard
        self.global_step = 0

    def train(self) -> dict:
        """
        Lance l'entraînement complet.

        Returns:
            Dict avec l'historique d'entraînement.
        """
        print(f"{'='*60}")
        print(f"Début de l'entraînement")
        print(f"  Epochs : {self.config['epochs']}")
        print(f"  LR : {self.config['learning_rate']}")
        print(f"  Device : {self.device}")
        print(f"  Paramètres : {sum(p.numel() for p in self.model.parameters() if p.requires_grad)/1e6:.1f}M")
        print(f"{'='*60}")

        for epoch in range(self.current_epoch, self.config["epochs"]):
            self.current_epoch = epoch

            # Entraînement d'une epoch
            train_metrics = self._train_one_epoch(epoch)

            # Validation
            val_metrics = {}
            if self.val_loader is not None:
                val_metrics = self._validate(epoch)

            # Scheduler step
            self.scheduler.step()

            # Sauvegarder l'historique
            epoch_record = {
                "epoch": epoch,
                "lr": self.scheduler.get_last_lr()[0],
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            self.training_history.append(epoch_record)

            # Checkpoint
            if (epoch + 1) % self.config["checkpoint_interval"] == 0:
                self._save_checkpoint(epoch, val_metrics.get("loss", train_metrics["loss"]))

            # Early stopping
            if val_metrics:
                val_loss = val_metrics["loss"]
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.epochs_without_improvement = 0
                    self._save_checkpoint(epoch, val_loss, is_best=True)
                else:
                    self.epochs_without_improvement += 1

                if self.epochs_without_improvement >= self.config["early_stopping_patience"]:
                    print(f"\nEarly stopping à l'epoch {epoch} (patience={self.config['early_stopping_patience']})")
                    break

            # Affichage
            lr = self.scheduler.get_last_lr()[0]
            msg = f"Epoch {epoch+1}/{self.config['epochs']} | LR: {lr:.2e} | Train Loss: {train_metrics['loss']:.4f}"
            if val_metrics:
                msg += f" | Val Loss: {val_metrics['loss']:.4f}"
            print(msg)

        # Fermer TensorBoard
        self.writer.close()

        return {"history": self.training_history}

    def _train_one_epoch(self, epoch: int) -> dict:
        """Entraîne une epoch complète."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            images = batch["image"].to(self.device)
            targets = batch.get("pseudo_depth", batch.get("depth")).to(self.device)

            # Forward
            predictions = self.model(images)
            loss_dict = self.criterion(predictions, targets)
            loss = loss_dict["total"]

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config["gradient_clip_max_norm"],
            )

            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            self.global_step += 1

            # TensorBoard : loss par step
            self.writer.add_scalar("train/loss_step", loss.item(), self.global_step)

            # Log intermédiaire
            if (batch_idx + 1) % self.config["log_interval"] == 0:
                avg_loss = total_loss / n_batches
                print(f"  Batch {batch_idx+1}/{len(self.train_loader)} | Loss: {avg_loss:.4f}")

        avg_loss = total_loss / max(n_batches, 1)

        # TensorBoard : loss moyenne par epoch
        self.writer.add_scalar("train/loss_epoch", avg_loss, epoch)
        self.writer.add_scalar("train/lr", self.scheduler.get_last_lr()[0], epoch)

        return {"loss": avg_loss}

    @torch.no_grad()
    def _validate(self, epoch: int) -> dict:
        """Évalue le modèle sur le set de validation."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in self.val_loader:
            images = batch["image"].to(self.device)
            targets = batch.get("pseudo_depth", batch.get("depth")).to(self.device)

            predictions = self.model(images)
            loss_dict = self.criterion(predictions, targets)

            total_loss += loss_dict["total"].item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        # TensorBoard : validation loss par epoch
        self.writer.add_scalar("val/loss", avg_loss, epoch)

        return {"loss": avg_loss}

    def _save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """Sauvegarde un checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "loss": loss,
            "config": self.config,
            "training_history": self.training_history,
        }

        # Checkpoint régulier
        path = self.output_dir / "checkpoints" / f"checkpoint_epoch_{epoch+1:03d}.pt"
        torch.save(checkpoint, path)

        # Meilleur modèle
        if is_best:
            best_path = self.output_dir / "checkpoints" / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"  Meilleur modèle sauvegardé (loss={loss:.4f})")

    def resume_from_checkpoint(self, checkpoint_path: str):
        """Reprend l'entraînement depuis un checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"] + 1
        self.training_history = checkpoint.get("training_history", [])
        print(f"Reprise depuis l'epoch {self.current_epoch}")
