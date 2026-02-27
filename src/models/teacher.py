"""
teacher.py — Modèle Teacher (DINOv2-Giant + DPT Decoder).

Le Teacher est le modèle le plus performant, basé sur DINOv2-Giant (1.1B params).
Il est entraîné UNIQUEMENT sur des images synthétiques avec des labels de profondeur
précis, puis utilisé en mode inférence pour générer des pseudo-labels sur les images
réelles non étiquetées.

Pipeline entraînement :
    Image synthétique + GT depth → Teacher (backbone frozen, decoder trainable) → Loss

Pipeline inférence (après entraînement) :
    Image réelle → Teacher (tout frozen) → Pseudo-label depth map

Le Teacher n'est JAMAIS entraîné sur des images réelles.

Ref: Section 5, étape 1 & 2 du papier Depth Anything V2
"""

import torch
import torch.nn as nn
from typing import Optional

from .backbone import DINOv2Backbone
from .decoder import DPTDecoder


class TeacherModel(nn.Module):
    """
    Modèle Teacher pour l'entraînement sur données synthétiques
    puis la génération de pseudo-labels.

    Deux modes :
    - Entraînement : backbone frozen, decoder trainable
    - Inférence : tout frozen (après entraînement)

    Args:
        pretrained_weights: Chemin vers les poids pré-entraînés du Teacher.
            Si None, charge les poids DINOv2-Giant par défaut.
        image_size: Taille des images d'entrée (par défaut 518).
        decoder_hidden_dim: Dimension interne du décodeur DPT.
        freeze_all: Si True, fige tout le modèle (mode inférence).
            Si False, seul le backbone est figé (mode entraînement).
    """

    BACKBONE_NAME = "dinov2_vitg14"
    EMBED_DIM = 1536  # Dimension features DINOv2-Giant

    def __init__(
        self,
        pretrained_weights: Optional[str] = None,
        image_size: int = 518,
        decoder_hidden_dim: int = 256,
        freeze_all: bool = False,
    ):
        super().__init__()
        self.image_size = image_size

        # Backbone DINOv2-Giant (toujours figé)
        self.backbone = DINOv2Backbone(
            model_name=self.BACKBONE_NAME,
            pretrained=True,
            frozen=True,  # Backbone toujours frozen
        )

        # Décodeur DPT (trainable pendant l'entraînement)
        self.decoder = DPTDecoder(
            input_dim=self.EMBED_DIM,
            hidden_dim=decoder_hidden_dim,
            output_dim=1,
            image_size=image_size,
        )

        # Charger les poids pré-entraînés si disponibles
        if pretrained_weights is not None:
            self._load_pretrained(pretrained_weights)

        # Figer tout le modèle si mode inférence
        if freeze_all:
            self.eval()
            self.requires_grad_(False)

    def _load_pretrained(self, weights_path: str):
        """Charge les poids pré-entraînés (officiels ou entraînés sur synthétiques)."""
        print(f"[Teacher] Chargement des poids depuis : {weights_path}")
        checkpoint = torch.load(weights_path, map_location="cpu")
        # Supporter les deux formats : state_dict brut ou checkpoint complet
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        self.load_state_dict(state_dict, strict=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass : génère une depth map à partir d'une image.

        En mode entraînement : les gradients remontent dans le decoder.
        En mode inférence : aucun gradient (tout frozen).

        Args:
            x: Images d'entrée [B, 3, H, W].

        Returns:
            Depth maps [B, 1, H, W].
        """
        with torch.no_grad():
            features = self.backbone(x)
        depth = self.decoder(features, target_size=(x.shape[2], x.shape[3]))
        return depth

    def freeze_for_inference(self):
        """Fige tout le modèle pour la génération de pseudo-labels."""
        self.eval()
        self.requires_grad_(False)
        print("[Teacher] Modèle figé pour inférence.")

    def count_parameters(self) -> dict:
        """Compte les paramètres totaux et entraînables."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "total_M": total / 1e6,
            "trainable_M": trainable / 1e6,
            "frozen_M": (total - trainable) / 1e6,
        }

    @torch.no_grad()
    def generate_pseudo_labels(
        self,
        dataloader: torch.utils.data.DataLoader,
        save_dir: str,
        device: str = "cuda",
    ):
        """
        Génère des pseudo-labels de profondeur pour un dataset entier.

        Args:
            dataloader: DataLoader d'images réelles non étiquetées.
            save_dir: Répertoire de sauvegarde des pseudo-labels.
            device: Device de calcul ("cuda" ou "cpu").

        Ref: Phase 4.1 de la roadmap
        """
        import numpy as np
        from pathlib import Path
        from tqdm import tqdm

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        self.freeze_for_inference()
        self.to(device)

        count = 0
        for batch in tqdm(dataloader, desc="Génération pseudo-labels"):
            images = batch["image"].to(device)
            paths = batch.get("path", [f"{count + i:06d}" for i in range(len(images))])

            depth_maps = self.forward(images)  # [B, 1, H, W]

            for i in range(depth_maps.shape[0]):
                depth_np = depth_maps[i, 0].cpu().numpy().astype(np.float32)
                stem = Path(paths[i]).stem if isinstance(paths[i], str) else f"{count:06d}"
                np.save(save_path / f"{stem}.npy", depth_np)
                count += 1

        print(f"[Teacher] {count} pseudo-labels sauvegardées dans {save_dir}")
