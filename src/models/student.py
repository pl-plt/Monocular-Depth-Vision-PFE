"""
student.py — Modèle Student (ViT-Small + DPT Decoder).

Le Student est le modèle que l'on entraîne via distillation. Il utilise
DINOv2-Small (25M params) comme backbone et un décodeur DPT identique
à celui du Teacher.

Pipeline d'entraînement :
    Image réelle + Pseudo-label (du Teacher)
        → Student prédit depth
        → Loss(prédiction, pseudo-label)
        → Backpropagation

Le backbone est initialisé avec les poids pré-entraînés DINOv2-Small,
le décodeur est initialisé aléatoirement.

Ref: Section 5, étape 3 du papier Depth Anything V2
"""

import torch
import torch.nn as nn
from typing import Optional

from .backbone import DINOv2Backbone
from .decoder import DPTDecoder


class StudentModel(nn.Module):
    """
    Modèle Student pour l'estimation de profondeur.

    Plus léger que le Teacher, il apprend à reproduire les prédictions
    du Teacher via distillation sur des pseudo-labels.

    Args:
        backbone_name: Variante DINOv2 pour le backbone.
            Par défaut "dinov2_vits14" (Small, 25M params).
        pretrained_backbone: Si True, utilise les poids DINOv2 pré-entraînés.
        freeze_backbone: Si True, gèle le backbone (fine-tuning decoder seul).
        image_size: Taille des images d'entrée.
        decoder_hidden_dim: Dimension interne du décodeur DPT.
    """

    def __init__(
        self,
        backbone_name: str = "dinov2_vits14",
        pretrained_backbone: bool = True,
        freeze_backbone: bool = False,
        image_size: int = 518,
        decoder_hidden_dim: int = 256,
    ):
        super().__init__()
        self.image_size = image_size
        self.backbone_name = backbone_name

        # Backbone DINOv2-Small (pré-entraîné, entraînable)
        embed_dim = DINOv2Backbone.EMBED_DIMS.get(backbone_name, 384)
        self.backbone = DINOv2Backbone(
            model_name=backbone_name,
            pretrained=pretrained_backbone,
            frozen=freeze_backbone,
        )

        # Décodeur DPT (même architecture que le Teacher)
        self.decoder = DPTDecoder(
            input_dim=embed_dim,
            hidden_dim=decoder_hidden_dim,
            output_dim=1,
            image_size=image_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass du Student.

        Args:
            x: Images d'entrée [B, 3, H, W].

        Returns:
            Depth maps prédites [B, 1, H, W].
        """
        features = self.backbone(x)
        depth = self.decoder(features, target_size=(x.shape[2], x.shape[3]))
        return depth

    def get_trainable_params(self) -> list:
        """Retourne les paramètres entraînables pour l'optimiseur."""
        return [p for p in self.parameters() if p.requires_grad]

    def count_parameters(self) -> dict:
        """Compte les paramètres totaux et entraînables."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
            "total_M": total / 1e6,
            "trainable_M": trainable / 1e6,
        }

    def load_checkpoint(self, checkpoint_path: str):
        """Charge un checkpoint d'entraînement."""
        print(f"[Student] Chargement checkpoint : {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "model_state_dict" in checkpoint:
            self.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.load_state_dict(checkpoint)
