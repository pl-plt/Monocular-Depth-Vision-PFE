"""
backbone.py — Wrapper DINOv2 pour extraction de features multi-échelle.

DINOv2 est utilisé comme encodeur (backbone) pour extraire des features visuelles
à partir d'images. Le ViT découpe l'image en patches 14x14 et produit des features
à plusieurs niveaux (multi-scale) nécessaires au décodeur DPT.

Modèles disponibles :
- DINOv2-Giant (1.1B params) → Teacher
- DINOv2-Small (25M params)  → Student

Ref: Section 5 du papier Depth Anything V2
"""

import torch
import torch.nn as nn
from typing import List, Optional


class DINOv2Backbone(nn.Module):
    """
    Backbone DINOv2 pour extraction de features multi-échelle.

    Extrait les features intermédiaires du ViT à 4 niveaux (couches)
    pour les transmettre au décodeur DPT.

    Args:
        model_name: Nom du modèle DINOv2 à charger.
            - "dinov2_vitg14" pour le Teacher (Giant, 1.1B params)
            - "dinov2_vits14" pour le Student (Small, 25M params)
        pretrained: Si True, charge les poids pré-entraînés.
        frozen: Si True, gèle tous les poids (mode inference).
        intermediate_layers: Indices des couches intermédiaires à extraire.
    """

    # Dimensions de sortie par variante DINOv2
    EMBED_DIMS = {
        "dinov2_vits14": 384,    # Small
        "dinov2_vitb14": 768,    # Base
        "dinov2_vitl14": 1024,   # Large
        "dinov2_vitg14": 1536,   # Giant
    }

    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        pretrained: bool = True,
        frozen: bool = False,
        intermediate_layers: Optional[List[int]] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.embed_dim = self.EMBED_DIMS.get(model_name, 384)

        # Charger le modèle DINOv2 depuis torch.hub
        self.backbone = self._load_dinov2(model_name, pretrained)

        # Couches intermédiaires à extraire (4 niveaux pour DPT)
        # Par défaut : couches régulièrement espacées
        if intermediate_layers is None:
            n_layers = self._get_num_layers()
            step = n_layers // 4
            self.intermediate_layers = [
                step - 1,
                2 * step - 1,
                3 * step - 1,
                n_layers - 1,
            ]
        else:
            self.intermediate_layers = intermediate_layers

        # Geler les poids si demandé (Teacher)
        if frozen:
            self.freeze()

    def _load_dinov2(self, model_name: str, pretrained: bool) -> nn.Module:
        """Charge le modèle DINOv2 depuis torch.hub."""
        model = torch.hub.load(
            'facebookresearch/dinov2',
            model_name,
            pretrained=pretrained,
        )
        return model

    def _get_num_layers(self) -> int:
        """Retourne le nombre de couches du transformer."""
        # Small: 12, Base: 12, Large: 24, Giant: 40
        layer_counts = {
            "dinov2_vits14": 12,
            "dinov2_vitb14": 12,
            "dinov2_vitl14": 24,
            "dinov2_vitg14": 40,
        }
        return layer_counts.get(self.model_name, 12)

    def freeze(self):
        """Gèle tous les paramètres du backbone (pour le Teacher)."""
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass : extraction de features multi-échelle.

        Args:
            x: Images d'entrée [B, 3, H, W] (H, W multiples de 14).

        Returns:
            Liste de 4 tenseurs de features intermédiaires,
            chacun de shape [B, embed_dim, H/14, W/14].
        """
        B, C, H, W = x.shape
        h = H // 14
        w = W // 14

        # Extraire les features des couches intermédiaires
        features = self.backbone.get_intermediate_layers(
            x,
            n=self.intermediate_layers,
            reshape=True,  # Retourne directement en [B, C, H', W']
        )

        return list(features)

    @property
    def output_dim(self) -> int:
        """Dimension des features de sortie."""
        return self.embed_dim
