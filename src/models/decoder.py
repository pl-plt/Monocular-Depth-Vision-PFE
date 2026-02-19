"""
decoder.py — Décodeur DPT (Dense Prediction Transformer).

Le décodeur DPT fusionne les features multi-échelle extraites par le backbone
DINOv2 et les upsample progressivement vers la résolution finale pour produire
une carte de profondeur dense.

Architecture :
    Features multi-échelle (4 niveaux)
        → Reassemble blocks (projection + resize)
        → Fusion blocks (convolutions + upsampling progressif)
        → Head (convolution finale → depth map [B, 1, H, W])

Ref: Section 5 du papier + architecture DPT originale (Ranftl et al.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class ReassembleBlock(nn.Module):
    """
    Bloc de réassemblage : projette les features d'une couche intermédiaire
    vers une dimension commune et les redimensionne.

    Args:
        input_dim: Dimension des features d'entrée (embed_dim du backbone).
        output_dim: Dimension de sortie après projection.
        scale_factor: Facteur d'échelle spatiale (>1 = upsample, <1 = downsample).
    """

    def __init__(self, input_dim: int, output_dim: int, scale_factor: float = 1.0):
        super().__init__()
        self.projection = nn.Conv2d(input_dim, output_dim, kernel_size=1)
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Features [B, C_in, H, W]
        Returns:
            Features projetées et redimensionnées [B, C_out, H', W']
        """
        x = self.projection(x)
        if self.scale_factor != 1.0:
            x = F.interpolate(
                x,
                scale_factor=self.scale_factor,
                mode="bilinear",
                align_corners=True,
            )
        return x


class FusionBlock(nn.Module):
    """
    Bloc de fusion : fusionne les features de deux niveaux consécutifs
    via des convolutions résiduelles et upsampling.

    Args:
        channels: Nombre de canaux (identique entrée/sortie).
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Features du niveau courant [B, C, H, W].
            skip: Features du niveau supérieur (optionnel) [B, C, H', W'].
        Returns:
            Features fusionnées et upsamplées [B, C, 2H, 2W].
        """
        if skip is not None:
            # Upsampler x à la taille du skip si nécessaire
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(
                    x, size=skip.shape[-2:], mode="bilinear", align_corners=True
                )
            x = x + skip

        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = x + residual

        # Upsampling x2
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return x


class DPTDecoder(nn.Module):
    """
    Décodeur DPT complet pour l'estimation de profondeur.

    Prend les 4 niveaux de features du backbone DINOv2,
    les fusionne progressivement et produit une depth map.

    Args:
        input_dim: Dimension des features du backbone (embed_dim).
        hidden_dim: Dimension interne du décodeur (par défaut 256).
        output_dim: Nombre de canaux de sortie (1 pour depth map).
        patch_size: Taille des patches du ViT (14 pour DINOv2).
        image_size: Taille de l'image d'entrée (518 par défaut).
    """

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 256,
        output_dim: int = 1,
        patch_size: int = 14,
        image_size: int = 518,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.image_size = image_size

        # Reassemble blocks : projeter les 4 niveaux vers hidden_dim
        self.reassemble_blocks = nn.ModuleList([
            ReassembleBlock(input_dim, hidden_dim, scale_factor=4.0),   # Niveau 1 (le plus profond)
            ReassembleBlock(input_dim, hidden_dim, scale_factor=2.0),   # Niveau 2
            ReassembleBlock(input_dim, hidden_dim, scale_factor=1.0),   # Niveau 3
            ReassembleBlock(input_dim, hidden_dim, scale_factor=0.5),   # Niveau 4 (le plus fin)
        ])

        # Fusion blocks : fusion progressive bottom-up
        self.fusion_blocks = nn.ModuleList([
            FusionBlock(hidden_dim),
            FusionBlock(hidden_dim),
            FusionBlock(hidden_dim),
            FusionBlock(hidden_dim),
        ])

        # Head : convolution finale pour produire la depth map
        self.head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, output_dim, kernel_size=1),
            nn.ReLU(inplace=True),  # Depth toujours positive
        )

    def forward(self, features: List[torch.Tensor], target_size: tuple = None) -> torch.Tensor:
        """
        Forward pass du décodeur DPT.

        Args:
            features: Liste de 4 tenseurs de features multi-échelle
                      [B, embed_dim, H/14, W/14] chacun.
            target_size: (H, W) taille de sortie souhaitée. Si None, utilise image_size.

        Returns:
            Depth map [B, 1, H, W].
        """
        if target_size is None:
            target_size = (self.image_size, self.image_size)

        # Étape 1 : Reassemble — projeter chaque niveau
        reassembled = []
        for i, (block, feat) in enumerate(zip(self.reassemble_blocks, features)):
            reassembled.append(block(feat))

        # Étape 2 : Fusion progressive (bottom-up)
        # On part du niveau le plus profond (index 0)
        fused = self.fusion_blocks[0](reassembled[0])
        for i in range(1, len(self.fusion_blocks)):
            fused = self.fusion_blocks[i](fused, reassembled[i])

        # Étape 3 : Head — prédiction depth
        depth = self.head(fused)

        # Resize à la taille cible
        depth = F.interpolate(
            depth, size=target_size, mode="bilinear", align_corners=True
        )

        return depth
