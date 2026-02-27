"""
gradient_matching.py — Gradient Matching Loss (L_gm) et Loss combinée.

La Gradient Matching Loss pénalise les différences de gradients spatiaux
entre la prédiction et la ground truth, ce qui encourage des bords nets
dans les depth maps.

Formule :
    L_gm = (1/n) * Σ ||∇d_i - ∇d_i*||₁

    où ∇ représente les gradients spatiaux (Sobel filters).

La loss combinée de Depth Anything V2 est :
    L = L_ssi + α * L_gm

Ref: Section 5 du papier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .scale_invariant import ScaleInvariantLoss


class GradientMatchingLoss(nn.Module):
    """
    Gradient Matching Loss pour la netteté des bords.

    Utilise des filtres de Sobel pour calculer les gradients spatiaux
    de la prédiction et de la target, puis calcule la distance L1.

    Args:
        eps: Epsilon pour stabilité numérique.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

        # Filtres de Sobel pour gradients horizontaux et verticaux
        # Kernel Sobel 3x3
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1],
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]

        sobel_y = torch.tensor([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1],
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]

        # Enregistrer comme buffers (non-trainable, déplacés avec .to(device))
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _compute_gradients(self, depth: torch.Tensor) -> tuple:
        """
        Calcule les gradients spatiaux via filtres de Sobel.

        Args:
            depth: Depth map [B, 1, H, W].

        Returns:
            (grad_x, grad_y) : Gradients horizontaux et verticaux.
        """
        grad_x = F.conv2d(depth, self.sobel_x, padding=1)
        grad_y = F.conv2d(depth, self.sobel_y, padding=1)
        return grad_x, grad_y

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Calcul de la loss L_gm.

        Args:
            prediction: Depth maps prédites [B, 1, H, W].
            target: Depth maps ground truth [B, 1, H, W].
            valid_mask: Masque de pixels valides [B, 1, H, W] (optionnel).

        Returns:
            Loss scalaire.
        """
        # Gradients de la prédiction
        pred_gx, pred_gy = self._compute_gradients(prediction)

        # Gradients de la target
        tgt_gx, tgt_gy = self._compute_gradients(target)

        # Distance L1 entre gradients
        diff_x = (pred_gx - tgt_gx).abs()
        diff_y = (pred_gy - tgt_gy).abs()

        if valid_mask is not None:
            diff_x = diff_x * valid_mask.float()
            diff_y = diff_y * valid_mask.float()
            n = valid_mask.sum().clamp(min=1).float()
        else:
            n = prediction.numel()

        loss = (diff_x.sum() + diff_y.sum()) / n
        return loss


class DepthAnythingLoss(nn.Module):
    """
    Loss combinée de Depth Anything V2 : L = L_ssi + α * L_gm.

    Args:
        lambda_ssi: Coefficient λ pour la scale-invariant loss.
        alpha_gm: Poids α de la gradient matching loss.
        top_k_masking: Fraction de pixels à masquer (top-K%).
        use_log: Si True, calcule en espace log-depth.
    """

    def __init__(
        self,
        lambda_ssi: float = 0.5,
        alpha_gm: float = 0.5,
        top_k_masking: float = 0.1,
        use_log: bool = True,
    ):
        super().__init__()
        self.alpha_gm = alpha_gm

        self.ssi_loss = ScaleInvariantLoss(
            lambda_ssi=lambda_ssi,
            use_log=use_log,
            top_k_masking=top_k_masking,
        )
        self.gm_loss = GradientMatchingLoss()

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor = None,
    ) -> dict:
        """
        Calcul de la loss combinée.

        Args:
            prediction: Depth maps prédites [B, 1, H, W].
            target: Depth maps ground truth [B, 1, H, W].
            valid_mask: Masque de pixels valides (optionnel).

        Returns:
            Dict avec "total", "ssi", "gm".
        """
        l_ssi = self.ssi_loss(prediction, target, valid_mask)
        l_gm = self.gm_loss(prediction, target, valid_mask)

        total = l_ssi + self.alpha_gm * l_gm

        return {
            "total": total,
            "ssi": l_ssi.detach(),
            "gm": l_gm.detach(),
        }
