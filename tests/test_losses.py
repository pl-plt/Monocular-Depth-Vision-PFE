"""
test_losses.py — Tests unitaires pour les fonctions de perte.

Vérifie :
- ScaleInvariantLoss : shape, non-négativité, gradient flow
- GradientMatchingLoss : filtres Sobel, shape
- DepthAnythingLoss : combinaison pondérée
"""

import pytest
import torch
from src.losses.scale_invariant import ScaleInvariantLoss
from src.losses.gradient_matching import GradientMatchingLoss, DepthAnythingLoss


class TestScaleInvariantLoss:
    """Tests de la perte scale-invariant."""

    def test_output_scalar(self):
        """La loss doit retourner un scalaire."""
        loss_fn = ScaleInvariantLoss()
        pred = torch.rand(2, 1, 64, 64) + 0.01   # garder > 0
        gt = torch.rand(2, 1, 64, 64) + 0.01
        mask = torch.ones(2, 1, 64, 64)
        loss = loss_fn(pred, gt, mask)
        assert loss.dim() == 0

    def test_non_negative(self):
        """La loss doit être >= 0."""
        loss_fn = ScaleInvariantLoss()
        pred = torch.rand(4, 1, 32, 32) + 0.01
        gt = torch.rand(4, 1, 32, 32) + 0.01
        mask = torch.ones(4, 1, 32, 32)
        loss = loss_fn(pred, gt, mask)
        assert loss.item() >= 0

    def test_zero_on_identical(self):
        """Loss ≈ 0 quand pred == gt."""
        loss_fn = ScaleInvariantLoss()
        depth = torch.rand(2, 1, 32, 32) + 0.01
        mask = torch.ones(2, 1, 32, 32)
        loss = loss_fn(depth, depth.clone(), mask)
        assert loss.item() < 1e-5

    def test_gradient_flows(self):
        """Le gradient doit remonter vers pred."""
        loss_fn = ScaleInvariantLoss()
        pred = torch.rand(2, 1, 32, 32, requires_grad=True) + 0.01
        gt = torch.rand(2, 1, 32, 32) + 0.01
        mask = torch.ones(2, 1, 32, 32)
        loss = loss_fn(pred, gt, mask)
        loss.backward()
        assert pred.grad is not None
        assert pred.grad.abs().sum() > 0

    def test_top_k_masking(self):
        """Top-K masking ne doit pas planter et doit retourner un scalaire."""
        loss_fn = ScaleInvariantLoss(top_k_masking=0.10)
        pred = torch.rand(2, 1, 32, 32) + 0.01
        gt = torch.rand(2, 1, 32, 32) + 0.01
        mask = torch.ones(2, 1, 32, 32)
        loss = loss_fn(pred, gt, mask)
        assert loss.dim() == 0
        assert loss.item() >= 0


class TestGradientMatchingLoss:
    """Tests de la perte gradient matching."""

    def test_output_scalar(self):
        """La loss doit retourner un scalaire."""
        loss_fn = GradientMatchingLoss()
        pred = torch.rand(2, 1, 64, 64)
        gt = torch.rand(2, 1, 64, 64)
        mask = torch.ones(2, 1, 64, 64)
        loss = loss_fn(pred, gt, mask)
        assert loss.dim() == 0

    def test_non_negative(self):
        """La loss doit être >= 0."""
        loss_fn = GradientMatchingLoss()
        pred = torch.rand(4, 1, 32, 32)
        gt = torch.rand(4, 1, 32, 32)
        mask = torch.ones(4, 1, 32, 32)
        loss = loss_fn(pred, gt, mask)
        assert loss.item() >= 0

    def test_zero_on_identical(self):
        """Loss ≈ 0 quand pred == gt (mêmes gradients)."""
        loss_fn = GradientMatchingLoss()
        depth = torch.rand(2, 1, 32, 32)
        mask = torch.ones(2, 1, 32, 32)
        loss = loss_fn(depth, depth.clone(), mask)
        assert loss.item() < 1e-5

    def test_gradient_flows(self):
        """Le gradient doit remonter vers pred."""
        loss_fn = GradientMatchingLoss()
        pred = torch.rand(2, 1, 32, 32, requires_grad=True)
        gt = torch.rand(2, 1, 32, 32)
        mask = torch.ones(2, 1, 32, 32)
        loss = loss_fn(pred, gt, mask)
        loss.backward()
        assert pred.grad is not None

    def test_scale_invariance(self):
        """
        L_gm en log-space doit être (quasi) invariante à un facteur d'échelle global.
        log(k*pred) - log(k*gt) = log(pred) - log(gt) → gradients identiques.
        """
        loss_fn = GradientMatchingLoss()
        torch.manual_seed(0)
        pred = torch.rand(2, 1, 32, 32) + 0.1
        gt = torch.rand(2, 1, 32, 32) + 0.1
        mask = torch.ones(2, 1, 32, 32)

        loss_a = loss_fn(pred, gt, mask).item()
        loss_b = loss_fn(pred * 10, gt * 10, mask).item()

        assert abs(loss_a - loss_b) < 1e-3, (
            f"L_gm n'est pas scale-invariante : loss_a={loss_a:.4f}, loss_b={loss_b:.4f}. "
            f"Les gradients doivent être calculés en log-space."
        )


class TestDepthAnythingLoss:
    """Tests de la perte combinée."""

    def test_output_dict_with_total(self):
        """La loss combinée retourne un dict avec clé 'total' scalaire."""
        loss_fn = DepthAnythingLoss(alpha_gm=0.5)
        pred = torch.rand(2, 1, 32, 32) + 0.01
        gt = torch.rand(2, 1, 32, 32) + 0.01
        result = loss_fn(pred, gt)
        assert isinstance(result, dict)
        assert "total" in result
        assert result["total"].dim() == 0

    def test_weighted_combination(self):
        """alpha_gm=0 ⇒ total ≈ L_ssi seule."""
        loss_fn_alpha0 = DepthAnythingLoss(alpha_gm=0.0)
        loss_fn_ssi = ScaleInvariantLoss()
        pred = torch.rand(2, 1, 32, 32) + 0.01
        gt = torch.rand(2, 1, 32, 32) + 0.01
        mask = torch.ones(2, 1, 32, 32)
        combined = loss_fn_alpha0(pred, gt)
        ssi_only = loss_fn_ssi(pred, gt, mask)
        assert abs(combined["total"].item() - ssi_only.item()) < 1e-5
