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
        loss_fn = ScaleInvariantLoss(top_k_percent=0.10)
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


class TestDepthAnythingLoss:
    """Tests de la perte combinée."""

    def test_output_scalar(self):
        """La loss combinée retourne un scalaire."""
        loss_fn = DepthAnythingLoss(alpha=0.5)
        pred = torch.rand(2, 1, 32, 32) + 0.01
        gt = torch.rand(2, 1, 32, 32) + 0.01
        mask = torch.ones(2, 1, 32, 32)
        loss = loss_fn(pred, gt, mask)
        assert loss.dim() == 0

    def test_weighted_combination(self):
        """alpha=0 ⇒ loss == L_ssi seule."""
        loss_fn_alpha0 = DepthAnythingLoss(alpha=0.0)
        loss_fn_ssi = ScaleInvariantLoss()
        pred = torch.rand(2, 1, 32, 32) + 0.01
        gt = torch.rand(2, 1, 32, 32) + 0.01
        mask = torch.ones(2, 1, 32, 32)
        combined = loss_fn_alpha0(pred, gt, mask)
        ssi_only = loss_fn_ssi(pred, gt, mask)
        assert abs(combined.item() - ssi_only.item()) < 1e-5
