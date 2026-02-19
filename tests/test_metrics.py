"""
test_metrics.py — Tests unitaires pour les métriques d'évaluation.

Vérifie :
- Le calcul de chaque métrique (AbsRel, RMSE, log10, δ1, δ2, δ3)
- Les cas limites (pred == gt, pred constante)
- Le formatage des résultats
"""

import pytest
import torch
from src.evaluation.metrics import DepthMetrics


class TestDepthMetrics:
    """Tests des métriques de profondeur."""

    @pytest.fixture
    def metrics(self):
        return DepthMetrics(use_median_scaling=False)

    def test_perfect_prediction(self, metrics):
        """Toutes les métriques sont parfaites quand pred == gt."""
        depth = torch.rand(2, 1, 64, 64) + 0.1
        mask = torch.ones(2, 1, 64, 64)
        results = metrics.compute(depth, depth.clone(), mask)

        assert results['abs_rel'] < 1e-5
        assert results['rmse'] < 1e-5
        assert results['log10'] < 1e-5
        assert results['delta_1'] > 0.999
        assert results['delta_2'] > 0.999
        assert results['delta_3'] > 0.999

    def test_abs_rel_positive(self, metrics):
        """AbsRel est toujours >= 0."""
        pred = torch.rand(2, 1, 32, 32) + 0.1
        gt = torch.rand(2, 1, 32, 32) + 0.1
        mask = torch.ones(2, 1, 32, 32)
        results = metrics.compute(pred, gt, mask)
        assert results['abs_rel'] >= 0

    def test_rmse_positive(self, metrics):
        """RMSE est toujours >= 0."""
        pred = torch.rand(2, 1, 32, 32) + 0.1
        gt = torch.rand(2, 1, 32, 32) + 0.1
        mask = torch.ones(2, 1, 32, 32)
        results = metrics.compute(pred, gt, mask)
        assert results['rmse'] >= 0

    def test_delta_thresholds_ordered(self, metrics):
        """δ1 ≤ δ2 ≤ δ3 (seuils croissants ⇒ accuracy croissante)."""
        pred = torch.rand(4, 1, 32, 32) + 0.1
        gt = torch.rand(4, 1, 32, 32) + 0.1
        mask = torch.ones(4, 1, 32, 32)
        results = metrics.compute(pred, gt, mask)
        assert results['delta_1'] <= results['delta_2'] + 1e-6
        assert results['delta_2'] <= results['delta_3'] + 1e-6

    def test_delta_in_range(self, metrics):
        """δ doit être dans [0, 1]."""
        pred = torch.rand(2, 1, 32, 32) + 0.1
        gt = torch.rand(2, 1, 32, 32) + 0.1
        mask = torch.ones(2, 1, 32, 32)
        results = metrics.compute(pred, gt, mask)
        for key in ['delta_1', 'delta_2', 'delta_3']:
            assert 0 <= results[key] <= 1.0 + 1e-6

    def test_format_results(self, metrics):
        """Le formatage retourne une chaîne non vide."""
        pred = torch.rand(2, 1, 32, 32) + 0.1
        gt = torch.rand(2, 1, 32, 32) + 0.1
        mask = torch.ones(2, 1, 32, 32)
        results = metrics.compute(pred, gt, mask)
        formatted = metrics.format_results(results)
        assert isinstance(formatted, str)
        assert len(formatted) > 0
        assert 'AbsRel' in formatted or 'abs_rel' in formatted.lower()

    def test_known_values(self, metrics):
        """Test avec des valeurs connues pour vérifier le calcul."""
        # pred = 2*gt ⇒ abs_rel = |pred-gt|/gt = |2gt-gt|/gt = 1.0
        gt = torch.ones(1, 1, 4, 4) * 2.0
        pred = torch.ones(1, 1, 4, 4) * 4.0
        mask = torch.ones(1, 1, 4, 4)
        results = metrics.compute(pred, gt, mask)
        assert abs(results['abs_rel'] - 1.0) < 1e-3

    def test_with_median_scaling(self):
        """Le median scaling ne doit pas planter."""
        metrics = DepthMetrics(use_median_scaling=True)
        pred = torch.rand(2, 1, 32, 32) + 0.1
        gt = torch.rand(2, 1, 32, 32) + 0.1
        mask = torch.ones(2, 1, 32, 32)
        results = metrics.compute(pred, gt, mask)
        assert isinstance(results, dict)
        assert 'abs_rel' in results
