"""
test_metrics.py — Tests unitaires pour les métriques d'évaluation.

Vérifie :
- Le calcul de chaque métrique (AbsRel, RMSE, log10, δ1, δ2, δ3)
- Les cas limites (pred == gt, pred constante)
- Le formatage des résultats
"""

import pytest
import numpy as np
from src.evaluation.metrics import DepthMetrics


class TestDepthMetrics:
    """Tests des métriques de profondeur."""

    @pytest.fixture
    def metrics(self):
        return DepthMetrics(use_median_scaling=False)

    def test_perfect_prediction(self, metrics):
        """Toutes les métriques sont parfaites quand pred == gt."""
        depth = np.random.rand(64, 64).astype(np.float32) + 0.1
        results = metrics.compute(depth, depth.copy())

        assert results['absrel'] < 1e-5
        assert results['rmse'] < 1e-5
        assert results['log10'] < 1e-5
        assert results['delta1'] > 0.999
        assert results['delta2'] > 0.999
        assert results['delta3'] > 0.999

    def test_abs_rel_positive(self, metrics):
        """AbsRel est toujours >= 0."""
        pred = np.random.rand(32, 32).astype(np.float32) + 0.1
        gt = np.random.rand(32, 32).astype(np.float32) + 0.1
        results = metrics.compute(pred, gt)
        assert results['absrel'] >= 0

    def test_rmse_positive(self, metrics):
        """RMSE est toujours >= 0."""
        pred = np.random.rand(32, 32).astype(np.float32) + 0.1
        gt = np.random.rand(32, 32).astype(np.float32) + 0.1
        results = metrics.compute(pred, gt)
        assert results['rmse'] >= 0

    def test_delta_thresholds_ordered(self, metrics):
        """δ1 ≤ δ2 ≤ δ3 (seuils croissants => accuracy croissante)."""
        pred = np.random.rand(32, 32).astype(np.float32) + 0.1
        gt = np.random.rand(32, 32).astype(np.float32) + 0.1
        results = metrics.compute(pred, gt)
        assert results['delta1'] <= results['delta2'] + 1e-6
        assert results['delta2'] <= results['delta3'] + 1e-6

    def test_delta_in_range(self, metrics):
        """δ doit être dans [0, 1]."""
        pred = np.random.rand(32, 32).astype(np.float32) + 0.1
        gt = np.random.rand(32, 32).astype(np.float32) + 0.1
        results = metrics.compute(pred, gt)
        for key in ['delta1', 'delta2', 'delta3']:
            assert 0 <= results[key] <= 1.0 + 1e-6

    def test_format_results(self, metrics):
        """Le formatage retourne une chaîne non vide."""
        pred = np.random.rand(32, 32).astype(np.float32) + 0.1
        gt = np.random.rand(32, 32).astype(np.float32) + 0.1
        results = metrics.compute(pred, gt)
        formatted = DepthMetrics.format_results(results)
        assert isinstance(formatted, str)
        assert len(formatted) > 0
        assert 'AbsRel' in formatted

    def test_known_values(self, metrics):
        """Test avec des valeurs connues pour vérifier le calcul."""
        # pred = 2*gt => abs_rel = |pred-gt|/gt = |2gt-gt|/gt = 1.0
        gt = np.ones((4, 4), dtype=np.float32) * 2.0
        pred = np.ones((4, 4), dtype=np.float32) * 4.0
        results = metrics.compute(pred, gt)
        assert abs(results['absrel'] - 1.0) < 1e-3

    def test_with_median_scaling(self):
        """Le median scaling ne doit pas planter."""
        metrics = DepthMetrics(use_median_scaling=True)
        pred = np.random.rand(32, 32).astype(np.float32) + 0.1
        gt = np.random.rand(32, 32).astype(np.float32) + 0.1
        results = metrics.compute(pred, gt)
        assert isinstance(results, dict)
        assert 'absrel' in results
