"""
metrics.py — Métriques d'évaluation pour l'estimation de profondeur.

Métriques standard (Section 7 du papier) :
- AbsRel : Erreur relative absolue moyenne
- RMSE   : Root Mean Squared Error
- log10  : Erreur log10 moyenne
- δ1     : % de pixels avec max(pred/gt, gt/pred) < 1.25
- δ2     : ... < 1.25²
- δ3     : ... < 1.25³

Valeurs de référence pour DAv2-Small sur NYU-D (Table 2) :
    AbsRel = 0.053  |  δ1 = 0.992

Ref: Phase 5 de la roadmap
"""

import torch
import numpy as np
from typing import Dict, Optional


class DepthMetrics:
    """
    Calcul des métriques d'évaluation de profondeur.

    Supporte les benchmarks NYU-Depth V2 et KITTI avec
    leurs configurations de profondeur respectives.

    Args:
        min_depth: Profondeur minimale (m).
        max_depth: Profondeur maximale (m).
        use_median_scaling: Si True, aligne pred et GT par la médiane.
    """

    def __init__(
        self,
        min_depth: float = 1e-3,
        max_depth: float = 10.0,
        use_median_scaling: bool = True,
    ):
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.use_median_scaling = use_median_scaling

    def compute(
        self,
        prediction: np.ndarray,
        ground_truth: np.ndarray,
        valid_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Calcule toutes les métriques pour une paire pred/GT.

        Args:
            prediction: Depth map prédite [H, W].
            ground_truth: Depth map ground truth [H, W].
            valid_mask: Masque de pixels valides [H, W] (optionnel).

        Returns:
            Dict avec AbsRel, RMSE, log10, δ1, δ2, δ3.
        """
        pred = prediction.copy()
        gt = ground_truth.copy()

        # Masque de validité
        if valid_mask is None:
            valid_mask = (gt > self.min_depth) & (gt < self.max_depth)
        valid_mask = valid_mask & (pred > self.min_depth)

        pred = pred[valid_mask]
        gt = gt[valid_mask]

        if len(pred) == 0:
            return {k: 0.0 for k in ["absrel", "rmse", "log10", "delta1", "delta2", "delta3"]}

        # Alignement par la médiane (pour profondeur relative)
        if self.use_median_scaling:
            scale = np.median(gt) / np.median(pred)
            pred = pred * scale

        # Clip aux bornes
        pred = np.clip(pred, self.min_depth, self.max_depth)

        # === Métriques d'erreur ===

        # AbsRel : (1/n) * Σ |pred - gt| / gt
        absrel = np.mean(np.abs(pred - gt) / gt)

        # RMSE : sqrt((1/n) * Σ (pred - gt)²)
        rmse = np.sqrt(np.mean((pred - gt) ** 2))

        # log10 : (1/n) * Σ |log10(pred) - log10(gt)|
        log10_err = np.mean(np.abs(np.log10(pred) - np.log10(gt)))

        # === Métriques de précision (accuracy thresholds) ===

        # δ_k : % pixels avec max(pred/gt, gt/pred) < 1.25^k
        ratio = np.maximum(pred / gt, gt / pred)
        delta1 = np.mean(ratio < 1.25)
        delta2 = np.mean(ratio < 1.25 ** 2)
        delta3 = np.mean(ratio < 1.25 ** 3)

        return {
            "absrel": float(absrel),
            "rmse": float(rmse),
            "log10": float(log10_err),
            "delta1": float(delta1),
            "delta2": float(delta2),
            "delta3": float(delta3),
        }

    def compute_batch(
        self,
        predictions: list,
        ground_truths: list,
        valid_masks: Optional[list] = None,
    ) -> Dict[str, float]:
        """
        Calcule les métriques moyennes sur un batch d'images.

        Args:
            predictions: Liste de depth maps prédites.
            ground_truths: Liste de depth maps ground truth.
            valid_masks: Liste de masques (optionnel).

        Returns:
            Dict avec les métriques moyennées.
        """
        all_metrics = []
        for i in range(len(predictions)):
            mask = valid_masks[i] if valid_masks else None
            metrics = self.compute(predictions[i], ground_truths[i], mask)
            all_metrics.append(metrics)

        # Moyenner
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])

        return avg_metrics

    @staticmethod
    def format_results(metrics: Dict[str, float]) -> str:
        """Formate les métriques en tableau lisible."""
        lines = [
            f"{'Métrique':<12} {'Valeur':>10}",
            f"{'-'*24}",
            f"{'AbsRel':<12} {metrics['absrel']:>10.4f}",
            f"{'RMSE':<12} {metrics['rmse']:>10.4f}",
            f"{'log10':<12} {metrics['log10']:>10.4f}",
            f"{'δ1':<12} {metrics['delta1']:>10.4f}",
            f"{'δ2':<12} {metrics['delta2']:>10.4f}",
            f"{'δ3':<12} {metrics['delta3']:>10.4f}",
        ]
        return "\n".join(lines)
