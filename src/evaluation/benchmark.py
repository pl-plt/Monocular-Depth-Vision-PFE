"""
benchmark.py — Évaluation sur les benchmarks standards (NYU-Depth V2, KITTI).

Évalue un modèle sur les benchmarks et compare avec les résultats officiels
de Depth Anything V2.

Benchmarks supportés :
- NYU-Depth V2 : 654 images test (indoor, Eigen crop)
- KITTI : 697 images test (outdoor/driving, Garg crop)

Résultats de référence DAv2-Small (Table 2 du papier) :
    NYU-D  : AbsRel=0.053, δ1=0.992
    KITTI  : AbsRel=0.041, δ1=0.993 (estimé)

Ref: Phase 5 de la roadmap + Section 7.2 du papier
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict
from torch.utils.data import DataLoader

from .metrics import DepthMetrics
from ..data.datasets import EvaluationDataset
from ..data.transforms import get_eval_transforms


# Résultats officiels pour comparaison (Table 2 du papier)
OFFICIAL_RESULTS = {
    "DAv2-Small": {
        "nyu": {"absrel": 0.053, "rmse": 0.232, "delta1": 0.992},
        "kitti": {"absrel": 0.041, "delta1": 0.993},
    },
    "DAv2-Base": {
        "nyu": {"absrel": 0.046, "rmse": 0.206, "delta1": 0.994},
    },
    "DAv2-Large": {
        "nyu": {"absrel": 0.043, "rmse": 0.197, "delta1": 0.995},
    },
    "DAv2-Giant": {
        "nyu": {"absrel": 0.038, "rmse": 0.179, "delta1": 0.996},
    },
}


class BenchmarkEvaluator:
    """
    Évalue un modèle sur les benchmarks et compare avec les résultats officiels.

    Args:
        model: Modèle à évaluer (Student ou Teacher).
        device: Device de calcul.
        image_size: Taille de resize pour l'inférence.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cuda",
        image_size: int = 518,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.image_size = image_size

    @torch.no_grad()
    def evaluate(
        self,
        benchmark: str,
        data_dir: str,
        batch_size: int = 8,
    ) -> Dict[str, float]:
        """
        Évalue le modèle sur un benchmark.

        Args:
            benchmark: "nyu" ou "kitti".
            data_dir: Chemin vers les données du benchmark.
            batch_size: Taille des batches.

        Returns:
            Dict avec les métriques.
        """
        # Configuration selon le benchmark
        benchmark_config = EvaluationDataset.BENCHMARK_CONFIGS[benchmark]

        # Dataset et DataLoader
        transform = get_eval_transforms(image_size=self.image_size)
        dataset = EvaluationDataset(
            root=data_dir,
            benchmark=benchmark,
            transform=transform,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
        )

        # Métriques
        evaluator = DepthMetrics(
            min_depth=benchmark_config["min_depth"],
            max_depth=benchmark_config["max_depth"],
        )

        all_predictions = []
        all_ground_truths = []

        print(f"Évaluation sur {benchmark.upper()} ({len(dataset)} images)...")

        for batch in dataloader:
            images = batch["image"].to(self.device)
            depth_gt = batch["depth_gt"]

            # Prédiction
            depth_pred = self.model(images)  # [B, 1, H, W]
            depth_pred = depth_pred.squeeze(1).cpu().numpy()  # [B, H, W]
            depth_gt = depth_gt.squeeze(1).numpy()  # [B, H, W]

            for i in range(len(depth_pred)):
                all_predictions.append(depth_pred[i])
                all_ground_truths.append(depth_gt[i])

        # Calculer les métriques
        metrics = evaluator.compute_batch(all_predictions, all_ground_truths)

        print(f"\nRésultats {benchmark.upper()} :")
        print(DepthMetrics.format_results(metrics))

        return metrics

    def compare_with_official(
        self,
        our_metrics: Dict[str, float],
        benchmark: str,
        reference_model: str = "DAv2-Small",
    ) -> str:
        """
        Compare nos résultats avec les résultats officiels.

        Args:
            our_metrics: Nos métriques calculées.
            benchmark: Nom du benchmark.
            reference_model: Modèle de référence pour la comparaison.

        Returns:
            Tableau de comparaison formaté.
        """
        official = OFFICIAL_RESULTS.get(reference_model, {}).get(benchmark, {})

        lines = [
            f"\n{'='*60}",
            f"Comparaison avec {reference_model} sur {benchmark.upper()}",
            f"{'='*60}",
            f"{'Métrique':<12} {'Notre modèle':>14} {reference_model:>14} {'Gap (%)':>10}",
            f"{'-'*52}",
        ]

        for key in ["absrel", "rmse", "delta1", "delta2", "delta3"]:
            our_val = our_metrics.get(key, None)
            off_val = official.get(key, None)

            if our_val is not None and off_val is not None:
                # Pour absrel/rmse : plus bas = mieux
                # Pour delta : plus haut = mieux
                if key.startswith("delta"):
                    gap = ((off_val - our_val) / off_val) * 100
                else:
                    gap = ((our_val - off_val) / off_val) * 100

                lines.append(
                    f"{key:<12} {our_val:>14.4f} {off_val:>14.4f} {gap:>+9.1f}%"
                )
            elif our_val is not None:
                lines.append(f"{key:<12} {our_val:>14.4f} {'N/A':>14}")

        return "\n".join(lines)

    def full_evaluation(
        self,
        nyu_dir: Optional[str] = None,
        kitti_dir: Optional[str] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Évaluation complète sur tous les benchmarks disponibles.

        Args:
            nyu_dir: Chemin vers NYU-Depth V2 test.
            kitti_dir: Chemin vers KITTI test.

        Returns:
            Dict benchmark → métriques.
        """
        results = {}

        if nyu_dir:
            metrics = self.evaluate("nyu", nyu_dir)
            results["nyu"] = metrics
            print(self.compare_with_official(metrics, "nyu"))

        if kitti_dir:
            metrics = self.evaluate("kitti", kitti_dir)
            results["kitti"] = metrics
            print(self.compare_with_official(metrics, "kitti"))

        return results
