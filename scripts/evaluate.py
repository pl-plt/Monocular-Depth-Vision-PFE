"""
evaluate.py — Phase 5 : Évaluation sur benchmarks.

Évalue le modèle Student sur :
- NYU-Depth V2 (654 images, indoor)
- KITTI (697 images, outdoor/driving)

Compare avec les résultats officiels de DAv2-Small :
    NYU-D : AbsRel=0.053, δ1=0.992
    KITTI : AbsRel=0.041, δ1=0.993

Objectifs :
- Minimum : AbsRel < 0.08, δ1 > 0.95 (gap < 30%)
- Moyen   : gap < 20% vs modèle officiel
- Excellent: gap < 20% + ablation studies

Usage :
    python scripts/evaluate.py \
        --checkpoint outputs/checkpoints/best_model.pt \
        --nyu_dir datasets/benchmarks/nyu_depth_v2 \
        --kitti_dir datasets/benchmarks/kitti \
        --output_dir outputs/evaluation
"""

import sys
import json
import argparse
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.student import StudentModel
from src.evaluation.benchmark import BenchmarkEvaluator
from src.evaluation.metrics import DepthMetrics
from src.utils.helpers import get_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 5 — Évaluation sur benchmarks")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Chemin vers le checkpoint du Student")
    parser.add_argument("--backbone", type=str, default="dinov2_vits14",
                        help="Backbone du Student")
    parser.add_argument("--image_size", type=int, default=518,
                        help="Taille de resize")

    # Benchmarks
    parser.add_argument("--nyu_dir", type=str, default=None,
                        help="Chemin vers NYU-Depth V2 test")
    parser.add_argument("--kitti_dir", type=str, default=None,
                        help="Chemin vers KITTI test")

    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size pour l'évaluation")
    parser.add_argument("--output_dir", type=str, default="outputs/evaluation",
                        help="Répertoire de sortie des résultats")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(42)
    device = get_device()

    print("=" * 60)
    print("Phase 5 : Évaluation sur benchmarks")
    print("=" * 60)

    # 1. Charger le modèle Student
    print("\n--- Chargement du Student ---")
    student = StudentModel(
        backbone_name=args.backbone,
        image_size=args.image_size,
    )
    student.load_checkpoint(args.checkpoint)
    student = student.to(device)
    student.eval()

    params = student.count_parameters()
    print(f"  Paramètres : {params['total_M']:.1f}M")

    # 2. Évaluation
    evaluator = BenchmarkEvaluator(
        model=student,
        device=str(device),
        image_size=args.image_size,
    )

    results = evaluator.full_evaluation(
        nyu_dir=args.nyu_dir,
        kitti_dir=args.kitti_dir,
    )

    # 3. Sauvegarder les résultats
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRésultats sauvegardés dans : {results_path}")

    # 4. Tableau récapitulatif
    print("\n" + "=" * 60)
    print("RÉCAPITULATIF")
    print("=" * 60)
    print(f"{'Modèle':<25} {'AbsRel (NYU)':>14} {'δ1 (NYU)':>10} {'Params':>10}")
    print("-" * 60)
    print(f"{'DAv2-Small (officiel)':<25} {'0.053':>14} {'0.992':>10} {'25M':>10}")

    if "nyu" in results:
        nyu = results["nyu"]
        print(f"{'Notre modèle':<25} {nyu['absrel']:>14.4f} {nyu['delta1']:>10.4f} {params['total_M']:>9.1f}M")

        # Gap
        gap_absrel = ((nyu["absrel"] - 0.053) / 0.053) * 100
        gap_delta1 = ((0.992 - nyu["delta1"]) / 0.992) * 100
        print(f"\nGap vs officiel :")
        print(f"  AbsRel : {gap_absrel:+.1f}%")
        print(f"  δ1     : {gap_delta1:+.1f}%")

        # Évaluation du niveau
        if gap_absrel < 20 and gap_delta1 < 5:
            print("\n🏆 Excellence (gap < 20%)")
        elif gap_absrel < 30:
            print("\n✅ Objectif moyen atteint (gap < 30%)")
        elif nyu["absrel"] < 0.08 and nyu["delta1"] > 0.95:
            print("\n✅ Objectif minimum atteint")
        else:
            print("\n⚠ En dessous des objectifs minimum")

    print("\n✅ Évaluation terminée.")


if __name__ == "__main__":
    main()
