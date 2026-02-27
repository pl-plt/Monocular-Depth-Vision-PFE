"""
generate_pseudo_labels.py — Phase 4.1 : Génération des pseudo-labels.

Utilise le Teacher (DINOv2-Giant entraîné sur synthétiques) pour prédire
des depth maps sur les images réelles non étiquetées.

Temps de calcul estimé :
- 50,000 images × 0.2 sec = ~2.8 heures
- 200,000 images × 0.2 sec = ~11 heures
- Prévoir ×1.5 pour I/O et overhead

Format de sortie : .npy (numpy float32)

Usage :
    python scripts/generate_pseudo_labels.py \
        --teacher_weights outputs/checkpoints/teacher_best.pt \
        --images_dir datasets/real_unlabeled/sa1b/images \
        --output_dir outputs/pseudo_labels \
        --batch_size 16
"""

import sys
import argparse
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.teacher import TeacherModel
from src.training.pseudo_labels import PseudoLabelGenerator
from src.utils.helpers import get_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 4.1 — Génération pseudo-labels")
    parser.add_argument("--teacher_weights", type=str, required=True,
                        help="Chemin vers les poids du Teacher")
    parser.add_argument("--images_dir", type=str, required=True,
                        help="Répertoire d'images réelles non étiquetées")
    parser.add_argument("--output_dir", type=str, default="outputs/pseudo_labels",
                        help="Répertoire de sortie")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Taille des batches")
    parser.add_argument("--image_size", type=int, default=518,
                        help="Taille de resize")
    parser.add_argument("--save_format", type=str, default="npy",
                        choices=["npy", "png"],
                        help="Format de sauvegarde")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Workers pour le DataLoader")
    parser.add_argument("--quality_check", action="store_true",
                        help="Lancer un quality check après génération")
    parser.add_argument("--half", action="store_true",
                        help="Inférence en fp16 (réduit la VRAM de ~50%%)")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(42)
    device = get_device()

    # 1. Charger le Teacher
    print("=" * 60)
    print("Phase 4.1 : Génération des pseudo-labels")
    print("=" * 60)

    teacher = TeacherModel(
        pretrained_weights=args.teacher_weights,
        image_size=args.image_size,
    ).to(device)

    # 2. Générateur de pseudo-labels
    generator = PseudoLabelGenerator(
        teacher=teacher,
        device=str(device),
        output_dir=args.output_dir,
        save_format=args.save_format,
        use_half=args.half,
    )

    # 3. Générer
    generator.generate(
        image_dir=args.images_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
    )

    # 4. Quality check
    if args.quality_check:
        print("\n--- Quality Check ---")
        generator.quality_check(n_samples=100)

    print("\n✅ Génération terminée.")


if __name__ == "__main__":
    main()
