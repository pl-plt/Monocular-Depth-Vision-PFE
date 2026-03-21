"""
run_inference.py — Inférence avec le modèle Student entraîné.

Charge le Student (DINOv2-Small + DPT), exécute l'inférence sur
un répertoire d'images et sauvegarde les depth maps prédites.

Usage :
    python scripts/run_inference.py \
        --images_dir chemin/vers/images/ \
        --weights outputs/checkpoints/student/best_model.pt

    python scripts/run_inference.py \
        --images_dir datasets/benchmarks/nyu_depth_v2/images \
        --weights outputs/checkpoints/best_model.pt \
        --output_dir outputs/visualizations/inference
"""

import os
import sys
import time
import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.student import StudentModel
from src.utils.helpers import get_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Inférence Student")
    parser.add_argument("--images_dir", type=str, required=True,
                        help="Répertoire d'images de test")
    parser.add_argument("--weights", type=str, required=True,
                        help="Chemin vers les poids du Student (.pt)")
    parser.add_argument("--backbone", type=str, default="dinov2_vits14",
                        help="Backbone DINOv2 du Student")
    parser.add_argument("--output_dir", type=str,
                        default="outputs/visualizations/inference",
                        help="Répertoire de sortie des depth maps")
    parser.add_argument("--image_size", type=int, default=518,
                        help="Taille de resize (multiple de 14)")
    parser.add_argument("--max_images", type=int, default=100,
                        help="Nombre max d'images à traiter")
    return parser.parse_args()


def preprocess(image: Image.Image, image_size: int) -> torch.Tensor:
    """Prétraite une image PIL pour le Student."""
    image = TF.resize(image, (image_size, image_size))
    tensor = TF.to_tensor(image)
    tensor = TF.normalize(tensor, mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
    return tensor.unsqueeze(0)


def main():
    args = parse_args()
    set_seed(42)
    device = get_device()

    # 1. Charger le Student
    print("\n--- Chargement du Student ---")
    student = StudentModel(
        backbone_name=args.backbone,
        pretrained_backbone=True,
        image_size=args.image_size,
    )
    student.load_checkpoint(args.weights)
    student = student.to(device)
    student.eval()

    params = student.count_parameters()
    print(f"  Paramètres : {params['total_M']:.1f}M")

    # 2. Charger les images
    images_dir = Path(args.images_dir)
    if not images_dir.exists():
        print(f"Répertoire non trouvé : {images_dir}")
        sys.exit(1)

    extensions = {".jpg", ".jpeg", ".png"}
    image_paths = sorted([
        f for f in images_dir.iterdir()
        if f.suffix.lower() in extensions
    ])[:args.max_images]

    if not image_paths:
        print(f"Aucune image trouvée dans {images_dir}")
        sys.exit(1)

    print(f"\n{len(image_paths)} images trouvées dans {images_dir}")

    # 3. Inférence
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    times = []

    print("\n--- Inférence ---")
    with torch.no_grad():
        for img_path in image_paths:
            image = Image.open(img_path).convert("RGB")
            input_tensor = preprocess(image, args.image_size).to(device)

            start = time.time()
            depth = student(input_tensor)  # [1, 1, H, W]
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            elapsed = time.time() - start
            times.append(elapsed)

            # Sauvegarder en .npy
            depth_np = depth.squeeze().cpu().numpy()
            np.save(output_dir / f"{img_path.stem}.npy", depth_np)

            print(f"  {img_path.name}: {elapsed:.3f}s")

    # 4. Résultats
    print(f"\n--- Résultats ---")
    print(f"  Images traitées  : {len(times)}")
    print(f"  Temps moyen      : {np.mean(times):.3f}s/image")
    print(f"  Temps total      : {sum(times):.1f}s")
    print(f"  Depth maps dans  : {output_dir}")


if __name__ == "__main__":
    main()
