"""
run_inference.py — Phase 0 : Inférence baseline avec le modèle officiel.

Ce script :
1. Charge le modèle Depth-Anything-V2-Small officiel
2. Exécute l'inférence sur des images de test
3. Visualise les depth maps générées
4. Mesure le temps d'inférence moyen

Critères de succès (Phase 0) :
- Temps inférence < 0.5 sec/image
- Modèle officiel tourne sans erreur sur H100

Usage :
    python scripts/run_inference.py --images_dir datasets/benchmarks/nyu_depth_v2/images
    python scripts/run_inference.py --images_dir test_images --weights path/to/weights.pth
"""

import os
import sys
import time
import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image

# Ajouter le répertoire racine au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.helpers import get_device, check_gpu_setup, timer


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 0 — Inférence baseline")
    parser.add_argument("--images_dir", type=str, required=True, help="Répertoire d'images de test")
    parser.add_argument("--weights", type=str, default=None, help="Chemin vers les poids du modèle officiel")
    parser.add_argument("--output_dir", type=str, default="outputs/visualizations/baseline", help="Répertoire de sortie")
    parser.add_argument("--image_size", type=int, default=518, help="Taille de resize")
    parser.add_argument("--max_images", type=int, default=100, help="Nombre max d'images")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Vérifier le setup GPU
    check_gpu_setup()
    device = get_device()

    # 2. Charger le modèle officiel
    print("\n--- Chargement du modèle officiel ---")
    # TODO: Charger Depth-Anything-V2-Small depuis le repo officiel
    # Option 1 : poids GitHub
    # model = torch.hub.load('DepthAnything/Depth-Anything-V2', 'depth_anything_v2_vits')
    # Option 2 : poids locaux
    # from src.models.teacher import TeacherModel
    # model = TeacherModel(pretrained_weights=args.weights)
    print("⚠ TODO: Implémenter le chargement du modèle officiel")
    print("  Récupérer les poids depuis : https://github.com/DepthAnything/Depth-Anything-V2")

    # 3. Charger les images de test
    images_dir = Path(args.images_dir)
    if not images_dir.exists():
        print(f"Répertoire non trouvé : {images_dir}")
        print("Télécharger des images de test (NYU-D ou KITTI) d'abord.")
        return

    extensions = {".jpg", ".jpeg", ".png"}
    image_paths = sorted([
        f for f in images_dir.iterdir()
        if f.suffix.lower() in extensions
    ])[:args.max_images]

    print(f"\n{len(image_paths)} images trouvées dans {images_dir}")

    # 4. Inférence + mesure du temps
    print("\n--- Inférence ---")
    times = []
    os.makedirs(args.output_dir, exist_ok=True)

    # TODO: Décommenter quand le modèle est chargé
    # model.eval()
    # with torch.no_grad():
    #     for img_path in image_paths:
    #         image = Image.open(img_path).convert("RGB")
    #         # Preprocessing...
    #         start = time.time()
    #         depth = model(input_tensor)
    #         elapsed = time.time() - start
    #         times.append(elapsed)
    #         print(f"  {img_path.name}: {elapsed:.3f}s")

    # 5. Résultats
    if times:
        print(f"\n--- Résultats ---")
        print(f"  Images traitées  : {len(times)}")
        print(f"  Temps moyen      : {np.mean(times):.3f}s/image")
        print(f"  Temps total      : {sum(times):.1f}s")
        print(f"  Target           : < 0.5 sec/image")
        print(f"  Status           : {'✅ OK' if np.mean(times) < 0.5 else '❌ Trop lent'}")


if __name__ == "__main__":
    main()
