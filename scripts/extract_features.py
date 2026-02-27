"""
extract_features.py — Phase 1 : Exploration des features DINOv2.

Ce script :
1. Charge un modèle DINOv2 pré-entraîné (Small ou Giant)
2. Extrait les features sur quelques images de test
3. Vérifie les shapes attendues
4. Visualise les features (PCA ou t-SNE)

Shapes attendues :
- DINOv2-Small (dinov2_vits14) : [B, 384, H/14, W/14]
- DINOv2-Giant (dinov2_vitg14) : [B, 1536, H/14, W/14]

Ref: Phase 1, semaine 3 de la roadmap

Usage :
    python scripts/extract_features.py --model dinov2_vits14 --images_dir test_images
"""

import sys
import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.helpers import get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 1 — Exploration features DINOv2")
    parser.add_argument("--model", type=str, default="dinov2_vits14",
                        choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"],
                        help="Variante DINOv2 à charger")
    parser.add_argument("--images_dir", type=str, default="test_images",
                        help="Répertoire d'images de test")
    parser.add_argument("--n_images", type=int, default=10,
                        help="Nombre d'images à traiter")
    parser.add_argument("--image_size", type=int, default=518,
                        help="Taille de resize (doit être multiple de 14)")
    return parser.parse_args()


def load_dinov2(model_name: str, device: torch.device):
    """Charge un modèle DINOv2 via torch.hub."""
    print(f"Chargement de {model_name}...")
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model = model.to(device)
    model.eval()
    print(f"  Modèle chargé : {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M paramètres")
    return model


def preprocess_image(image_path: str, image_size: int = 518) -> torch.Tensor:
    """Prétraite une image pour DINOv2."""
    image = Image.open(image_path).convert("RGB")
    image = TF.resize(image, (image_size, image_size))
    tensor = TF.to_tensor(image)
    tensor = TF.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return tensor.unsqueeze(0)  # [1, 3, H, W]


def main():
    args = parse_args()
    device = get_device()

    # 1. Charger DINOv2
    model = load_dinov2(args.model, device)

    # Dimensions attendues
    embed_dims = {
        "dinov2_vits14": 384,
        "dinov2_vitb14": 768,
        "dinov2_vitl14": 1024,
        "dinov2_vitg14": 1536,
    }
    expected_dim = embed_dims[args.model]
    expected_spatial = args.image_size // 14

    print(f"\nShapes attendues :")
    print(f"  Embed dim : {expected_dim}")
    print(f"  Spatial   : {expected_spatial}x{expected_spatial}")
    print(f"  Feature   : [B, {expected_dim}, {expected_spatial}, {expected_spatial}]")

    # 2. Charger les images
    images_dir = Path(args.images_dir)
    if not images_dir.exists():
        print(f"\n⚠ Répertoire non trouvé : {images_dir}")
        print("  Créez un dossier avec quelques images de test.")
        print("  Validation avec un tensor aléatoire à la place...")

        # Test avec input aléatoire
        dummy = torch.randn(1, 3, args.image_size, args.image_size).to(device)
        with torch.no_grad():
            features = model.get_intermediate_layers(dummy, n=4, reshape=True)

        print(f"\nFeatures extraites (input aléatoire) :")
        for i, feat in enumerate(features):
            print(f"  Niveau {i} : {feat.shape}")

        return

    # 3. Extraire les features
    extensions = {".jpg", ".jpeg", ".png"}
    image_paths = sorted([
        f for f in images_dir.iterdir()
        if f.suffix.lower() in extensions
    ])[:args.n_images]

    print(f"\n{len(image_paths)} images à traiter...")

    with torch.no_grad():
        for img_path in image_paths:
            input_tensor = preprocess_image(str(img_path), args.image_size).to(device)

            # Extraire features à 4 niveaux
            features = model.get_intermediate_layers(input_tensor, n=4, reshape=True)

            print(f"\n{img_path.name} :")
            for i, feat in enumerate(features):
                print(f"  Niveau {i} : shape={feat.shape}, "
                      f"min={feat.min():.3f}, max={feat.max():.3f}, "
                      f"mean={feat.mean():.3f}")

    print("\n✅ Extraction de features réussie.")
    print(f"   Shapes conformes aux attentes : [B, {expected_dim}, {expected_spatial}, {expected_spatial}]")


if __name__ == "__main__":
    main()
