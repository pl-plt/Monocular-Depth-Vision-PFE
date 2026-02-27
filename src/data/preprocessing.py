"""
preprocessing.py — Nettoyage, filtrage et validation des données.

Pipeline de preprocessing (Phase 2, semaines 6-7) :
1. Suppression des images corrompues
2. Filtrage par résolution minimale
3. Vérification cohérence depth maps synthétiques
4. Création des splits train/val (90/10)
5. Statistiques et sanity checks

Ref: Phase 2 de la roadmap
"""

import os
import json
import random
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

from PIL import Image
from tqdm import tqdm


def validate_images(
    image_dir: str,
    min_resolution: int = 512,
    remove_corrupted: bool = False,
) -> dict:
    """
    Valide un répertoire d'images : détecte les fichiers corrompus et trop petits.

    Args:
        image_dir: Répertoire contenant les images.
        min_resolution: Résolution minimale acceptable (largeur ET hauteur).
        remove_corrupted: Si True, supprime les images corrompues.

    Returns:
        Dict avec statistiques : total, valid, corrupted, too_small.
    """
    image_dir = Path(image_dir)
    stats = {"total": 0, "valid": 0, "corrupted": 0, "too_small": 0, "corrupted_files": [], "small_files": []}

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    image_files = [f for f in image_dir.iterdir() if f.suffix.lower() in extensions]
    stats["total"] = len(image_files)

    for img_path in tqdm(image_files, desc="Validation des images"):
        try:
            img = Image.open(img_path)
            img.verify()  # Vérifie intégrité sans charger en mémoire

            # Re-ouvrir pour vérifier taille (verify ferme le fichier)
            img = Image.open(img_path)
            w, h = img.size

            if w < min_resolution or h < min_resolution:
                stats["too_small"] += 1
                stats["small_files"].append(str(img_path))
            else:
                stats["valid"] += 1

        except Exception as e:
            stats["corrupted"] += 1
            stats["corrupted_files"].append(str(img_path))
            if remove_corrupted:
                os.remove(img_path)
                print(f"  Supprimé : {img_path} ({e})")

    return stats


def validate_depth_maps(
    depth_dir: str,
    image_dir: str,
) -> dict:
    """
    Vérifie la cohérence des depth maps synthétiques.

    Contrôles :
    - Chaque image a un depth map correspondant
    - Les depth maps ne contiennent pas de NaN ou Inf
    - Les valeurs sont dans une plage raisonnable

    Args:
        depth_dir: Répertoire des depth maps.
        image_dir: Répertoire des images correspondantes.

    Returns:
        Dict avec statistiques.
    """
    depth_dir = Path(depth_dir)
    image_dir = Path(image_dir)
    stats = {
        "total": 0,
        "valid": 0,
        "missing_pair": 0,
        "has_nan": 0,
        "has_inf": 0,
        "negative_values": 0,
    }

    depth_files = sorted(depth_dir.glob("*.npy")) + sorted(depth_dir.glob("*.png"))
    stats["total"] = len(depth_files)

    for depth_path in tqdm(depth_files, desc="Validation depth maps"):
        stem = depth_path.stem

        # Vérifier existence de l'image correspondante
        image_exists = any(
            (image_dir / f"{stem}{ext}").exists()
            for ext in [".png", ".jpg", ".jpeg"]
        )
        if not image_exists:
            stats["missing_pair"] += 1
            continue

        # Charger et vérifier la depth map
        try:
            if depth_path.suffix == ".npy":
                depth = np.load(depth_path)
            else:
                depth = np.array(Image.open(depth_path)).astype(np.float32)

            if np.any(np.isnan(depth)):
                stats["has_nan"] += 1
            elif np.any(np.isinf(depth)):
                stats["has_inf"] += 1
            elif np.any(depth < 0):
                stats["negative_values"] += 1
            else:
                stats["valid"] += 1

        except Exception:
            stats["missing_pair"] += 1

    return stats


def create_train_val_split(
    data_dir: str,
    val_ratio: float = 0.1,
    seed: int = 42,
    output_file: Optional[str] = None,
) -> Tuple[list, list]:
    """
    Crée un split train/validation (90/10 par défaut).

    Args:
        data_dir: Répertoire contenant les images.
        val_ratio: Proportion du set de validation.
        seed: Graine aléatoire pour reproductibilité.
        output_file: Fichier JSON de sortie (optionnel).

    Returns:
        (train_files, val_files) : Listes de chemins.
    """
    data_dir = Path(data_dir)
    extensions = {".jpg", ".jpeg", ".png"}
    all_files = sorted([
        str(f.relative_to(data_dir))
        for f in data_dir.iterdir()
        if f.suffix.lower() in extensions
    ])

    random.seed(seed)
    random.shuffle(all_files)

    n_val = int(len(all_files) * val_ratio)
    val_files = all_files[:n_val]
    train_files = all_files[n_val:]

    print(f"Split créé : {len(train_files)} train / {len(val_files)} val")

    if output_file:
        split_data = {
            "seed": seed,
            "val_ratio": val_ratio,
            "train": train_files,
            "val": val_files,
        }
        with open(output_file, "w") as f:
            json.dump(split_data, f, indent=2)
        print(f"Split sauvegardé dans : {output_file}")

    return train_files, val_files


def compute_dataset_stats(image_dir: str, n_samples: int = 1000) -> dict:
    """
    Calcule les statistiques du dataset (pour sanity check).

    Args:
        image_dir: Répertoire des images.
        n_samples: Nombre d'images à échantillonner.

    Returns:
        Dict avec mean, std, sizes distribution.
    """
    image_dir = Path(image_dir)
    extensions = {".jpg", ".jpeg", ".png"}
    all_files = [f for f in image_dir.iterdir() if f.suffix.lower() in extensions]

    if len(all_files) > n_samples:
        all_files = random.sample(all_files, n_samples)

    widths, heights = [], []
    pixel_sums = np.zeros(3, dtype=np.float64)
    pixel_sq_sums = np.zeros(3, dtype=np.float64)
    n_pixels = 0

    for img_path in tqdm(all_files, desc="Calcul statistiques"):
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        widths.append(w)
        heights.append(h)

        pixels = np.array(img).astype(np.float64) / 255.0
        pixel_sums += pixels.reshape(-1, 3).sum(axis=0)
        pixel_sq_sums += (pixels.reshape(-1, 3) ** 2).sum(axis=0)
        n_pixels += pixels.shape[0] * pixels.shape[1]

    mean = pixel_sums / n_pixels
    std = np.sqrt(pixel_sq_sums / n_pixels - mean ** 2)

    return {
        "n_samples": len(all_files),
        "mean_rgb": mean.tolist(),
        "std_rgb": std.tolist(),
        "width_range": (min(widths), max(widths)),
        "height_range": (min(heights), max(heights)),
        "avg_resolution": (np.mean(widths), np.mean(heights)),
    }
