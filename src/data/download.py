"""
download.py — Scripts de téléchargement des datasets.

Datasets à télécharger :
- Synthétiques : Hypersim, Virtual KITTI 2 (Phase 2, semaines 4-5)
- Réelles non étiquetées : SA-1B subset (Phase 2, semaines 4-5)
- Benchmarks : NYU-Depth V2 test, KITTI test (Phase 0)

Ref: Phase 2 de la roadmap + Appendice A du papier
"""

import os
import argparse
from pathlib import Path
from typing import Optional


def download_hypersim(
    output_dir: str,
    max_samples: int = 50_000,
    scenes: Optional[list] = None,
):
    """
    Télécharge le dataset Hypersim (images synthétiques indoor, ~50k images).

    Source : https://github.com/apple/ml-hypersim
    Stockage estimé : ~50 GB

    Args:
        output_dir: Répertoire de destination.
        max_samples: Nombre max d'images à télécharger.
        scenes: Liste de scènes spécifiques (None = toutes).
    """
    # TODO: Implémenter le téléchargement Hypersim
    # - Utiliser l'API officielle ou téléchargement direct
    # - Échantillonnage stratifié (diversité de scènes)
    # - Sauvegarder images RGB + depth maps
    raise NotImplementedError(
        "Implémenter le téléchargement Hypersim. "
        "Voir : https://github.com/apple/ml-hypersim"
    )


def download_virtual_kitti(
    output_dir: str,
    max_samples: int = 20_000,
):
    """
    Télécharge Virtual KITTI 2 (images synthétiques outdoor/driving).

    Source : https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/
    Stockage estimé : ~15 GB

    Args:
        output_dir: Répertoire de destination.
        max_samples: Nombre max d'images.
    """
    # TODO: Implémenter le téléchargement Virtual KITTI 2
    raise NotImplementedError(
        "Implémenter le téléchargement Virtual KITTI 2."
    )


def download_sa1b_subset(
    output_dir: str,
    max_samples: int = 200_000,
    min_resolution: int = 512,
):
    """
    Télécharge un subset de SA-1B (images réelles non étiquetées).

    Source : https://ai.meta.com/datasets/segment-anything/
    Stockage estimé : 200-500 GB selon le nombre d'images.

    Volume cible progressif :
    - Phase 1 : 50,000 images (proto rapide)
    - Phase 2 : 200,000 images (si Phase 1 OK)
    - Phase 3 : 500,000 images (si temps disponible)

    Args:
        output_dir: Répertoire de destination.
        max_samples: Nombre max d'images.
        min_resolution: Résolution minimale (filtrage).
    """
    # TODO: Implémenter le téléchargement SA-1B subset
    # - Filtrer images trop petites (< min_resolution)
    # - Diversité indoor/outdoor
    raise NotImplementedError(
        "Implémenter le téléchargement SA-1B. "
        "Voir : https://ai.meta.com/datasets/segment-anything/"
    )


def download_nyu_depth_v2_test(output_dir: str):
    """
    Télécharge le test set NYU-Depth V2 (654 images, indoor).

    Utilisé pour l'évaluation (Phase 0 baseline + Phase 5).

    Args:
        output_dir: Répertoire de destination.
    """
    # TODO: Implémenter le téléchargement NYU-D test set
    raise NotImplementedError(
        "Implémenter le téléchargement NYU-Depth V2 test set."
    )


def download_kitti_test(output_dir: str):
    """
    Télécharge le test set KITTI (697 images, outdoor/driving).

    Utilisé pour l'évaluation (Phase 5).

    Args:
        output_dir: Répertoire de destination.
    """
    # TODO: Implémenter le téléchargement KITTI test set
    raise NotImplementedError(
        "Implémenter le téléchargement KITTI eigen split test set."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Téléchargement des datasets")
    parser.add_argument(
        "--dataset",
        choices=["hypersim", "vkitti", "sa1b", "nyu_test", "kitti_test", "all"],
        required=True,
        help="Dataset à télécharger.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets",
        help="Répertoire de sortie.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=50_000,
        help="Nombre max d'échantillons.",
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    download_funcs = {
        "hypersim": lambda: download_hypersim(
            f"{args.output_dir}/synthetic/hypersim", args.max_samples
        ),
        "vkitti": lambda: download_virtual_kitti(
            f"{args.output_dir}/synthetic/vkitti2", args.max_samples
        ),
        "sa1b": lambda: download_sa1b_subset(
            f"{args.output_dir}/real_unlabeled/sa1b", args.max_samples
        ),
        "nyu_test": lambda: download_nyu_depth_v2_test(
            f"{args.output_dir}/benchmarks/nyu_depth_v2"
        ),
        "kitti_test": lambda: download_kitti_test(
            f"{args.output_dir}/benchmarks/kitti"
        ),
    }

    if args.dataset == "all":
        for name, func in download_funcs.items():
            print(f"\n{'='*60}")
            print(f"Téléchargement : {name}")
            print(f"{'='*60}")
            func()
    else:
        download_funcs[args.dataset]()
