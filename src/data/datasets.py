"""
datasets.py — Classes Dataset PyTorch pour Depth Anything V2.

Trois types de datasets :
1. SyntheticDepthDataset : Images synthétiques avec GT depth (Hypersim, etc.)
   → Utilisé pour entraîner le Teacher
2. PseudoLabeledDataset : Images réelles + pseudo-labels du Teacher
   → Utilisé pour entraîner le Student (distillation)
3. EvaluationDataset : Benchmarks avec GT depth (NYU-D, KITTI)
   → Utilisé pour évaluer les modèles

Ref: Phase 2 de la roadmap
     Appendice A du papier (sources des données d'entraînement)
"""

import os
import numpy as np
from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image


class SyntheticDepthDataset(Dataset):
    """
    Dataset d'images synthétiques avec depth maps ground truth.

    Sources possibles (Section A du papier, 595K images au total) :
    - Hypersim (indoor)
    - Virtual KITTI 2 (outdoor/driving)
    - TartanAir
    - IRS
    - BlendedMVS

    Structure attendue :
        root/
        ├── images/
        │   ├── 000000.png
        │   └── ...
        └── depth/
            ├── 000000.npy (ou .png 16-bit)
            └── ...

    Args:
        root: Chemin racine du dataset.
        split: "train" ou "val".
        transform: Transformations à appliquer (images + depth).
        max_samples: Nombre maximum d'échantillons (pour debug/proto).
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        max_samples: Optional[int] = None,
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform

        # Charger la liste des fichiers
        self.image_dir = self.root / "images"
        self.depth_dir = self.root / "depth"

        self.samples = self._load_file_list()

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def _load_file_list(self) -> list:
        """Charge la liste des paires (image, depth)."""
        # TODO: Adapter selon la structure réelle des datasets
        image_files = sorted(self.image_dir.glob("*.png")) + \
                      sorted(self.image_dir.glob("*.jpg"))

        samples = []
        for img_path in image_files:
            stem = img_path.stem
            # Chercher le depth map correspondant
            depth_path = self.depth_dir / f"{stem}.npy"
            if not depth_path.exists():
                depth_path = self.depth_dir / f"{stem}.png"
            if depth_path.exists():
                samples.append((str(img_path), str(depth_path)))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            dict avec :
                - "image": Tensor [3, H, W]
                - "depth": Tensor [1, H, W]
                - "path": chemin de l'image source
        """
        img_path, depth_path = self.samples[idx]

        # Charger l'image
        image = Image.open(img_path).convert("RGB")

        # Charger la depth map
        if depth_path.endswith(".npy"):
            depth = np.load(depth_path).astype(np.float32)
        else:
            depth = np.array(Image.open(depth_path)).astype(np.float32)
            depth = depth / 65535.0  # Normaliser si 16-bit PNG

        depth = Image.fromarray(depth)

        # Appliquer les transformations
        if self.transform is not None:
            image, depth = self.transform(image, depth)

        return {
            "image": image,
            "depth": depth,
            "path": img_path,
        }


class PseudoLabeledDataset(Dataset):
    """
    Dataset d'images réelles avec pseudo-labels de profondeur.

    Les pseudo-labels sont générés par le Teacher (Phase 4.1).
    C'est le dataset principal pour entraîner le Student.

    Sources d'images réelles (Section A du papier, 62M images au total) :
    - SA-1B (Segment Anything)
    - Open Images
    - BDD100K
    - etc.

    Structure attendue :
        root/
        ├── images/
        │   ├── 000000.jpg
        │   └── ...
        └── pseudo_labels/
            ├── 000000.npy
            └── ...

    Args:
        image_dir: Répertoire contenant les images réelles.
        pseudo_label_dir: Répertoire contenant les pseudo-labels.
        transform: Transformations à appliquer.
        max_samples: Nombre max d'échantillons.
    """

    def __init__(
        self,
        image_dir: str,
        pseudo_label_dir: str,
        transform: Optional[Callable] = None,
        max_samples: Optional[int] = None,
    ):
        self.image_dir = Path(image_dir)
        self.pseudo_label_dir = Path(pseudo_label_dir)
        self.transform = transform

        self.samples = self._load_file_list()

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def _load_file_list(self) -> list:
        """Charge la liste des paires (image, pseudo-label)."""
        pseudo_files = sorted(self.pseudo_label_dir.glob("*.npy"))

        samples = []
        for pl_path in pseudo_files:
            stem = pl_path.stem
            # Chercher l'image correspondante
            for ext in [".jpg", ".png", ".jpeg"]:
                img_path = self.image_dir / f"{stem}{ext}"
                if img_path.exists():
                    samples.append((str(img_path), str(pl_path)))
                    break

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            dict avec :
                - "image": Tensor [3, H, W]
                - "pseudo_depth": Tensor [1, H, W] (généré par Teacher)
                - "path": chemin de l'image source
        """
        img_path, pl_path = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        pseudo_depth = np.load(pl_path).astype(np.float32)
        pseudo_depth = Image.fromarray(pseudo_depth)

        if self.transform is not None:
            image, pseudo_depth = self.transform(image, pseudo_depth)

        return {
            "image": image,
            "pseudo_depth": pseudo_depth,
            "path": img_path,
        }


class EvaluationDataset(Dataset):
    """
    Dataset d'évaluation avec ground truth (NYU-Depth V2, KITTI).

    Utilisé pour calculer les métriques de benchmark (AbsRel, RMSE, δ1-3).

    Benchmarks supportés :
    - NYU-Depth V2 : 654 images test (indoor, 640x480)
    - KITTI : 697 images test (outdoor/driving)

    Args:
        root: Chemin racine du dataset de test.
        benchmark: "nyu" ou "kitti".
        transform: Transformations d'évaluation (resize, normalize).
    """

    # Configurations par benchmark
    BENCHMARK_CONFIGS = {
        "nyu": {
            "min_depth": 1e-3,
            "max_depth": 10.0,
            "crop": (45, 471, 41, 601),  # Eigen crop (t, b, l, r)
        },
        "kitti": {
            "min_depth": 1e-3,
            "max_depth": 80.0,
            "crop": None,  # Garg crop appliqué séparément
        },
    }

    def __init__(
        self,
        root: str,
        benchmark: str = "nyu",
        transform: Optional[Callable] = None,
    ):
        self.root = Path(root)
        self.benchmark = benchmark
        self.transform = transform
        self.config = self.BENCHMARK_CONFIGS[benchmark]

        self.samples = self._load_file_list()

    def _load_file_list(self) -> list:
        """Charge la liste des paires (image, depth GT)."""
        # TODO: Adapter selon le format exact du benchmark
        image_dir = self.root / "images"
        depth_dir = self.root / "depth"

        image_files = sorted(image_dir.glob("*.*"))
        samples = []
        for img_path in image_files:
            stem = img_path.stem
            for ext in [".npy", ".png"]:
                depth_path = depth_dir / f"{stem}{ext}"
                if depth_path.exists():
                    samples.append((str(img_path), str(depth_path)))
                    break

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            dict avec :
                - "image": Tensor [3, H, W]
                - "depth_gt": Tensor [1, H, W] (ground truth métrique)
                - "path": chemin de l'image source
        """
        img_path, depth_path = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        if depth_path.endswith(".npy"):
            depth_gt = np.load(depth_path).astype(np.float32)
        else:
            depth_gt = np.array(Image.open(depth_path)).astype(np.float32) / 1000.0

        depth_gt = Image.fromarray(depth_gt)

        if self.transform is not None:
            image, depth_gt = self.transform(image, depth_gt)

        return {
            "image": image,
            "depth_gt": depth_gt,
            "path": img_path,
        }
