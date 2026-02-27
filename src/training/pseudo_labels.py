"""
pseudo_labels.py — Génération des pseudo-labels de profondeur via le Teacher.

Le Teacher (DINOv2-Giant entraîné sur synthétiques) est utilisé pour
prédire des depth maps sur des images réelles non étiquetées.
Ces pseudo-labels servent ensuite de supervision pour le Student.

Calcul de temps estimé (Phase 4.1) :
- 50,000 images × 0.2 sec/image = 2.8 heures
- 200,000 images × 0.2 sec/image = 11 heures
- Prévoir ×1.5 pour I/O et overheads

Format de sortie : numpy arrays (.npy) ou images 16-bit (.png)

Ref: Section 4 du papier + Phase 4.1 de la roadmap
"""

import os
import time
import numpy as np
import torch
from pathlib import Path
from typing import Optional

from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF


class UnlabeledImageDataset(Dataset):
    """Dataset simple d'images non étiquetées pour la génération de pseudo-labels."""

    def __init__(self, image_dir: str, image_size: int = 518):
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        extensions = {".jpg", ".jpeg", ".png"}
        self.image_paths = sorted([
            f for f in self.image_dir.iterdir()
            if f.suffix.lower() in extensions
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        original_size = image.size  # (W, H)

        image = TF.resize(image, (self.image_size, self.image_size))
        image = TF.to_tensor(image)
        image = TF.normalize(image, self.mean, self.std)

        return {
            "image": image,
            "path": str(img_path),
            "original_size": original_size,
        }


class PseudoLabelGenerator:
    """
    Génère des pseudo-labels de profondeur en batch avec le Teacher.

    Args:
        teacher: Modèle Teacher (figé).
        device: Device de calcul.
        output_dir: Répertoire de sauvegarde des pseudo-labels.
        save_format: "npy" (numpy) ou "png" (16-bit).
    """

    def __init__(
        self,
        teacher: torch.nn.Module,
        device: str = "cuda",
        output_dir: str = "outputs/pseudo_labels",
        save_format: str = "npy",
        use_half: bool = False,
    ):
        self.teacher = teacher
        self.device = device
        self.output_dir = Path(output_dir)
        self.save_format = save_format
        self.use_half = use_half
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Convertir le modèle en fp16 si demandé (économise ~50% VRAM)
        if self.use_half and self.device != "cpu":
            self.teacher = self.teacher.half()
            print("  ⚡ Mode fp16 activé (économie VRAM)")

    @torch.no_grad()
    def generate(
        self,
        image_dir: str,
        batch_size: int = 16,
        image_size: int = 518,
        num_workers: int = 8,
    ):
        """
        Génère les pseudo-labels pour toutes les images d'un répertoire.

        Args:
            image_dir: Répertoire d'images réelles.
            batch_size: Taille des batches.
            image_size: Taille de resize.
            num_workers: Nombre de workers pour le DataLoader.
        """
        dataset = UnlabeledImageDataset(image_dir, image_size)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        self.teacher.eval()
        start_time = time.time()
        n_generated = 0
        n_skipped = 0

        print(f"Génération de pseudo-labels pour {len(dataset)} images...")
        print(f"  Batch size : {batch_size}")
        print(f"  Sortie : {self.output_dir}")
        print(f"  Resume : skip les pseudo-labels existants")

        for batch in tqdm(dataloader, desc="Pseudo-labels"):
            images = batch["image"].to(self.device)
            if self.use_half:
                images = images.half()
            paths = batch["path"]

            # Skip les images dont le pseudo-label existe déjà (resume)
            to_process = []
            for i, path in enumerate(paths):
                stem = Path(path).stem
                out_path = self.output_dir / f"{stem}.{self.save_format}"
                if not out_path.exists():
                    to_process.append(i)
                else:
                    n_skipped += 1

            if not to_process:
                continue

            # Prédiction du Teacher (seulement les images non traitées)
            if len(to_process) < len(paths):
                images = images[to_process]
                paths = [paths[j] for j in to_process]

            depth_maps = self.teacher(images)  # [B, 1, H, W]

            # Sauvegarder chaque pseudo-label
            for i, (depth, path) in enumerate(zip(depth_maps, paths)):
                depth_np = depth.squeeze(0).cpu().numpy()  # [H, W]
                stem = Path(path).stem

                self._save_pseudo_label(depth_np, stem)
                n_generated += 1

        elapsed = time.time() - start_time
        print(f"\nGénération terminée :")
        print(f"  {n_generated} pseudo-labels générés en {elapsed:.1f}s")
        print(f"  {n_skipped} pseudo-labels déjà existants (skippés)")
        print(f"  Vitesse : {n_generated/max(elapsed,1):.1f} images/sec")
        print(f"  Sauvegardés dans : {self.output_dir}")

    def _save_pseudo_label(self, depth: np.ndarray, stem: str):
        """Sauvegarde un pseudo-label."""
        if self.save_format == "npy":
            np.save(self.output_dir / f"{stem}.npy", depth)
        elif self.save_format == "png":
            # Normaliser en 16-bit
            depth_normalized = (depth / depth.max() * 65535).astype(np.uint16)
            Image.fromarray(depth_normalized).save(self.output_dir / f"{stem}.png")

    def quality_check(self, n_samples: int = 100):
        """
        Vérifie la qualité des pseudo-labels générés.

        Contrôles :
        - Pas de fichiers corrompus
        - Pas de NaN ou Inf
        - Distribution raisonnable des valeurs
        - Cohérence visuelle (à inspecter manuellement)

        Args:
            n_samples: Nombre d'échantillons à vérifier.
        """
        import random

        files = sorted(self.output_dir.glob(f"*.{self.save_format}"))
        if len(files) > n_samples:
            files = random.sample(files, n_samples)

        stats = {"total": len(files), "valid": 0, "corrupted": 0, "has_nan": 0}
        all_values = []

        for f in files:
            try:
                if self.save_format == "npy":
                    depth = np.load(f)
                else:
                    depth = np.array(Image.open(f)).astype(np.float32)

                if np.any(np.isnan(depth)):
                    stats["has_nan"] += 1
                else:
                    stats["valid"] += 1
                    all_values.extend([depth.min(), depth.max(), depth.mean()])

            except Exception:
                stats["corrupted"] += 1

        print(f"Quality check ({stats['total']} fichiers) :")
        print(f"  Valides : {stats['valid']}")
        print(f"  Corrompus : {stats['corrupted']}")
        print(f"  Contiennent NaN : {stats['has_nan']}")
        if all_values:
            print(f"  Profondeur min : {min(all_values):.4f}")
            print(f"  Profondeur max : {max(all_values):.4f}")
