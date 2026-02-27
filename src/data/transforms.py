"""
transforms.py — Transformations et augmentations de données.

Deux pipelines :
- Train : augmentations agressives (flip, crop, color jitter)
- Eval  : seulement resize + normalize

Les transformations sont appliquées simultanément sur l'image et la depth map
pour maintenir la cohérence spatiale.

Ref: Phase 2 de la roadmap (semaines 6-7)
"""

import random
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from typing import Tuple


class PairedTransform:
    """
    Classe de base pour les transformations appliquées
    simultanément sur une image et sa depth map.
    """

    def __call__(
        self,
        image: Image.Image,
        depth: Image.Image,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class TrainTransform(PairedTransform):
    """
    Transformations d'entraînement :
    1. Resize à (518, 518)
    2. Random horizontal flip (p=0.5)
    3. Random crop à (490, 490) — multiple de 14 pour DINOv2
    4. Color jitter (image seulement)
    5. ToTensor + Normalize (ImageNet stats)

    Args:
        image_size: Taille de resize initiale (doit être multiple de 14).
        crop_size: Taille du crop aléatoire (doit être multiple de 14).
        horizontal_flip_prob: Probabilité de flip horizontal.
    """

    def __init__(
        self,
        image_size: int = 518,
        crop_size: int = 490,
        horizontal_flip_prob: float = 0.5,
    ):
        self.image_size = image_size
        self.crop_size = crop_size
        self.horizontal_flip_prob = horizontal_flip_prob

        # Normalisation ImageNet
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __call__(
        self,
        image: Image.Image,
        depth: Image.Image,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Resize
        image = TF.resize(image, (self.image_size, self.image_size))
        depth = TF.resize(depth, (self.image_size, self.image_size), interpolation=TF.InterpolationMode.NEAREST)

        # 2. Random horizontal flip
        if random.random() < self.horizontal_flip_prob:
            image = TF.hflip(image)
            depth = TF.hflip(depth)

        # 3. Random crop
        i, j, h, w = self._get_random_crop_params(
            (self.image_size, self.image_size),
            (self.crop_size, self.crop_size),
        )
        image = TF.crop(image, i, j, h, w)
        depth = TF.crop(depth, i, j, h, w)

        # 4. Color jitter (image seulement)
        image = TF.adjust_brightness(image, random.uniform(0.9, 1.1))
        image = TF.adjust_contrast(image, random.uniform(0.9, 1.1))

        # 5. ToTensor + Normalize
        image = TF.to_tensor(image)
        image = TF.normalize(image, self.mean, self.std)

        depth = torch.from_numpy(np.array(depth)).float().unsqueeze(0)

        return image, depth

    @staticmethod
    def _get_random_crop_params(
        input_size: tuple,
        output_size: tuple,
    ) -> Tuple[int, int, int, int]:
        """Calcule les paramètres du random crop."""
        h_in, w_in = input_size
        h_out, w_out = output_size
        i = random.randint(0, h_in - h_out)
        j = random.randint(0, w_in - w_out)
        return i, j, h_out, w_out


class EvalTransform(PairedTransform):
    """
    Transformations d'évaluation :
    1. Resize à image_size
    2. ToTensor + Normalize (ImageNet stats)

    Pas d'augmentation aléatoire.

    Args:
        image_size: Taille de resize.
    """

    def __init__(self, image_size: int = 518):
        self.image_size = image_size
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __call__(
        self,
        image: Image.Image,
        depth: Image.Image,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        image = TF.resize(image, (self.image_size, self.image_size))
        depth = TF.resize(depth, (self.image_size, self.image_size), interpolation=TF.InterpolationMode.NEAREST)

        image = TF.to_tensor(image)
        image = TF.normalize(image, self.mean, self.std)

        depth = torch.from_numpy(np.array(depth)).float().unsqueeze(0)

        return image, depth


def get_train_transforms(image_size: int = 518, crop_size: int = 490) -> TrainTransform:
    """Factory pour les transformations d'entraînement.
    
    Note: crop_size doit être un multiple de 14 (taille des patches DINOv2).
    """
    return TrainTransform(image_size=image_size, crop_size=crop_size)


def get_eval_transforms(image_size: int = 518) -> EvalTransform:
    """Factory pour les transformations d'évaluation."""
    return EvalTransform(image_size=image_size)
