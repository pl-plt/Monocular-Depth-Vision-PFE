"""
helpers.py — Fonctions utilitaires générales.

Inclut : seed, device, comptage paramètres, timer, etc.
"""

import os
import random
import time
import torch
import numpy as np
from typing import Optional
from contextlib import contextmanager


def set_seed(seed: int = 42):
    """
    Fixe la graine aléatoire pour reproductibilité.

    Args:
        seed: Graine aléatoire.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(gpu_id: Optional[int] = None) -> torch.device:
    """
    Retourne le device optimal.

    Args:
        gpu_id: ID du GPU (None = auto-detect).

    Returns:
        torch.device configuré.
    """
    if torch.cuda.is_available():
        if gpu_id is not None:
            device = torch.device(f"cuda:{gpu_id}")
        else:
            device = torch.device("cuda")
        print(f"Device : {torch.cuda.get_device_name(device)}")
        print(f"  Mémoire : {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("Device : CPU (pas de GPU détecté)")

    return device


def count_parameters(model: torch.nn.Module) -> dict:
    """
    Compte les paramètres d'un modèle.

    Args:
        model: Modèle PyTorch.

    Returns:
        Dict avec total, trainable, frozen (en millions).
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "total_M": total / 1e6,
        "trainable_M": trainable / 1e6,
        "frozen_M": frozen / 1e6,
    }


@contextmanager
def timer(description: str = ""):
    """
    Context manager pour mesurer le temps d'exécution.

    Usage:
        with timer("Inférence"):
            model(images)
    """
    start = time.time()
    yield
    elapsed = time.time() - start
    if description:
        print(f"⏱ {description} : {elapsed:.2f}s")


def check_gpu_setup():
    """
    Vérifie la configuration GPU (Phase 0 de la roadmap).

    Affiche : disponibilité CUDA, version, device name, mémoire.
    """
    print("=" * 50)
    print("Vérification GPU Setup")
    print("=" * 50)
    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA disponible : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version    : {torch.version.cuda}")
        print(f"cuDNN version   : {torch.backends.cudnn.version()}")
        n_gpus = torch.cuda.device_count()
        print(f"GPU count       : {n_gpus}")

        for i in range(n_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"\n  GPU {i}: {props.name}")
            print(f"    Mémoire      : {props.total_memory / 1e9:.1f} GB")
            print(f"    Compute cap. : {props.major}.{props.minor}")
    else:
        print("\n⚠ Aucun GPU détecté. L'entraînement sera très lent.")
        print("  Plan B : Google Colab Pro ou AWS EC2 g4dn")

    print("=" * 50)


def denormalize_image(tensor, mean=None, std=None):
    """
    Dénormalise un tensor image pour affichage.

    Args:
        tensor: Image normalisée [C, H, W] ou [B, C, H, W].
        mean: Moyenne ImageNet.
        std: Std ImageNet.

    Returns:
        Image dénormalisée [0, 1].
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]

    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)

    if tensor.dim() == 4:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)

    tensor = tensor * std.to(tensor.device) + mean.to(tensor.device)
    return tensor.clamp(0, 1)
