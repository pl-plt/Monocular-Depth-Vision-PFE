"""
Module training — Boucle d'entraînement, distillation et pseudo-labels.
"""

from .trainer import Trainer
from .distillation import DistillationPipeline
from .pseudo_labels import PseudoLabelGenerator

__all__ = [
    "Trainer",
    "DistillationPipeline",
    "PseudoLabelGenerator",
]
