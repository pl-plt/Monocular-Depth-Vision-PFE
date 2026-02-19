"""
Module losses — Fonctions de perte pour Depth Anything V2.

Deux losses principales :
- L_ssi : Scale-and-shift invariant loss
- L_gm  : Gradient matching loss (netteté des bords)

Ref: Section 5 du papier
"""

from .scale_invariant import ScaleInvariantLoss
from .gradient_matching import GradientMatchingLoss, DepthAnythingLoss

__all__ = [
    "ScaleInvariantLoss",
    "GradientMatchingLoss",
    "DepthAnythingLoss",
]
