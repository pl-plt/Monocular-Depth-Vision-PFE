"""
download_data.py — Phase 2 : Téléchargement des datasets.

Point d'entrée pour télécharger tous les datasets nécessaires.

Datasets synthétiques (entraînement Teacher) :
- Hypersim : 50k images indoor
- Virtual KITTI 2 : 20k images outdoor

Datasets réels non étiquetés (pseudo-labels → entraînement Student) :
- SA-1B subset : 50k → 200k → 500k images (progressif)

Benchmarks (évaluation) :
- NYU-Depth V2 test : 654 images
- KITTI test : 697 images

Usage :
    python scripts/download_data.py --dataset all --output_dir datasets
    python scripts/download_data.py --dataset nyu_test --output_dir datasets
    python scripts/download_data.py --dataset hypersim --max_samples 10000
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.download import *

if __name__ == "__main__":
    # Le script utilise directement le __main__ de src/data/download.py
    # Relancer avec les bons arguments :
    import src.data.download
