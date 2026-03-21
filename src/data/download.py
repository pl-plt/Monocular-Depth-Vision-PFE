"""
download.py — Fonctions utilitaires de telechargement des datasets.

Chaque fonction delegue au script dedie dans scripts/.
Ce module peut etre importe par d'autres composants (notebooks, pipelines).

Datasets disponibles :
- Synthetiques : Hypersim, Virtual KITTI 2
- Reelles non etiquetees : SA-1B subset
- Benchmarks : NYU-Depth V2 test

Ref: Phase 2 de la roadmap + Appendice A du papier
"""

import subprocess
import sys
from pathlib import Path
from typing import Optional

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"


def _run_script(script_name: str, args: list):
    """Lance un script de telechargement dedie."""
    script_path = SCRIPTS_DIR / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Script introuvable : {script_path}")

    cmd = [sys.executable, str(script_path)] + [str(a) for a in args]
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{script_name} a echoue (code {result.returncode})")


def download_hypersim(
    output_dir: str = "datasets/synthetic/hypersim",
    max_scenes: Optional[int] = None,
    resume: bool = True,
    workers: int = 1,
):
    """
    Telecharge le dataset Hypersim (images synthetiques indoor).

    Delegue a scripts/download_hypersim.py.

    Args:
        output_dir: Repertoire de destination.
        max_scenes: Nombre max de scenes a traiter (None = toutes).
        resume: Reprendre un telechargement interrompu.
        workers: Nombre de telechargements paralleles.
    """
    args = ["--output_dir", output_dir]
    if max_scenes is not None:
        args += ["--max_scenes", str(max_scenes)]
    if resume:
        args.append("--resume")
    if workers > 1:
        args += ["--workers", str(workers)]
    _run_script("download_hypersim.py", args)


def download_virtual_kitti(
    output_dir: str = "datasets/synthetic/vkitti2",
    resume: bool = True,
):
    """
    Telecharge Virtual KITTI 2 (images synthetiques outdoor/driving).

    Delegue a scripts/download_vkitti2.py.

    Args:
        output_dir: Repertoire de destination.
        resume: Reprendre un telechargement interrompu.
    """
    args = ["--output_dir", output_dir]
    if resume:
        args.append("--resume")
    _run_script("download_vkitti2.py", args)


def download_sa1b_subset(
    links_file: str,
    output_dir: str = "datasets/real_unlabeled/sa1b/images",
    n_tars: int = 4,
):
    """
    Telecharge un subset de SA-1B (images reelles non etiquetees).

    Delegue a scripts/download_sa1b.py.

    Args:
        links_file: Chemin vers le fichier TSV de liens SA-1B.
        output_dir: Repertoire de destination.
        n_tars: Nombre de tars a telecharger (~11K images chacun).
    """
    args = [
        "--links_file", links_file,
        "--output_dir", output_dir,
        "--n_tars", str(n_tars),
    ]
    _run_script("download_sa1b.py", args)


def download_nyu_depth_v2_test(
    output_dir: str = "datasets/real_depth/nyudepthv2",
    raw_dir: str = "datasets/raw/nyudepthv2",
    resume: bool = True,
):
    """
    Telecharge/extrait le test set NYU-Depth V2 (654 images, indoor).

    Delegue a scripts/download_nyu_test.py.

    Args:
        output_dir: Repertoire de destination.
        raw_dir: Repertoire pour le .mat brut.
        resume: Reprendre un telechargement interrompu.
    """
    args = ["--output_dir", output_dir, "--raw_dir", raw_dir]
    if resume:
        args.append("--resume")
    _run_script("download_nyu_test.py", args)


def download_indoor_images(
    dataset: str = "all",
    output_dir: str = "datasets/real_unlabeled/indoor",
    resume: bool = True,
):
    """
    Telecharge des images indoor (NYU train, SUN RGB-D, DA-2K).

    Delegue a scripts/download_indoor_images.py.

    Args:
        dataset: Sous-dataset ('nyu', 'sun', 'da_2k', 'all').
        output_dir: Repertoire de destination.
        resume: Reprendre un telechargement interrompu.
    """
    args = ["--dataset", dataset, "--output_dir", output_dir]
    if resume:
        args.append("--resume")
    _run_script("download_indoor_images.py", args)


__all__ = [
    "download_hypersim",
    "download_virtual_kitti",
    "download_sa1b_subset",
    "download_nyu_depth_v2_test",
    "download_indoor_images",
]
