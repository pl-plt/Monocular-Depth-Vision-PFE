"""
download_hypersim.py — Téléchargement et préprocessing du dataset Hypersim.

Hypersim (Apple, 2021) : 461 scènes indoor, ~74K images, 1024x768.
Les données brutes sont en HDF5 (HDR color + depth en mètres).

Ce script :
1. Télécharge les fichiers nécessaires depuis le CDN Apple
2. Applique le tone mapping HDR → sRGB (PNG 8-bit)
3. Filtre les depth maps (pixels invalides, inf)
4. Sauvegarde dans la structure attendue par SyntheticDepthDataset :
   datasets/synthetic/hypersim/
   ├── images/   (*.png tone-mapped)
   └── depth/    (*.npy float32, mètres)

Sources :
    - Repo officiel :  https://github.com/apple/ml-hypersim
    - Contrib download : https://github.com/apple/ml-hypersim/tree/main/contrib/99991
    - Papier : https://arxiv.org/abs/2011.02523

Usage :
    # Télécharger + préprocesser tout (attention : ~200-300 GB de téléchargement)
    python scripts/download_hypersim.py \
        --output_dir datasets/synthetic/hypersim \
        --raw_dir datasets/raw/hypersim

    # Limiter à N scènes (pour test rapide)
    python scripts/download_hypersim.py \
        --output_dir datasets/synthetic/hypersim \
        --raw_dir datasets/raw/hypersim \
        --max_scenes 5

    # Reprendre un téléchargement interrompu
    python scripts/download_hypersim.py \
        --output_dir datasets/synthetic/hypersim \
        --raw_dir datasets/raw/hypersim \
        --resume

    # Préprocesser uniquement (déjà téléchargé)
    python scripts/download_hypersim.py \
        --output_dir datasets/synthetic/hypersim \
        --raw_dir datasets/raw/hypersim \
        --skip_download

Dépendances :
    pip install h5py numpy Pillow pandas tqdm requests
"""

import os
import sys
import argparse
import hashlib
import zipfile
import io
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import numpy as np
import h5py
from PIL import Image
from tqdm import tqdm

# ============================================================
# Configuration Hypersim
# ============================================================

# URL de base du CDN Apple
CDN_BASE = "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes"

# Liste officielle des 461 scènes Hypersim.
# Extraite depuis https://github.com/apple/ml-hypersim/blob/main/contrib/99991/download.py
# Les scènes ne sont pas continues (volumes et indices manquants).
_OFFICIAL_SCENES = (
    # volume 001
    "ai_001_001", "ai_001_002", "ai_001_003", "ai_001_004", "ai_001_005",
    "ai_001_006", "ai_001_007", "ai_001_008", "ai_001_009", "ai_001_010",
    # volume 002
    "ai_002_001", "ai_002_002", "ai_002_003", "ai_002_004", "ai_002_005",
    "ai_002_006", "ai_002_007", "ai_002_008", "ai_002_009", "ai_002_010",
    # volume 003 (missing _003)
    "ai_003_001", "ai_003_002", "ai_003_004", "ai_003_005", "ai_003_006",
    "ai_003_007", "ai_003_008", "ai_003_009", "ai_003_010",
    # volume 004
    "ai_004_001", "ai_004_002", "ai_004_003", "ai_004_004", "ai_004_005",
    "ai_004_006", "ai_004_007", "ai_004_008", "ai_004_009", "ai_004_010",
    # volume 005 (missing _002)
    "ai_005_001", "ai_005_003", "ai_005_004", "ai_005_005", "ai_005_006",
    "ai_005_007", "ai_005_008", "ai_005_009", "ai_005_010",
    # volume 006 (missing _005)
    "ai_006_001", "ai_006_002", "ai_006_003", "ai_006_004", "ai_006_006",
    "ai_006_007", "ai_006_008", "ai_006_009", "ai_006_010",
    # volume 007 (missing _003)
    "ai_007_001", "ai_007_002", "ai_007_004", "ai_007_005", "ai_007_006",
    "ai_007_007", "ai_007_008", "ai_007_009", "ai_007_010",
    # volume 008
    "ai_008_001", "ai_008_002", "ai_008_003", "ai_008_004", "ai_008_005",
    "ai_008_006", "ai_008_007", "ai_008_008", "ai_008_009", "ai_008_010",
    # volume 009 (missing _010)
    "ai_009_001", "ai_009_002", "ai_009_003", "ai_009_004", "ai_009_005",
    "ai_009_006", "ai_009_007", "ai_009_008", "ai_009_009",
    # volume 010 (missing _010)
    "ai_010_001", "ai_010_002", "ai_010_003", "ai_010_004", "ai_010_005",
    "ai_010_006", "ai_010_007", "ai_010_008", "ai_010_009",
    # volume 011 (missing _002)
    "ai_011_001", "ai_011_003", "ai_011_004", "ai_011_005", "ai_011_006",
    "ai_011_007", "ai_011_008", "ai_011_009", "ai_011_010",
    # volume 012 (sparse)
    "ai_012_001", "ai_012_004", "ai_012_005", "ai_012_007", "ai_012_009",
    "ai_012_010",
    # volume 013 (sparse)
    "ai_013_001", "ai_013_002", "ai_013_003", "ai_013_004", "ai_013_007",
    "ai_013_009", "ai_013_010",
    # volume 014 (sparse)
    "ai_014_003", "ai_014_006", "ai_014_010",
    # volume 015 (missing _002)
    "ai_015_001", "ai_015_003", "ai_015_004", "ai_015_005", "ai_015_006",
    "ai_015_007", "ai_015_008", "ai_015_009", "ai_015_010",
    # volume 016 (missing _008)
    "ai_016_001", "ai_016_002", "ai_016_003", "ai_016_004", "ai_016_005",
    "ai_016_006", "ai_016_007", "ai_016_009", "ai_016_010",
    # volume 017
    "ai_017_001", "ai_017_002", "ai_017_003", "ai_017_004", "ai_017_005",
    "ai_017_006", "ai_017_007", "ai_017_008", "ai_017_009", "ai_017_010",
    # volume 018
    "ai_018_001", "ai_018_002", "ai_018_003", "ai_018_004", "ai_018_005",
    "ai_018_006", "ai_018_007", "ai_018_008", "ai_018_009", "ai_018_010",
    # volume 019 (missing _005, _010)
    "ai_019_001", "ai_019_002", "ai_019_003", "ai_019_004", "ai_019_006",
    "ai_019_007", "ai_019_008", "ai_019_009",
    # volume 020 absent
    # volume 021 (sparse)
    "ai_021_001", "ai_021_002", "ai_021_003", "ai_021_007", "ai_021_008",
    "ai_021_009", "ai_021_010",
    # volume 022 (missing _008)
    "ai_022_001", "ai_022_002", "ai_022_003", "ai_022_004", "ai_022_005",
    "ai_022_006", "ai_022_007", "ai_022_009", "ai_022_010",
    # volume 023
    "ai_023_001", "ai_023_002", "ai_023_003", "ai_023_004", "ai_023_005",
    "ai_023_006", "ai_023_007", "ai_023_008", "ai_023_009", "ai_023_010",
    # volume 024 (19 scenes, up to _019)
    "ai_024_001", "ai_024_002", "ai_024_003", "ai_024_004", "ai_024_005",
    "ai_024_006", "ai_024_007", "ai_024_008", "ai_024_009", "ai_024_010",
    "ai_024_011", "ai_024_012", "ai_024_013", "ai_024_014", "ai_024_015",
    "ai_024_016", "ai_024_017", "ai_024_018", "ai_024_019",
    # volume 025 absent
    # volume 026 (missing _010; up to _020)
    "ai_026_001", "ai_026_002", "ai_026_003", "ai_026_004", "ai_026_005",
    "ai_026_006", "ai_026_007", "ai_026_008", "ai_026_009", "ai_026_011",
    "ai_026_012", "ai_026_013", "ai_026_014", "ai_026_015", "ai_026_016",
    "ai_026_017", "ai_026_018", "ai_026_019", "ai_026_020",
    # volume 027 (missing _002)
    "ai_027_001", "ai_027_003", "ai_027_004", "ai_027_005", "ai_027_006",
    "ai_027_007", "ai_027_008", "ai_027_009", "ai_027_010",
    # volume 028 (missing _007)
    "ai_028_001", "ai_028_002", "ai_028_003", "ai_028_004", "ai_028_005",
    "ai_028_006", "ai_028_008", "ai_028_009",
    # volume 029
    "ai_029_001", "ai_029_002", "ai_029_003", "ai_029_004", "ai_029_005",
    # volume 030 (missing _006)
    "ai_030_001", "ai_030_002", "ai_030_003", "ai_030_004", "ai_030_005",
    "ai_030_007", "ai_030_008", "ai_030_009", "ai_030_010",
    # volume 031 (missing _002, _005)
    "ai_031_001", "ai_031_003", "ai_031_004", "ai_031_006", "ai_031_007",
    "ai_031_008", "ai_031_009", "ai_031_010",
    # volume 032 (missing _006, _010)
    "ai_032_001", "ai_032_002", "ai_032_003", "ai_032_004", "ai_032_005",
    "ai_032_007", "ai_032_008", "ai_032_009",
    # volume 033 (missing _003, _006)
    "ai_033_001", "ai_033_002", "ai_033_004", "ai_033_005", "ai_033_007",
    "ai_033_008", "ai_033_009", "ai_033_010",
    # volume 034 (missing _004)
    "ai_034_001", "ai_034_002", "ai_034_003", "ai_034_005",
    # volume 035
    "ai_035_001", "ai_035_002", "ai_035_003", "ai_035_004", "ai_035_005",
    "ai_035_006", "ai_035_007", "ai_035_008", "ai_035_009", "ai_035_010",
    # volume 036 (missing _004, _009)
    "ai_036_001", "ai_036_002", "ai_036_003", "ai_036_005", "ai_036_006",
    "ai_036_007", "ai_036_008", "ai_036_010",
    # volume 037
    "ai_037_001", "ai_037_002", "ai_037_003", "ai_037_004", "ai_037_005",
    "ai_037_006", "ai_037_007", "ai_037_008", "ai_037_009", "ai_037_010",
    # volume 038 (missing _001, _003, _008)
    "ai_038_002", "ai_038_004", "ai_038_005", "ai_038_006", "ai_038_007",
    "ai_038_009", "ai_038_010",
    # volume 039 (missing _001)
    "ai_039_002", "ai_039_003", "ai_039_004", "ai_039_005", "ai_039_006",
    "ai_039_007", "ai_039_008", "ai_039_009", "ai_039_010",
    # volume 040 absent
    # volume 041
    "ai_041_001", "ai_041_002", "ai_041_003", "ai_041_004", "ai_041_005",
    "ai_041_006", "ai_041_007", "ai_041_008", "ai_041_009", "ai_041_010",
    # volume 042
    "ai_042_001", "ai_042_002", "ai_042_003", "ai_042_004", "ai_042_005",
    # volume 043 (missing _001)
    "ai_043_002", "ai_043_003", "ai_043_004", "ai_043_005", "ai_043_006",
    "ai_043_007", "ai_043_008", "ai_043_009", "ai_043_010",
    # volume 044
    "ai_044_001", "ai_044_002", "ai_044_003", "ai_044_004", "ai_044_005",
    "ai_044_006", "ai_044_007", "ai_044_008", "ai_044_009", "ai_044_010",
    # volume 045 (sparse)
    "ai_045_001", "ai_045_004", "ai_045_005", "ai_045_006", "ai_045_008",
    "ai_045_010",
    # volume 046
    "ai_046_001", "ai_046_002", "ai_046_003", "ai_046_004", "ai_046_005",
    "ai_046_006", "ai_046_007", "ai_046_008",
    # volume 047 (missing _010)
    "ai_047_001", "ai_047_002", "ai_047_003", "ai_047_004", "ai_047_005",
    "ai_047_006", "ai_047_007", "ai_047_008", "ai_047_009",
    # volume 048
    "ai_048_001", "ai_048_002", "ai_048_003", "ai_048_004", "ai_048_005",
    "ai_048_006", "ai_048_007", "ai_048_008", "ai_048_009", "ai_048_010",
    # volume 049 absent
    # volume 050
    "ai_050_001", "ai_050_002", "ai_050_003", "ai_050_004", "ai_050_005",
    # volume 051
    "ai_051_001", "ai_051_002", "ai_051_003", "ai_051_004", "ai_051_005",
    # volume 052
    "ai_052_001", "ai_052_002", "ai_052_003", "ai_052_004", "ai_052_005",
    "ai_052_006", "ai_052_007", "ai_052_008", "ai_052_009", "ai_052_010",
    # volume 053 (missing _011, _015)
    "ai_053_001", "ai_053_002", "ai_053_003", "ai_053_004", "ai_053_005",
    "ai_053_006", "ai_053_007", "ai_053_008", "ai_053_009", "ai_053_010",
    "ai_053_012", "ai_053_013", "ai_053_014", "ai_053_016", "ai_053_017",
    "ai_053_018", "ai_053_019", "ai_053_020",
    # volume 054
    "ai_054_001", "ai_054_002", "ai_054_003", "ai_054_004", "ai_054_005",
    "ai_054_006", "ai_054_007", "ai_054_008", "ai_054_009", "ai_054_010",
    # volume 055
    "ai_055_001", "ai_055_002", "ai_055_003", "ai_055_004", "ai_055_005",
    "ai_055_006", "ai_055_007", "ai_055_008", "ai_055_009", "ai_055_010",
)


def get_scene_list():
    """
    Retourne la liste officielle des 461 scènes Hypersim.
    Source : https://github.com/apple/ml-hypersim/blob/main/contrib/99991/download.py
    Aucune requête HTTP — pas de faux 404.
    """
    return list(_OFFICIAL_SCENES)


# ============================================================
# Téléchargement
# ============================================================

def download_file(url: str, dest_path: Path, resume: bool = False) -> bool:
    """Télécharge un fichier avec barre de progression et support reprise."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Si le fichier existe déjà et pas de reprise, skip
    if dest_path.exists() and not resume:
        return True
    
    headers = {}
    mode = "wb"
    initial_size = 0
    
    if resume and dest_path.exists():
        initial_size = dest_path.stat().st_size
        headers["Range"] = f"bytes={initial_size}-"
        mode = "ab"
    
    try:
        response = requests.get(url, headers=headers, stream=True, timeout=60)
        
        if response.status_code == 416:
            # Range not satisfiable = file already complete
            return True
        
        if response.status_code not in (200, 206):
            return False
        
        total = int(response.headers.get("content-length", 0)) + initial_size
        
        with open(dest_path, mode) as f:
            with tqdm(
                total=total,
                initial=initial_size,
                unit="B",
                unit_scale=True,
                desc=dest_path.name,
                leave=False,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        return True
    except (requests.RequestException, IOError) as e:
        print(f"  ⚠ Erreur téléchargement {url}: {e}")
        return False


def download_scene(scene_name: str, raw_dir: Path, resume: bool = False) -> bool:
    """
    Télécharge le ZIP d'une scène depuis le CDN Apple.
    Retourne True si le téléchargement a réussi ou si déjà présent.
    """
    zip_url = f"{CDN_BASE}/{scene_name}.zip"
    zip_path = raw_dir / f"{scene_name}.zip"
    extract_dir = raw_dir / scene_name
    
    # Si déjà extrait, skip
    if extract_dir.exists() and any(extract_dir.iterdir()):
        return True
    
    # Télécharger le ZIP
    if not zip_path.exists() or resume:
        success = download_file(zip_url, zip_path, resume=resume)
        if not success:
            return False
    
    # Extraire
    try:
        print(f"  Extraction : {scene_name}...", flush=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(raw_dir)
        # Supprimer le ZIP après extraction pour économiser l'espace
        zip_path.unlink(missing_ok=True)
        return True
    except (zipfile.BadZipFile, IOError) as e:
        print(f"  ⚠ Erreur extraction {scene_name}: {e}")
        return False


# ============================================================
# Tone mapping HDR → sRGB
# ============================================================

def tone_map_hypersim(
    rgb_hdr: np.ndarray,
    render_entity_id: np.ndarray,
    gamma: float = 1.0 / 2.2,
    percentile: int = 90,
    brightness_target: float = 0.8,
) -> np.ndarray:
    """
    Tone mapping officiel Hypersim (Apple).
    
    Args:
        rgb_hdr: Image HDR float32 [H, W, 3].
        render_entity_id: Masque de validité [H, W] int32.
        gamma: Gamma pour la correction.
        percentile: Percentile pour le calcul de luminosité.
        brightness_target: Luminosité cible au percentile.
    
    Returns:
        Image uint8 [H, W, 3] tone-mapped.
    """
    inv_gamma = 1.0 / gamma
    valid_mask = render_entity_id != -1
    
    if np.count_nonzero(valid_mask) == 0:
        return np.zeros(rgb_hdr.shape, dtype=np.uint8)
    
    # Luminosité perceptuelle
    brightness = 0.3 * rgb_hdr[:, :, 0] + 0.59 * rgb_hdr[:, :, 1] + 0.11 * rgb_hdr[:, :, 2]
    brightness_valid = brightness[valid_mask]
    
    eps = 1e-4
    brightness_nth = np.percentile(brightness_valid, percentile)
    
    if brightness_nth < eps:
        scale = 0.0
    else:
        scale = np.power(brightness_target, inv_gamma) / brightness_nth
    
    rgb_tm = np.power(np.maximum(scale * rgb_hdr, 0), gamma)
    rgb_tm = np.clip(rgb_tm, 0, 1)
    
    return (rgb_tm * 255).astype(np.uint8)


# ============================================================
# Préprocessing d'une scène
# ============================================================

def process_scene(
    scene_dir: Path,
    output_images_dir: Path,
    output_depth_dir: Path,
    max_depth: float = 100.0,
) -> int:
    """
    Préprocesse une scène Hypersim : HDF5 → PNG + NPY.
    
    Args:
        scene_dir: Répertoire de la scène extraite.
        output_images_dir: Dossier de sortie pour les images PNG.
        output_depth_dir: Dossier de sortie pour les depth maps NPY.
        max_depth: Profondeur maximale en mètres (clamp).
    
    Returns:
        Nombre d'images traitées.
    """
    images_base = scene_dir / "images"
    if not images_base.exists():
        return 0
    
    scene_name = scene_dir.name
    count = 0
    
    # Trouver toutes les caméras
    cam_dirs = sorted(images_base.glob("scene_cam_*_final_hdf5"))
    
    for cam_dir in cam_dirs:
        cam_name = cam_dir.name.replace("_final_hdf5", "")
        # Ex: scene_cam_00_final_hdf5 → cam_id = "cam_00"
        cam_id = cam_name.replace("scene_", "")
        
        # Répertoire geometry correspondant
        geom_dir = images_base / f"{cam_name}_geometry_hdf5"
        if not geom_dir.exists():
            continue
        
        # Trouver tous les frames
        color_files = sorted(cam_dir.glob("frame.*.color.hdf5"))
        
        for color_file in color_files:
            frame_str = color_file.name.split(".")[1]  # ex: "0000"
            
            # Fichiers associés
            depth_file = geom_dir / f"frame.{frame_str}.depth_meters.hdf5"
            entity_file = geom_dir / f"frame.{frame_str}.render_entity_id.hdf5"
            
            if not depth_file.exists():
                continue
            
            # Nom de sortie unique : scene_cam_frame
            out_name = f"{scene_name}_{cam_id}_{frame_str}"
            out_img = output_images_dir / f"{out_name}.png"
            out_depth = output_depth_dir / f"{out_name}.npy"
            
            # Skip si déjà traité
            if out_img.exists() and out_depth.exists():
                count += 1
                continue
            
            try:
                # Charger color HDR
                with h5py.File(color_file, "r") as f:
                    rgb_hdr = f["dataset"][:].astype(np.float32)
                
                # Charger render_entity_id (masque de validité)
                if entity_file.exists():
                    with h5py.File(entity_file, "r") as f:
                        render_entity_id = f["dataset"][:].astype(np.int32)
                else:
                    # Si pas de masque, considérer tout comme valide
                    render_entity_id = np.zeros(rgb_hdr.shape[:2], dtype=np.int32)
                
                # Charger depth
                with h5py.File(depth_file, "r") as f:
                    depth = f["dataset"][:].astype(np.float32)
                
                # Vérifier que l'image a des pixels valides
                valid_mask = render_entity_id != -1
                if np.count_nonzero(valid_mask) < 100:
                    continue
                
                # Tone mapping HDR → sRGB
                rgb_uint8 = tone_map_hypersim(rgb_hdr, render_entity_id)
                
                # Filtrer depth
                depth[~valid_mask] = 0.0
                depth[~np.isfinite(depth)] = 0.0
                depth = np.clip(depth, 0, max_depth)
                
                # Sauvegarder
                Image.fromarray(rgb_uint8).save(out_img)
                np.save(out_depth, depth)
                
                count += 1
                
            except Exception as e:
                print(f"  ⚠ Erreur frame {out_name}: {e}")
                continue
    
    return count


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Téléchargement et préprocessing du dataset Hypersim"
    )
    parser.add_argument("--output_dir", type=str, default="datasets/synthetic/hypersim",
                        help="Répertoire de sortie (images/ + depth/)")
    parser.add_argument("--raw_dir", type=str, default="datasets/raw/hypersim",
                        help="Répertoire pour les fichiers bruts téléchargés")
    parser.add_argument("--max_scenes", type=int, default=None,
                        help="Nombre max de scènes à traiter (pour test)")
    parser.add_argument("--max_depth", type=float, default=100.0,
                        help="Profondeur max en mètres (clamp)")
    parser.add_argument("--resume", action="store_true",
                        help="Reprendre un téléchargement interrompu")
    parser.add_argument("--skip_download", action="store_true",
                        help="Passer le téléchargement (préprocesser uniquement)")
    parser.add_argument("--skip_preprocess", action="store_true",
                        help="Passer le préprocessing (télécharger uniquement)")
    parser.add_argument("--delete_raw", action="store_true",
                        help="Supprimer les fichiers bruts après préprocessing")
    parser.add_argument("--workers", type=int, default=1,
                        help="Nombre de téléchargements parallèles")
    return parser.parse_args()


def main():
    args = parse_args()
    
    output_dir = Path(args.output_dir)
    raw_dir = Path(args.raw_dir)
    output_images = output_dir / "images"
    output_depth = output_dir / "depth"
    
    output_images.mkdir(parents=True, exist_ok=True)
    output_depth.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Liste des scènes
    all_scenes = get_scene_list()
    if args.max_scenes is not None:
        all_scenes = all_scenes[:args.max_scenes]
    
    print("=" * 60, flush=True)
    print("Hypersim — Téléchargement et préprocessing", flush=True)
    print("=" * 60, flush=True)
    print(f"  Scènes à traiter : {len(all_scenes)}", flush=True)
    print(f"  Raw dir          : {raw_dir}", flush=True)
    print(f"  Output dir       : {output_dir}", flush=True)
    print(f"  Max depth        : {args.max_depth}m", flush=True)
    print(flush=True)
    
    # ----- 1. Téléchargement -----
    if not args.skip_download:
        print("--- Phase 1 : Téléchargement ---", flush=True)
        success_count = 0
        fail_count = 0
        
        for i, scene in enumerate(tqdm(all_scenes, desc="Téléchargement scènes")):
            ok = download_scene(scene, raw_dir, resume=args.resume)
            if ok:
                success_count += 1
            else:
                fail_count += 1
                print(f"  ✗ Scène introuvable ou erreur : {scene}", flush=True)
        
        print(f"\n  Téléchargé : {success_count}/{len(all_scenes)} scènes", flush=True)
        if fail_count > 0:
            print(f"  Échoué     : {fail_count} scènes (certaines n'existent pas, c'est normal)", flush=True)
    
    # ----- 2. Préprocessing -----
    if not args.skip_preprocess:
        print("\n--- Phase 2 : Préprocessing HDF5 → PNG + NPY ---", flush=True)
        total_images = 0
        
        # Trouver les scènes effectivement téléchargées
        scene_dirs = sorted([
            d for d in raw_dir.iterdir()
            if d.is_dir() and d.name.startswith("ai_")
        ])
        
        if not scene_dirs:
            print("  ⚠ Aucune scène trouvée dans le raw_dir.", flush=True)
            print(f"    Vérifier : {raw_dir}", flush=True)
            sys.exit(1)
        
        print(f"  Scènes disponibles : {len(scene_dirs)}", flush=True)
        
        for scene_dir in tqdm(scene_dirs, desc="Préprocessing scènes"):
            n = process_scene(
                scene_dir,
                output_images,
                output_depth,
                max_depth=args.max_depth,
            )
            total_images += n
        
        # Compter les fichiers finaux
        n_images = len(list(output_images.glob("*.png")))
        n_depths = len(list(output_depth.glob("*.npy")))
        
        print(f"\n  Images PNG  : {n_images}", flush=True)
        print(f"  Depth NPY   : {n_depths}", flush=True)
        print(f"  Total traité : {total_images} frames", flush=True)
    
    # ----- 3. Nettoyage (optionnel) -----
    if args.delete_raw:
        import shutil
        print("\n--- Nettoyage des fichiers bruts ---", flush=True)
        shutil.rmtree(raw_dir, ignore_errors=True)
        print(f"  Supprimé : {raw_dir}", flush=True)
    
    print("\n" + "=" * 60, flush=True)
    print("✅ Dataset Hypersim prêt.", flush=True)
    print(f"   {output_dir}/images/  → Images PNG (tone-mapped)", flush=True)
    print(f"   {output_dir}/depth/   → Depth maps NPY (float32, mètres)", flush=True)
    print(f"\n   → Prochaine étape : entraîner le Teacher", flush=True)
    print(f"     python scripts/train_teacher.py --dataset_dir {output_dir}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
