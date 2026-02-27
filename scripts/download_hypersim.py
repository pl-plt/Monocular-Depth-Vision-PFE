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

# Volumes disponibles (1-55, certains manquants)
MISSING_VOLUMES = {20, 25, 40, 49}
ALL_VOLUMES = [v for v in range(1, 56) if v not in MISSING_VOLUMES]

# Scenes per volume (format: ai_VVV_NNN)
# On utilise le listing dynamique plutôt qu'une liste statique


def get_scene_list():
    """
    Retourne la liste de toutes les scènes Hypersim connues.
    Format : ai_VVV_NNN où VVV = volume (3 chiffres), NNN = scene (3 chiffres).
    
    On itère les volumes et les scènes (1-20 par volume, pas tous existent).
    """
    scenes = []
    for vol in ALL_VOLUMES:
        for scene_idx in range(1, 21):
            scene_name = f"ai_{vol:03d}_{scene_idx:03d}"
            scenes.append(scene_name)
    return scenes


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
