"""
download_indoor_images.py — Télécharge des images réelles d'intérieurs.

Ces images servent de données non étiquetées pour la génération de
pseudo-labels par le Teacher (Phase 4.1).

Datasets disponibles :
1. NYU Depth V2 (labeled) : 1,449 images indoor RGB (rapide, ~2 GB)
2. SUN RGB-D : ~10,335 images indoor (moyen, ~10 GB)

Les images sont extraites et sauvegardées dans :
    datasets/real_unlabeled/indoor/images/

Usage :
    # NYU Depth V2 labeled (1,449 images — test rapide)
    python scripts/download_indoor_images.py \
        --dataset nyu \
        --output_dir datasets/real_unlabeled/indoor

    # Tout (NYU + SUN RGB-D)
    python scripts/download_indoor_images.py \
        --dataset all \
        --output_dir datasets/real_unlabeled/indoor

Dépendances :
    pip install requests h5py numpy Pillow tqdm
"""

import os
import sys
import argparse
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import requests
import h5py
from PIL import Image
from tqdm import tqdm


# ============================================================
# NYU Depth V2 — 1,449 images indoor RGB
# ============================================================

NYU_URL = "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"
NYU_TRAIN_END = 795  # Eigen split: 0-794 train, 795-1448 test (Phase 5 benchmark)


def download_file(url: str, dest: Path, resume: bool = False) -> bool:
    """Télécharge un fichier avec barre de progression."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    headers = {}
    mode = "wb"
    initial_size = 0

    if resume and dest.exists():
        initial_size = dest.stat().st_size
        headers["Range"] = f"bytes={initial_size}-"
        mode = "ab"
    elif dest.exists():
        return True

    try:
        resp = requests.get(url, headers=headers, stream=True, timeout=120)
        if resp.status_code == 416:
            return True
        if resp.status_code not in (200, 206):
            print(f"  ⚠ HTTP {resp.status_code} pour {url}")
            return False

        total = int(resp.headers.get("content-length", 0)) + initial_size
        with open(dest, mode) as f:
            with tqdm(total=total, initial=initial_size, unit="B",
                      unit_scale=True, desc=dest.name, leave=False) as pbar:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        return True
    except Exception as e:
        print(f"  ⚠ Erreur : {e}")
        return False


def download_nyu(output_dir: Path, raw_dir: Path, resume: bool = False) -> int:
    """
    Télécharge NYU Depth V2 (labeled) et extrait les images RGB.
    Le .mat contient 1,449 images indoor de taille 640×480.
    On extrait uniquement les images RGB (pas les depth maps).
    """
    mat_path = raw_dir / "nyu_depth_v2_labeled.mat"
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # 1. Télécharger le .mat (~2.8 GB)
    print("\n--- NYU Depth V2 : Téléchargement ---", flush=True)
    print(f"  URL  : {NYU_URL}", flush=True)
    print(f"  Dest : {mat_path}", flush=True)

    if not download_file(NYU_URL, mat_path, resume=resume):
        print("  ✗ Échec du téléchargement NYU", flush=True)
        return 0

    # 2. Extraire les images RGB
    print("\n--- NYU Depth V2 : Extraction des images RGB ---", flush=True)

    try:
        with h5py.File(mat_path, "r") as f:
            images = f["images"]
            # MATLAB stocke [480, 640, 3, 1449] (col-major)
            # h5py reverse les dims → [1449, 3, 640, 480]
            print(f"  Shape HDF5 : {images.shape}", flush=True)
            n_images = images.shape[0]  # 1449
            print(f"  Images trouvées : {n_images}", flush=True)

            count = 0
            skipped_test = 0
            for i in tqdm(range(n_images), desc="Extraction NYU (train only)"):
                # Exclure les images de test (benchmark Phase 5)
                if i >= NYU_TRAIN_END:
                    skipped_test += 1
                    continue

                out_path = images_dir / f"nyu_{i:05d}.png"
                if out_path.exists():
                    count += 1
                    continue

                # images[i] → [3, 640, 480] (C, W, H)
                img = np.array(images[i])    # [3, 640, 480]
                img = np.transpose(img, (2, 1, 0))  # → [480, 640, 3] (H, W, C)
                img = img.astype(np.uint8)
                Image.fromarray(img).save(out_path)
                count += 1

            print(f"  ✓ {count} images TRAIN extraites", flush=True)
            print(f"  ✗ {skipped_test} images TEST exclues (benchmark Phase 5)", flush=True)
            return count

    except Exception as e:
        print(f"  ⚠ Erreur extraction NYU : {e}", flush=True)
        # Essayer le format alternatif (scipy loadmat)
        return _extract_nyu_scipy(mat_path, images_dir)


def _extract_nyu_scipy(mat_path: Path, images_dir: Path) -> int:
    """Fallback : extraction via scipy (format MATLAB v5)."""
    try:
        import scipy.io as sio
        print("  Tentative avec scipy.io.loadmat...", flush=True)
        data = sio.loadmat(str(mat_path))
        images = data["images"]  # [H, W, 3, N]
        n = images.shape[3]
        count = 0
        for i in tqdm(range(n), desc="Extraction NYU (scipy)"):
            out_path = images_dir / f"nyu_{i:05d}.png"
            if out_path.exists():
                count += 1
                continue
            img = images[:, :, :, i].astype(np.uint8)
            Image.fromarray(img).save(out_path)
            count += 1
        print(f"  ✓ {count} images extraites", flush=True)
        return count
    except Exception as e:
        print(f"  ⚠ Échec scipy aussi : {e}", flush=True)
        return 0


# ============================================================
# SUN RGB-D — ~10,335 images indoor
# ============================================================

# SUN RGB-D est distribué via un lien direct
SUN_RGBD_URL = "https://rgbd.cs.princeton.edu/data/SUNRGBD.zip"


def download_sun_rgbd(output_dir: Path, raw_dir: Path, resume: bool = False) -> int:
    """
    Télécharge SUN RGB-D et extrait les images RGB.
    Contient ~10,335 images d'intérieurs variés.
    """
    zip_path = raw_dir / "SUNRGBD.zip"
    extract_dir = raw_dir / "SUNRGBD"
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # 1. Télécharger le ZIP (~7 GB)
    print("\n--- SUN RGB-D : Téléchargement ---", flush=True)
    print(f"  URL  : {SUN_RGBD_URL}", flush=True)
    print(f"  Dest : {zip_path} (~7 GB)", flush=True)

    if not download_file(SUN_RGBD_URL, zip_path, resume=resume):
        print("  ✗ Échec du téléchargement SUN RGB-D", flush=True)
        return 0

    # 2. Extraire
    if not extract_dir.exists():
        print("\n--- SUN RGB-D : Extraction ---", flush=True)
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(raw_dir)
        except Exception as e:
            print(f"  ⚠ Erreur extraction : {e}", flush=True)
            return 0

    # 3. Trouver et copier les images RGB
    print("\n--- SUN RGB-D : Copie des images RGB ---", flush=True)
    extensions = {".jpg", ".jpeg", ".png"}
    count = 0

    # SUN RGB-D structure : SUNRGBD/kv1/NYUdata/... ou SUNRGBD/kv2/...
    # Les images RGB sont dans des sous-dossiers "image/"
    for img_path in tqdm(
        list(extract_dir.rglob("image/*")),
        desc="Copie SUN RGB-D"
    ):
        if img_path.suffix.lower() in extensions and img_path.is_file():
            dest = images_dir / f"sun_{count:06d}{img_path.suffix}"
            if not dest.exists():
                try:
                    # Copier et vérifier
                    img = Image.open(img_path).convert("RGB")
                    img.save(dest)
                except Exception:
                    continue
            count += 1

    print(f"  ✓ {count} images copiées", flush=True)
    return count


# ============================================================
# Depth Anything 2k
# /!\ BENCHMARK ONLY
# ============================================================

DA_2K_URL = "https://huggingface.co/datasets/depth-anything/DA-2K/resolve/main/DA-2K.zip"

def download_da_2k(output_dir: Path, raw_dir: Path, resume: bool = False) -> int:
    """
    Télécharge Depth Anything 2k et extrait les images RGB.
    Contient ~2,000 images d'intérieurs variés.
    """
    zip_path = raw_dir / "DA-2K.zip"
    extract_dir = raw_dir / "DA-2K"
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # 1. Télécharger le ZIP (~1.5 GB)
    print("\n--- DA-2K : Téléchargement ---", flush=True)
    print(f"  URL  : {DA_2K_URL}", flush=True)
    print(f"  Dest : {zip_path} (~1.5 GB)", flush=True)

    if not download_file(DA_2K_URL, zip_path, resume=resume):
        print("  ✗ Échec du téléchargement DA-2K", flush=True)
        return 0

    # 2. Extraire
    if not extract_dir.exists():
        print("\n--- DA-2K : Extraction ---", flush=True)
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(raw_dir)
        except Exception as e:
            print(f"  ⚠ Erreur extraction : {e}", flush=True)
            return 0

    # 3. Trouver et copier les images RGB
    print("\n--- DA-2K : Copie des images RGB ---", flush=True)
    extensions = {".jpg", ".jpeg", ".png"}
    count = 0

    # Structure DA-2K : DA-2K/.../image/
    for img_path in tqdm(
        list(extract_dir.rglob("image/*")),
        desc="Copie DA-2K"
    ):
        if img_path.suffix.lower() in extensions and img_path.is_file():
            dest = images_dir / f"da_2k_{count:06d}{img_path.suffix}"
            if not dest.exists():
                try:
                    # Copier et vérifier
                    img = Image.open(img_path).convert("RGB")
                    img.save(dest)
                except Exception:
                    continue
            count += 1

    print(f"  ✓ {count} images copiées", flush=True)
    return count


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Télécharger des images réelles d'intérieurs"
    )
    parser.add_argument("--dataset", type=str, default="nyu",
                        choices=["nyu", "sun", "da_2k", "all"],
                        help="Dataset à télécharger")
    parser.add_argument("--output_dir", type=str,
                        default="datasets/real_unlabeled/indoor",
                        help="Répertoire de sortie")
    parser.add_argument("--raw_dir", type=str,
                        default="datasets/raw/indoor",
                        help="Répertoire pour fichiers bruts")
    parser.add_argument("--resume", action="store_true",
                        help="Reprendre un téléchargement interrompu")
    parser.add_argument("--delete_raw", action="store_true",
                        help="Supprimer les fichiers bruts après extraction")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    raw_dir = Path(args.raw_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60, flush=True)
    print("Téléchargement d'images réelles d'intérieurs", flush=True)
    print("=" * 60, flush=True)
    print(f"  Dataset    : {args.dataset}", flush=True)
    print(f"  Output dir : {output_dir}", flush=True)

    total = 0

    if args.dataset in ("nyu", "all"):
        n = download_nyu(output_dir, raw_dir, resume=args.resume)
        total += n

    if args.dataset in ("sun", "all"):
        n = download_sun_rgbd(output_dir, raw_dir, resume=args.resume)
        total += n

    if args.dataset in ("da_2k", "all"):
        n = download_da_2k(output_dir, raw_dir, resume=args.resume)
        total += n

    # Compter le résultat final
    images_dir = output_dir / "images"
    if images_dir.exists():
        n_final = len([
            f for f in images_dir.iterdir()
            if f.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ])
    else:
        n_final = 0

    print(f"\n{'=' * 60}", flush=True)
    print(f"✅ Téléchargement terminé.", flush=True)
    print(f"   {n_final} images dans {images_dir}/", flush=True)
    print(f"\n   → Prochaine étape : générer les pseudo-labels", flush=True)
    print(f"     python scripts/generate_pseudo_labels.py \\", flush=True)
    print(f"       --teacher_weights outputs/teacher/checkpoints/best_model.pt \\", flush=True)
    print(f"       --images_dir {images_dir}", flush=True)
    print(f"{'=' * 60}", flush=True)

    # Nettoyage optionnel
    if args.delete_raw:
        import shutil
        shutil.rmtree(raw_dir, ignore_errors=True)
        print(f"  Fichiers bruts supprimés : {raw_dir}", flush=True)


if __name__ == "__main__":
    main()
