"""
download_nyu_test.py — Télécharge et extrait le test set NYU-Depth V2.

Utilise le même fichier .mat que download_indoor_images.py
(nyu_depth_v2_labeled.mat) — réutilise le téléchargement déjà fait si
disponible dans --raw_dir.

Eigen split :
    Indices 0–794   → train (795 images, utilisées pour pseudo-labels)
    Indices 795–1448 → test  (654 images, benchmark Phase 5)

Structure de sortie :
    output_dir/
    ├── images/
    │   ├── nyu_00795.png
    │   └── ...
    └── depth/
        ├── nyu_00795.npy  (float32, mètres)
        └── ...

Usage :
    python scripts/download_nyu_test.py \
        --output_dir datasets/real_depth/nyudepthv2 \
        --raw_dir datasets/raw/nyudepthv2 \
        --resume

Dépendances :
    pip install requests h5py numpy Pillow tqdm
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import requests
import h5py
from PIL import Image
from tqdm import tqdm


# ============================================================
# Constantes
# ============================================================

NYU_URL = "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"
NYU_TRAIN_END = 795   # Eigen split : 0–794 train, 795–1448 test
NYU_TOTAL = 1449


# ============================================================
# Utilitaires
# ============================================================

def download_file(url: str, dest: Path, resume: bool = False) -> bool:
    """Télécharge un fichier avec barre de progression et support de reprise."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    headers = {}
    mode = "wb"
    initial_size = 0

    if resume and dest.exists():
        initial_size = dest.stat().st_size
        headers["Range"] = f"bytes={initial_size}-"
        mode = "ab"
    elif dest.exists():
        print(f"  Déjà présent : {dest}", flush=True)
        return True

    print(f"  Téléchargement : {url}", flush=True)
    try:
        resp = requests.get(url, headers=headers, stream=True, timeout=120)
        if resp.status_code == 416:
            print(f"  Déjà complet : {dest}", flush=True)
            return True
        if resp.status_code not in (200, 206):
            print(f"  ✗ HTTP {resp.status_code}", flush=True)
            return False

        total = int(resp.headers.get("content-length", 0)) + initial_size
        with open(dest, mode) as f:
            with tqdm(total=total, initial=initial_size, unit="B",
                      unit_scale=True, desc=dest.name, leave=False) as pbar:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
                    pbar.update(len(chunk))
        return True
    except Exception as e:
        print(f"  ✗ Erreur téléchargement : {e}", flush=True)
        return False


# ============================================================
# Extraction du test set
# ============================================================

def extract_nyu_test(mat_path: Path, images_dir: Path, depth_dir: Path) -> int:
    """
    Extrait les images RGB et depth maps du test set (indices 795–1448).

    Format HDF5 (via h5py) :
        images : [1449, 3, 640, 480]  → reorder → [H=480, W=640, C=3]
        depths : [1449, 640, 480]     → reorder → [H=480, W=640]  (mètres)

    Returns:
        Nombre d'images extraites.
    """
    images_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Extraction test set NYU-Depth V2 ---", flush=True)

    try:
        with h5py.File(mat_path, "r") as f:
            # Vérifier la présence des clés HDF5
            print(f"  Clés HDF5 : {list(f.keys())}", flush=True)

            images_ds = f["images"]   # [1449, 3, 640, 480]
            depths_ds = f["depths"]   # [1449, 640, 480]  (mètres, float32/float64)

            n_total = images_ds.shape[0]
            print(f"  Images totales : {n_total}", flush=True)
            print(f"  Shape images  : {images_ds.shape}", flush=True)
            print(f"  Shape depths  : {depths_ds.shape}", flush=True)

            count = 0
            already = 0

            test_indices = range(NYU_TRAIN_END, n_total)  # 795–1448 = 654 images
            print(f"  Test indices : {NYU_TRAIN_END}–{n_total - 1} ({len(test_indices)} images)", flush=True)

            for i in tqdm(test_indices, desc="Extraction NYU test"):
                img_out  = images_dir / f"nyu_{i:05d}.png"
                dep_out  = depth_dir  / f"nyu_{i:05d}.npy"

                if img_out.exists() and dep_out.exists():
                    already += 1
                    continue

                # --- Image RGB ---
                # images[i] → [3, 640, 480]  (C, W, H)
                img = np.array(images_ds[i])         # [3, 640, 480]
                img = np.transpose(img, (2, 1, 0))   # → [480, 640, 3]  (H, W, C)
                img = img.astype(np.uint8)
                Image.fromarray(img).save(img_out)

                # --- Depth map ---
                # depths[i] → [640, 480]  (W, H)
                dep = np.array(depths_ds[i])         # [640, 480]
                dep = dep.T                           # → [480, 640]  (H, W)
                dep = dep.astype(np.float32)
                np.save(dep_out, dep)

                count += 1

            print(f"  ✓ {count} images extraites  ({already} déjà présentes)", flush=True)
            return count + already

    except OSError as e:
        print(f"  ✗ Impossible d'ouvrir le .mat avec h5py : {e}", flush=True)
        print("     Tentative avec scipy.io …", flush=True)
        return _extract_nyu_test_scipy(mat_path, images_dir, depth_dir)


def _extract_nyu_test_scipy(mat_path: Path, images_dir: Path, depth_dir: Path) -> int:
    """Fallback scipy pour les .mat MATLAB v5 (non-HDF5)."""
    try:
        import scipy.io as sio
        print("  Chargement via scipy.io.loadmat …", flush=True)
        data = sio.loadmat(str(mat_path))
        images = data["images"]   # [H, W, 3, N]  (MATLAB col-major)
        depths = data["depths"]   # [H, W, N]

        n_total = images.shape[3]
        count = 0
        for i in tqdm(range(NYU_TRAIN_END, n_total), desc="Extraction NYU test (scipy)"):
            img_out = images_dir / f"nyu_{i:05d}.png"
            dep_out = depth_dir  / f"nyu_{i:05d}.npy"

            if img_out.exists() and dep_out.exists():
                continue

            img = images[:, :, :, i].astype(np.uint8)
            Image.fromarray(img).save(img_out)

            dep = depths[:, :, i].astype(np.float32)
            np.save(dep_out, dep)
            count += 1

        print(f"  ✓ {count} images extraites (scipy)", flush=True)
        return count
    except Exception as e:
        print(f"  ✗ Échec scipy : {e}", flush=True)
        return 0


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Télécharge et extrait le test set NYU-Depth V2 (654 images)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets/real_depth/nyudepthv2",
        help="Répertoire de sortie : images/ et depth/ seront créés ici.",
    )
    parser.add_argument(
        "--raw_dir",
        type=str,
        default="datasets/raw/nyudepthv2",
        help="Répertoire pour le fichier .mat brut (réutilise un téléchargement existant).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Reprendre un téléchargement interrompu (défaut : True).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    raw_dir    = Path(args.raw_dir)
    images_dir = output_dir / "images"
    depth_dir  = output_dir / "depth"

    print("=" * 60, flush=True)
    print("  NYU-Depth V2 — Test Set (654 images)", flush=True)
    print("=" * 60, flush=True)
    print(f"  Output   : {output_dir}", flush=True)
    print(f"  Raw dir  : {raw_dir}", flush=True)

    # Chercher d'abord le .mat déjà téléchargé pour les données d'entraînement
    existing_mat = raw_dir / "nyu_depth_v2_labeled.mat"
    train_raw_dir = Path("datasets/raw/indoor")
    train_mat = train_raw_dir / "nyu_depth_v2_labeled.mat"

    mat_path = existing_mat
    if not existing_mat.exists() and train_mat.exists():
        print(f"\n  .mat trouvé dans le répertoire train : {train_mat}", flush=True)
        print("  Utilisation de ce fichier (pas de re-téléchargement nécessaire).", flush=True)
        mat_path = train_mat
    elif not existing_mat.exists():
        # Télécharger le .mat
        print(f"\n--- Téléchargement .mat (~2.8 GB) ---", flush=True)
        raw_dir.mkdir(parents=True, exist_ok=True)
        ok = download_file(NYU_URL, existing_mat, resume=args.resume)
        if not ok:
            print("  ✗ Téléchargement échoué. Abandon.", flush=True)
            sys.exit(1)
        mat_path = existing_mat
    else:
        print(f"\n  .mat déjà présent : {existing_mat}", flush=True)

    # Extraire le test set
    n = extract_nyu_test(mat_path, images_dir, depth_dir)

    print("\n" + "=" * 60, flush=True)
    print("  Résumé", flush=True)
    print("=" * 60, flush=True)
    print(f"  Images PNG  : {len(list(images_dir.glob('*.png')))}", flush=True)
    print(f"  Depth maps  : {len(list(depth_dir.glob('*.npy')))}", flush=True)
    print(f"  Output dir  : {output_dir}", flush=True)
    print("  ✓ Terminé.", flush=True)


if __name__ == "__main__":
    main()
