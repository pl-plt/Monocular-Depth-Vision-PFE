"""
download_vkitti2.py — Téléchargement et préprocessing de Virtual KITTI 2.

Virtual KITTI 2 (Naver Labs, 2020) : 5 scènes outdoor, 10 conditions météo,
~21 336 images, 1242×375. Complément outdoor à Hypersim (indoor).

Ce script :
1. Télécharge rgb.tar (~21 GB) et depth.tar (~6 GB) depuis le CDN Naver
2. Extrait et réorganise les paires image/depth
3. Sauvegarde dans la structure attendue par SyntheticDepthDataset :
   datasets/synthetic/vkitti2/
   ├── images/   (*.jpg)
   └── depth/    (*.npy float32, mètres)

Sources :
    https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/

Format depth :
    PNG uint16, valeurs en centimètres.
    Pixel 65535 → invalide (ciel ou hors-champ).
    Conversion : depth_m = px / 100.0 (clip à max_depth)

Usage :
    python scripts/download_vkitti2.py \
        --output_dir datasets/synthetic/vkitti2 \
        --raw_dir datasets/raw/vkitti2 \
        --resume

    # Télécharger uniquement, préprocesser ensuite
    python scripts/download_vkitti2.py --skip_preprocess --resume
    python scripts/download_vkitti2.py --skip_download

Dépendances :
    pip install requests numpy Pillow tqdm
"""

import argparse
import sys
import tarfile
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from tqdm import tqdm


# ============================================================
# URLs et constantes
# ============================================================

CDN_BASE = "https://download.europe.naverlabs.com/virtual_kitti_2.0.3"

FILES = {
    "rgb":   (f"{CDN_BASE}/vkitti_2.0.3_rgb.tar",   "vkitti_2.0.3_rgb.tar"),    # 7.01 GB
    "depth": (f"{CDN_BASE}/vkitti_2.0.3_depth.tar", "vkitti_2.0.3_depth.tar"),  # 7.58 GB
}

# 5 scènes × 10 conditions
SCENES    = ["Scene01", "Scene02", "Scene06", "Scene18", "Scene20"]
CLONES    = ["clone", "fog", "morning", "overcast", "rain", "sunset",
             "15-deg-left", "15-deg-right", "30-deg-left", "30-deg-right"]
CAMERAS   = ["Camera_0"]   # Camera_1 = passenger side ; Camera_0 suffit

# VKitti2 has no explicit invalid marker — all pixels are valid up to the far plane
# (655.35 m max: pixel intensity 1 = 1 cm, 65535 = 655.35 m).
# We simply clip at MAX_DEPTH_M without treating any pixel as invalid.
DEPTH_INVALID = None
MAX_DEPTH_M   = 80.0    # clamp at 80 m (consistent with KITTI outdoor convention)


# ============================================================
# Téléchargement
# ============================================================

def download_file(url: str, dest: Path, resume: bool = False) -> bool:
    """Télécharge un fichier avec barre de progression et support de reprise."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    headers = {}
    mode = "wb"
    initial_size = 0

    if dest.exists() and not resume:
        print(f"  Déjà présent : {dest.name} ({dest.stat().st_size / 1e9:.1f} GB)", flush=True)
        return True

    if resume and dest.exists():
        initial_size = dest.stat().st_size
        headers["Range"] = f"bytes={initial_size}-"
        mode = "ab"
        print(f"  Reprise depuis {initial_size / 1e9:.1f} GB : {dest.name}", flush=True)

    try:
        resp = requests.get(url, headers=headers, stream=True, timeout=120)
        if resp.status_code == 416:
            print(f"  Déjà complet : {dest.name}", flush=True)
            return True
        if resp.status_code not in (200, 206):
            print(f"  ✗ HTTP {resp.status_code} pour {url}", flush=True)
            return False

        total = int(resp.headers.get("content-length", 0)) + initial_size
        with open(dest, mode) as f:
            with tqdm(total=total, initial=initial_size, unit="B",
                      unit_scale=True, desc=dest.name) as pbar:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
                    pbar.update(len(chunk))
        return True
    except Exception as e:
        print(f"  ✗ Erreur : {e}", flush=True)
        return False


def extract_tar(tar_path: Path, extract_dir: Path, desc: str = "") -> bool:
    """Extrait un .tar avec barre de progression."""
    if not tar_path.exists():
        print(f"  ✗ Fichier introuvable : {tar_path}", flush=True)
        return False

    print(f"  Extraction : {tar_path.name} → {extract_dir}", flush=True)
    extract_dir.mkdir(parents=True, exist_ok=True)

    try:
        with tarfile.open(tar_path, "r") as tf:
            members = tf.getmembers()
            for m in tqdm(members, desc=desc or tar_path.name, unit="fichier"):
                tf.extract(m, extract_dir, filter="data")
        return True
    except Exception as e:
        print(f"  ✗ Erreur extraction : {e}", flush=True)
        return False


# ============================================================
# Préprocessing
# ============================================================

def process_vkitti2(
    rgb_root: Path,
    depth_root: Path,
    output_images: Path,
    output_depth: Path,
    cameras: list = None,
) -> int:
    """
    Convertit les paires RGB/depth de Virtual KITTI 2 au format SyntheticDepthDataset.

    Structure source :
        rgb_root/SceneXX/clone/frames/rgb/Camera_0/rgb_XXXXX.jpg
        depth_root/SceneXX/clone/frames/depth/Camera_0/depth_XXXXX.png

    Structure cible :
        output_images/vk_{scene}_{clone}_{cam}_{frame}.jpg
        output_depth/vk_{scene}_{clone}_{cam}_{frame}.npy

    Returns:
        Nombre total de paires traitées.
    """
    output_images.mkdir(parents=True, exist_ok=True)
    output_depth.mkdir(parents=True, exist_ok=True)

    if cameras is None:
        cameras = CAMERAS

    total = 0
    already = 0

    for scene in tqdm(SCENES, desc="Scènes VKitti2"):
        for clone in CLONES:
            for cam in cameras:
                rgb_dir   = rgb_root   / scene / clone / "frames" / "rgb"   / cam
                depth_dir = depth_root / scene / clone / "frames" / "depth" / cam

                if not rgb_dir.exists():
                    continue

                rgb_files = sorted(rgb_dir.glob("rgb_*.jpg"))
                for rgb_file in rgb_files:
                    frame_id = rgb_file.stem.replace("rgb_", "")
                    depth_file = depth_dir / f"depth_{frame_id}.png"

                    if not depth_file.exists():
                        continue

                    # Nommage unique
                    out_stem  = f"vk_{scene}_{clone}_{cam}_{frame_id}"
                    out_img   = output_images / f"{out_stem}.jpg"
                    out_dep   = output_depth  / f"{out_stem}.npy"

                    if out_img.exists() and out_dep.exists():
                        already += 1
                        continue

                    try:
                        # --- Image RGB : copier directement (JPEG déjà bon) ---
                        img = Image.open(rgb_file).convert("RGB")
                        img.save(out_img, quality=95)

                        # --- Depth map : uint16 cm → float32 mètres ---
                        depth_px = np.array(Image.open(depth_file), dtype=np.uint16)
                        depth_m = depth_px.astype(np.float32) / 100.0
                        depth_m = np.clip(depth_m, 0.0, MAX_DEPTH_M)
                        np.save(out_dep, depth_m)

                        total += 1
                    except Exception as e:
                        print(f"  ⚠ Erreur {out_stem}: {e}", flush=True)

    print(f"  ✓ {total} paires converties  ({already} déjà présentes)", flush=True)
    return total + already


# ============================================================
# Vérification de la structure extraite
# ============================================================

def find_roots(extract_dir: Path, kind: str) -> Path:
    """
    Trouve le répertoire racine RGB ou depth dans l'archive extraite.
    Retourne None si introuvable.
    """
    # L'archive extrait généralement vers extract_dir/vkitti_2.0.3_{kind}/
    candidates = [
        extract_dir / f"vkitti_2.0.3_{kind}",
        extract_dir / kind,
        extract_dir,
    ]
    for c in candidates:
        if c.exists() and any(c.glob("Scene*")):
            return c
    # Recherche profonde
    matches = list(extract_dir.glob(f"**/Scene01"))
    if matches:
        return matches[0].parent
    return None


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Téléchargement et préprocessing Virtual KITTI 2"
    )
    parser.add_argument("--output_dir", type=str,
                        default="datasets/synthetic/vkitti2",
                        help="Répertoire de sortie (images/ + depth/)")
    parser.add_argument("--raw_dir", type=str,
                        default="datasets/raw/vkitti2",
                        help="Répertoire pour les archives .tar")
    parser.add_argument("--resume", action="store_true",
                        help="Reprendre un téléchargement interrompu")
    parser.add_argument("--skip_download", action="store_true",
                        help="Passer le téléchargement (préprocesser uniquement)")
    parser.add_argument("--skip_preprocess", action="store_true",
                        help="Passer le préprocessing (télécharger uniquement)")
    parser.add_argument("--delete_raw", action="store_true",
                        help="Supprimer les archives .tar après extraction")
    parser.add_argument("--both_cameras", action="store_true",
                        help="Inclure Camera_0 et Camera_1 (double le nombre d'images)")
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    raw_dir    = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    cameras = ["Camera_0", "Camera_1"] if args.both_cameras else CAMERAS

    print("=" * 60, flush=True)
    print("  Virtual KITTI 2 — Téléchargement & Préprocessing", flush=True)
    print("=" * 60, flush=True)
    print(f"  Raw dir    : {raw_dir}", flush=True)
    print(f"  Output dir : {output_dir}", flush=True)
    print(f"  Caméras    : {cameras}", flush=True)
    print(f"  Scènes     : {SCENES}", flush=True)
    print(f"  Conditions : {len(CLONES)}", flush=True)
    print(flush=True)

    # ---- 1. Téléchargement ----
    if not args.skip_download:
        print(f"\n--- Phase 1 : Téléchargement (~15 GB total) ---", flush=True)
        for kind, (url, filename) in FILES.items():
            dest = raw_dir / filename
            print(f"\n  {kind.upper()} : {url}", flush=True)
            ok = download_file(url, dest, resume=args.resume)
            if not ok:
                print(f"  ✗ Téléchargement {kind} échoué. Abandon.", flush=True)
                sys.exit(1)

    # ---- 2. Extraction ----
    if not args.skip_preprocess:
        print("\n--- Phase 2 : Extraction des archives ---", flush=True)

        # Both tars extract into the same merged directory:
        #   extract_dir/Scene01/clone/frames/rgb/Camera_0/rgb_*.jpg   (from rgb tar)
        #   extract_dir/Scene01/clone/frames/depth/Camera_0/depth_*.png (from depth tar)
        extract_dir = raw_dir / "extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)

        # Sentinel paths — presence of these means that particular tar is extracted
        rgb_sentinel   = extract_dir / "Scene01" / "clone" / "frames" / "rgb"
        depth_sentinel = extract_dir / "Scene01" / "clone" / "frames" / "depth"

        for kind, (_, filename) in FILES.items():
            tar_path = raw_dir / filename
            sentinel = rgb_sentinel if kind == "rgb" else depth_sentinel

            if sentinel.exists():
                print(f"  Déjà extrait : {kind} ({sentinel})", flush=True)
                continue

            if not tar_path.exists():
                print(f"  ⚠ Archive manquante : {tar_path}", flush=True)
                print(f"     Lancer d'abord sans --skip_download", flush=True)
                sys.exit(1)

            ok = extract_tar(tar_path, extract_dir, desc=f"Extraction {kind}")
            if not ok:
                sys.exit(1)

        # Verify both sentinels are present
        if not rgb_sentinel.exists() or not depth_sentinel.exists():
            print(f"  ✗ Extraction incomplète :", flush=True)
            print(f"     RGB   : {'OK' if rgb_sentinel.exists() else 'MANQUANT'}", flush=True)
            print(f"     Depth : {'OK' if depth_sentinel.exists() else 'MANQUANT'}", flush=True)
            sys.exit(1)

        print(f"  ✓ Les deux archives extraites dans : {extract_dir}", flush=True)

        # ---- 3. Préprocessing ----
        print("\n--- Phase 3 : Conversion RGB/Depth → images/ + depth/ ---", flush=True)

        # Both rgb and depth live under the same merged extract_dir
        print(f"  Root commun : {extract_dir}", flush=True)

        n = process_vkitti2(
            rgb_root   = extract_dir,
            depth_root = extract_dir,
            output_images = output_dir / "images",
            output_depth  = output_dir / "depth",
            cameras = cameras,
        )

        # ---- 4. Nettoyage ----
        if args.delete_raw:
            import shutil
            print("\n--- Nettoyage ---", flush=True)
            shutil.rmtree(extract_dir, ignore_errors=True)
            print(f"  Supprimé : {extract_dir.name}/", flush=True)
            for _, (_, filename) in FILES.items():
                tar = raw_dir / filename
                if tar.exists():
                    tar.unlink()
                    print(f"  Supprimé : {tar.name}", flush=True)

    # ---- Résumé ----
    n_img = len(list((output_dir / "images").glob("*.*")))
    n_dep = len(list((output_dir / "depth").glob("*.npy")))

    print("\n" + "=" * 60, flush=True)
    print("  Résumé Virtual KITTI 2", flush=True)
    print("=" * 60, flush=True)
    print(f"  Images  : {n_img}", flush=True)
    print(f"  Depth   : {n_dep}", flush=True)
    print(f"  Output  : {output_dir}", flush=True)
    print("  ✓ Prêt pour l'entraînement Teacher.", flush=True)


if __name__ == "__main__":
    main()
