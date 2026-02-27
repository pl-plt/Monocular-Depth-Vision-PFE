"""
download_sa1b.py — Téléchargement d'un sous-ensemble de SA-1B pour pseudo-labels.

SA-1B (Segment Anything 1 Billion) est le dataset utilisé dans le papier
Depth Anything V2 comme source d'images réelles non étiquetées.
Chaque tar contient ~11K images .jpg (~10.5 GB).

On télécharge N tars (par défaut 4 → ~42 GB, ~44K images) et on extrait
uniquement les images .jpg (pas les annotations JSON).

Usage :
    # Télécharger 4 tars (~42 GB, ~44K images)
    python scripts/download_sa1b.py \
        --links_file download-sa-1b.txt \
        --output_dir datasets/real_unlabeled/sa1b/images \
        --n_tars 4

    # Télécharger un seul tar pour tester
    python scripts/download_sa1b.py \
        --links_file download-sa-1b.txt \
        --output_dir datasets/real_unlabeled/sa1b/images \
        --n_tars 1
"""

import os
import sys
import argparse
import tarfile
import tempfile
import shutil
from pathlib import Path

import requests
from tqdm import tqdm

os.environ.setdefault("PYTHONUNBUFFERED", "1")


def parse_links_file(links_file: str) -> list[tuple[str, str]]:
    """Parse le fichier TSV de liens SA-1B.
    
    Returns:
        Liste de (filename, url)
    """
    entries = []
    with open(links_file, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or i == 0:  # Skip header
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                entries.append((parts[0], parts[1]))
    return entries


def download_file(url: str, dest: Path, desc: str = "") -> bool:
    """Télécharge un fichier avec barre de progression et support resume."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    headers = {}
    mode = "wb"
    initial_size = 0

    # Support resume
    if dest.exists():
        initial_size = dest.stat().st_size
        headers["Range"] = f"bytes={initial_size}-"
        mode = "ab"

    try:
        response = requests.get(url, headers=headers, stream=True, timeout=60)

        if response.status_code == 416:
            # Déjà complet
            print(f"  ✓ {desc} déjà téléchargé", flush=True)
            return True

        if response.status_code not in (200, 206):
            print(f"  ✗ HTTP {response.status_code} pour {desc}", flush=True)
            return False

        total = int(response.headers.get("content-length", 0))
        if initial_size > 0 and response.status_code == 206:
            print(f"  ↻ Reprise à {initial_size / 1e9:.1f} GB", flush=True)

        with open(dest, mode) as f, tqdm(
            total=total + initial_size,
            initial=initial_size,
            unit="B",
            unit_scale=True,
            desc=desc,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):  # 8 MB
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

        return True

    except Exception as e:
        print(f"  ✗ Erreur téléchargement {desc}: {e}", flush=True)
        return False


def extract_images_from_tar(tar_path: Path, output_dir: Path) -> int:
    """Extrait uniquement les images .jpg d'un tar SA-1B.
    
    SA-1B tars contiennent :
    - sa_XXXXXX/sa_NNNNNN.jpg (images RGB)
    - sa_XXXXXX/sa_NNNNNN.json (annotations — ignorées)
    
    Returns:
        Nombre d'images extraites.
    """
    count = 0
    skipped = 0

    try:
        with tarfile.open(tar_path, "r") as tf:
            members = tf.getmembers()
            jpg_members = [m for m in members if m.name.endswith(".jpg") and m.isfile()]

            print(f"  Tar contient {len(members)} fichiers, {len(jpg_members)} images .jpg", flush=True)

            for member in tqdm(jpg_members, desc=f"  Extraction {tar_path.stem}"):
                # Extraire le nom de base (sans le dossier sa_XXXXXX/)
                basename = Path(member.name).name
                out_path = output_dir / basename

                if out_path.exists():
                    skipped += 1
                    continue

                # Extraire l'image
                with tf.extractfile(member) as src:
                    with open(out_path, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                count += 1

    except Exception as e:
        print(f"  ✗ Erreur extraction {tar_path.name}: {e}", flush=True)

    if skipped > 0:
        print(f"  ({skipped} images déjà existantes, skippées)", flush=True)

    return count


def main():
    parser = argparse.ArgumentParser(
        description="Téléchargement SA-1B subset pour pseudo-labels"
    )
    parser.add_argument("--links_file", type=str, required=True,
                        help="Fichier TSV avec les liens de téléchargement")
    parser.add_argument("--output_dir", type=str,
                        default="datasets/real_unlabeled/sa1b/images",
                        help="Répertoire de sortie des images")
    parser.add_argument("--raw_dir", type=str,
                        default="datasets/raw/sa1b",
                        help="Répertoire temporaire pour les tars")
    parser.add_argument("--n_tars", type=int, default=4,
                        help="Nombre de tars à télécharger (chaque tar ≈ 10.5 GB, ~11K images)")
    parser.add_argument("--delete_tar", action="store_true",
                        help="Supprimer chaque tar après extraction")
    parser.add_argument("--start_index", type=int, default=0,
                        help="Index de départ dans la liste des tars (pour reprendre)")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    raw_dir = Path(args.raw_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Parser les liens
    entries = parse_links_file(args.links_file)
    print("=" * 60, flush=True)
    print("Téléchargement SA-1B (subset)", flush=True)
    print("=" * 60, flush=True)
    print(f"  Liens disponibles : {len(entries)} tars", flush=True)
    print(f"  Tars à télécharger : {args.n_tars}", flush=True)
    print(f"  Taille estimée     : ~{args.n_tars * 10.5:.0f} GB", flush=True)
    print(f"  Images estimées    : ~{args.n_tars * 11_000:,}", flush=True)
    print(f"  Output dir         : {output_dir}", flush=True)
    print(f"  Raw dir            : {raw_dir}", flush=True)
    print(flush=True)

    # Sélectionner les tars (espacement régulier pour diversité)
    n_available = len(entries)
    n_tars = min(args.n_tars, n_available)

    # Espacement régulier dans la liste pour maximiser la diversité
    indices = [int(i * n_available / n_tars) for i in range(n_tars)]
    selected = [(entries[i], i) for i in indices]

    print(f"  Tars sélectionnés (espacement régulier pour diversité) :", flush=True)
    for (filename, _url), idx in selected:
        print(f"    [{idx:4d}] {filename}", flush=True)
    print(flush=True)

    # Vérifier l'espace disque
    total_images = 0
    total_tars_ok = 0
    total_tars_fail = 0

    for tar_idx, ((filename, url), list_idx) in enumerate(selected):
        print(f"\n{'─' * 60}", flush=True)
        print(f"  Tar {tar_idx + 1}/{n_tars} : {filename} (index {list_idx})", flush=True)
        print(f"{'─' * 60}", flush=True)

        tar_path = raw_dir / filename

        # 1. Télécharger
        if tar_path.exists() and tar_path.stat().st_size > 1e9:  # >1 GB = probablement complet
            print(f"  ↻ Tar déjà téléchargé ({tar_path.stat().st_size / 1e9:.1f} GB)", flush=True)
        else:
            success = download_file(url, tar_path, desc=filename)
            if not success:
                print(f"  ✗ Échec téléchargement {filename}, passage au suivant", flush=True)
                total_tars_fail += 1
                continue

        # 2. Extraire les images
        n_extracted = extract_images_from_tar(tar_path, output_dir)
        total_images += n_extracted
        total_tars_ok += 1
        print(f"  ✓ {n_extracted} images extraites de {filename}", flush=True)

        # 3. Supprimer le tar si demandé
        if args.delete_tar and tar_path.exists():
            tar_path.unlink()
            print(f"  🗑 Tar supprimé ({filename})", flush=True)

    # Résumé
    final_count = len(list(output_dir.glob("*.jpg")))
    print(f"\n{'=' * 60}", flush=True)
    print(f"✅ Téléchargement SA-1B terminé", flush=True)
    print(f"  Tars réussis  : {total_tars_ok}/{n_tars}", flush=True)
    print(f"  Tars échoués  : {total_tars_fail}", flush=True)
    print(f"  Images total  : {final_count}", flush=True)
    print(f"  Output dir    : {output_dir}", flush=True)
    print(f"\n  → Prochaine étape : générer les pseudo-labels", flush=True)
    print(f"    python scripts/generate_pseudo_labels.py \\", flush=True)
    print(f"      --teacher_weights outputs/teacher/checkpoints/best_model.pt \\", flush=True)
    print(f"      --images_dir {output_dir}", flush=True)
    print(f"{'=' * 60}", flush=True)


if __name__ == "__main__":
    main()
