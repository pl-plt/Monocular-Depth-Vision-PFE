"""
download_data.py — Phase 2 : Telechargement des datasets.

Dispatcher central qui delegue aux scripts de telechargement dedies.

Datasets synthetiques (entrainement Teacher) :
- Hypersim : ~50k images indoor
- Virtual KITTI 2 : ~20k images outdoor

Datasets reels non etiquetes (pseudo-labels -> entrainement Student) :
- SA-1B subset : 50k -> 200k -> 500k images (progressif)

Datasets reels etiquetes (images indoor) :
- NYU train + SUN RGB-D + DA-2K

Benchmarks (evaluation) :
- NYU-Depth V2 test : 654 images

Usage :
    python scripts/download_data.py --dataset all
    python scripts/download_data.py --dataset hypersim --max_scenes 10
    python scripts/download_data.py --dataset sa1b --n_tars 4
    python scripts/download_data.py --dataset nyu_test
    python scripts/download_data.py --dataset indoor --indoor_dataset all
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent


def run_script(script_name: str, args: list):
    """Lance un script de telechargement dedie en sous-processus."""
    script_path = SCRIPTS_DIR / script_name
    if not script_path.exists():
        print(f"ERREUR : script introuvable : {script_path}")
        sys.exit(1)

    cmd = [sys.executable, str(script_path)] + args
    print(f"\n{'='*60}")
    print(f"Lancement : {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"ERREUR : {script_name} a echoue (code {result.returncode})")
        return False
    return True


def download_hypersim(args):
    """Delegue a download_hypersim.py."""
    cmd_args = [
        "--output_dir", args.output_dir + "/synthetic/hypersim",
        "--resume",
    ]
    if args.max_scenes is not None:
        cmd_args += ["--max_scenes", str(args.max_scenes)]
    if args.workers is not None:
        cmd_args += ["--workers", str(args.workers)]
    return run_script("download_hypersim.py", cmd_args)


def download_vkitti2(args):
    """Delegue a download_vkitti2.py."""
    cmd_args = [
        "--output_dir", args.output_dir + "/synthetic/vkitti2",
        "--resume",
    ]
    return run_script("download_vkitti2.py", cmd_args)


def download_sa1b(args):
    """Delegue a download_sa1b.py."""
    links_file = Path(args.output_dir).parent / "download-sa-1b.txt"
    if not links_file.exists():
        links_file = SCRIPTS_DIR.parent / "download-sa-1b.txt"
    if not links_file.exists():
        print(f"ERREUR : fichier de liens SA-1B introuvable.")
        print(f"  Attendu : {links_file}")
        print(f"  Telecharger depuis https://ai.meta.com/datasets/segment-anything/")
        return False

    cmd_args = [
        "--links_file", str(links_file),
        "--output_dir", args.output_dir + "/real_unlabeled/sa1b/images",
        "--n_tars", str(args.n_tars),
    ]
    return run_script("download_sa1b.py", cmd_args)


def download_nyu_test(args):
    """Delegue a download_nyu_test.py."""
    cmd_args = [
        "--output_dir", args.output_dir + "/real_depth/nyudepthv2",
        "--resume",
    ]
    return run_script("download_nyu_test.py", cmd_args)


def download_indoor(args):
    """Delegue a download_indoor_images.py."""
    cmd_args = [
        "--dataset", args.indoor_dataset,
        "--output_dir", args.output_dir + "/real_unlabeled/indoor",
        "--resume",
    ]
    return run_script("download_indoor_images.py", cmd_args)


DATASETS = {
    "hypersim": download_hypersim,
    "vkitti2": download_vkitti2,
    "sa1b": download_sa1b,
    "nyu_test": download_nyu_test,
    "indoor": download_indoor,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dispatcher de telechargement des datasets Depth Anything V2."
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()) + ["all"],
        required=True,
        help="Dataset a telecharger (ou 'all' pour tout).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets",
        help="Repertoire racine de sortie (defaut : datasets/).",
    )
    parser.add_argument(
        "--max_scenes",
        type=int,
        default=None,
        help="Nombre max de scenes Hypersim (optionnel).",
    )
    parser.add_argument(
        "--n_tars",
        type=int,
        default=4,
        help="Nombre de tars SA-1B a telecharger (defaut : 4, ~44K images).",
    )
    parser.add_argument(
        "--indoor_dataset",
        type=str,
        choices=["nyu", "sun", "da_2k", "all"],
        default="all",
        help="Sous-dataset indoor a telecharger (defaut : all).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Nombre de telechargements paralleles pour Hypersim.",
    )

    args = parser.parse_args()

    if args.dataset == "all":
        results = {}
        for name, func in DATASETS.items():
            results[name] = func(args)

        print(f"\n{'='*60}")
        print("Resume des telechargements :")
        for name, success in results.items():
            status = "OK" if success else "ECHEC"
            print(f"  {name:15s} : {status}")
        print(f"{'='*60}")
    else:
        success = DATASETS[args.dataset](args)
        sys.exit(0 if success else 1)
