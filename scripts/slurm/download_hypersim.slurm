#!/bin/bash
# ============================================================
# SLURM Job — Téléchargement Hypersim
# ============================================================
# Télécharge et préprocesse le dataset Hypersim depuis le CDN Apple.
# C'est un job CPU-only (pas besoin de GPU).
#
# ⚠ ESPACE DISQUE : le téléchargement brut fait ~200-300 GB
#   et le dataset final ~50-80 GB.
#   Vérifier : df -h /mnt/hdd
#
# Soumission :   sbatch scripts/slurm/download_hypersim.slurm
# Monitoring :   squeue -u $USER
# Logs :         tail -f outputs/slurm/download_hypersim_<JOB_ID>.out
#
# Pour un test rapide (5 scènes) :
#   Modifier --max_scenes 5 ci-dessous
# ============================================================

#SBATCH --job-name=dav2-download
#SBATCH --output=outputs/slurm/download_hypersim_%j.out
#SBATCH --error=outputs/slurm/download_hypersim_%j.out
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

export PYTHONUNBUFFERED=1

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(dirname $(dirname $(realpath $0)))}"
cd "$PROJECT_DIR"

echo "============================================"
echo "  SLURM Job ID     : $SLURM_JOB_ID"
echo "  Node              : $SLURM_NODELIST"
echo "  Date              : $(date)"
echo "============================================"

# Environnement
eval "$(conda shell.bash hook)"
conda activate deeplearning

# Vérifier l'espace disque
echo ""
echo "Espace disque :"
df -h . | head -2
echo ""

mkdir -p outputs/slurm datasets/raw datasets/synthetic

# ============================================================
# Choix : téléchargement complet ou test rapide
# Décommenter la ligne souhaitée :
# ============================================================

# --- Test rapide : 5 scènes (~2-5 GB) ---
# python scripts/download_hypersim.py \
#     --output_dir datasets/synthetic/hypersim \
#     --raw_dir datasets/raw/hypersim \
#     --max_scenes 5 \
#     --resume

# --- Téléchargement complet (~200-300 GB, plusieurs heures) ---
python scripts/download_hypersim.py \
    --output_dir datasets/synthetic/hypersim \
    --raw_dir datasets/raw/hypersim \
    --resume \
    --delete_raw

EXIT_CODE=$?

echo ""
echo "============================================"
echo "  Terminé à : $(date)"
echo "  Exit code  : $EXIT_CODE"
echo ""
echo "  Contenu du dataset :"
echo "    Images : $(ls datasets/synthetic/hypersim/images/ 2>/dev/null | wc -l)"
echo "    Depth  : $(ls datasets/synthetic/hypersim/depth/ 2>/dev/null | wc -l)"
echo "============================================"

exit $EXIT_CODE
