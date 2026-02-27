#!/bin/bash
# ============================================================
# SLURM Job — Téléchargement images indoor réelles
# ============================================================
# Télécharge NYU Depth V2 (~1,449 images) et/ou SUN RGB-D (~10K images)
# pour servir de données non étiquetées (pseudo-labels).
# Job CPU-only, pas besoin de GPU.
#
# Soumission :   sbatch scripts/slurm/download_indoor.slurm
# Logs :         tail -f outputs/slurm/download_indoor_<JOB_ID>.out
# ============================================================

#SBATCH --job-name=dav2-dl-indoor
#SBATCH --output=outputs/slurm/download_indoor_%j.out
#SBATCH --error=outputs/slurm/download_indoor_%j.out
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
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

# Vérifier espace disque
echo ""
echo "Espace disque :"
df -h . | head -2
echo ""

mkdir -p outputs/slurm datasets/raw datasets/real_unlabeled

# ============================================================
# Choix du dataset :
#   nyu  → 1,449 images indoor (~2.8 GB download)   [rapide]
#   sun  → ~10,335 images indoor (~7 GB download)    [moyen]
#   da_2k → ~2,000 images indoor (~1.5 GB download)   [recommandé]
#   all  → NYU + SUN + DA-2K (~11.5 GB download)      [complet]
# ============================================================

python scripts/download_indoor_images.py \
    --dataset da_2k \
    --output_dir datasets/real_unlabeled/indoor \
    --raw_dir datasets/raw/indoor \
    --resume

EXIT_CODE=$?

echo ""
echo "============================================"
echo "  Terminé à : $(date)"
echo "  Exit code  : $EXIT_CODE"
echo ""
echo "  Images indoor :"
ls datasets/real_unlabeled/indoor/images/ 2>/dev/null | wc -l
echo "============================================"

exit $EXIT_CODE
