#!/bin/bash
# ============================================================
# SLURM Job — Téléchargement SA-1B (subset)
# ============================================================
# Télécharge N tars de SA-1B (~10.5 GB chacun, ~11K images).
# Par défaut: 4 tars = ~42 GB, ~44K images.
# Job CPU-only (pas de GPU), mais besoin d'espace disque.
#
# ⚠ ESPACE DISQUE : 4 tars = ~42 GB téléchargés + ~42 GB images
#   (--delete_tar supprime les tars après extraction)
#
# Soumission :   sbatch scripts/slurm/download_sa1b.slurm
# Monitoring :   squeue -u $USER
# Logs :         tail -f outputs/slurm/download_sa1b_<JOB_ID>.out
# ============================================================

#SBATCH --job-name=dav2-sa1b
#SBATCH --output=outputs/slurm/download_sa1b_%j.out
#SBATCH --error=outputs/slurm/download_sa1b_%j.out
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
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

mkdir -p outputs/slurm datasets/raw/sa1b datasets/real_unlabeled/sa1b/images

# Télécharger 4 tars (~42 GB) avec suppression des tars après extraction
python scripts/download_sa1b.py \
    --links_file download-sa-1b.txt \
    --output_dir datasets/real_unlabeled/sa1b/images \
    --raw_dir datasets/raw/sa1b \
    --n_tars 4 \
    --delete_tar

EXIT_CODE=$?

echo ""
echo "============================================"
echo "  Terminé à : $(date)"
echo "  Exit code  : $EXIT_CODE"
echo ""
echo "  Images SA-1B :"
ls datasets/real_unlabeled/sa1b/images/ 2>/dev/null | wc -l
echo "============================================"

exit $EXIT_CODE
