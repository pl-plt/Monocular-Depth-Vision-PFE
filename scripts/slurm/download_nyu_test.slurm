#!/bin/bash
# ============================================================
# SLURM Job — Téléchargement NYU-Depth V2 test set (654 images)
# ============================================================
# Extrait les 654 images de test + depth maps GT depuis le même
# fichier .mat que les données d'entraînement.
#
# Si datasets/raw/indoor/nyu_depth_v2_labeled.mat existe déjà
# (téléchargé lors de download_indoor), le script le réutilise
# sans re-télécharger (~2.8 GB économisés).
#
# Sortie :
#   datasets/real_depth/nyudepthv2/images/  → 654 PNG RGB
#   datasets/real_depth/nyudepthv2/depth/   → 654 .npy float32 (mètres)
#
# Soumission :   sbatch scripts/slurm/download_nyu_test.slurm
# Logs :         tail -f outputs/slurm/download_nyu_test_<JOB_ID>.out
# ============================================================

#SBATCH --job-name=dav2-nyu-test
#SBATCH --output=outputs/slurm/download_nyu_test_%j.out
#SBATCH --error=outputs/slurm/download_nyu_test_%j.out
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
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

# Vérifier espace disque (extraction ~600 MB)
echo ""
echo "Espace disque :"
df -h . | head -2
echo ""

mkdir -p outputs/slurm datasets/raw/nyudepthv2 datasets/real_depth/nyudepthv2

python scripts/download_nyu_test.py \
    --output_dir datasets/real_depth/nyudepthv2 \
    --raw_dir datasets/raw/nyudepthv2 \
    --resume

EXIT_CODE=$?

echo ""
echo "============================================"
echo "  Terminé à : $(date)"
echo "  Exit code  : $EXIT_CODE"
echo ""
echo "  Images test NYU  : $(ls datasets/real_depth/nyudepthv2/images/ 2>/dev/null | wc -l)"
echo "  Depth maps NYU   : $(ls datasets/real_depth/nyudepthv2/depth/ 2>/dev/null | wc -l)"
echo "============================================"

exit $EXIT_CODE
