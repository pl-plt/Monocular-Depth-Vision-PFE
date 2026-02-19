#!/bin/bash
# ============================================================
# SLURM Job — Phase 5 : Évaluation finale (NYU + KITTI)
# ============================================================

#SBATCH --job-name=dav2-eval
#SBATCH --output=outputs/slurm/evaluate_%j.out
#SBATCH --error=outputs/slurm/evaluate_%j.out
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00                     # Évaluation rapide
#SBATCH --nodes=1
#SBATCH --ntasks=1

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(dirname $(dirname $(realpath $0)))}"
cd "$PROJECT_DIR"

echo "============================================"
echo "  Évaluation finale — $SLURM_JOB_ID"
echo "  Date : $(date)"
echo "============================================"

# Environnement (décommenter selon ton cluster)
# module purge && module load python/3.10 cuda/12.1
# conda activate deeplearning

mkdir -p outputs/slurm outputs/evaluation

python scripts/evaluate.py \
    --checkpoint outputs/checkpoints/best_model.pt \
    --output_dir outputs/evaluation

EXIT_CODE=$?
echo "  Terminé à : $(date) | Exit code : $EXIT_CODE"
exit $EXIT_CODE
