#!/bin/bash
# ============================================================
# SLURM Job — Phase 4.1 : Génération de pseudo-labels (Teacher)
# ============================================================
# À lancer AVANT l'entraînement Student.
# Le Teacher (DINOv2-Giant) est lourd → besoin de VRAM importante.
# ============================================================

#SBATCH --job-name=dav2-pseudo
#SBATCH --output=outputs/slurm/pseudo_labels_%j.out
#SBATCH --error=outputs/slurm/pseudo_labels_%j.out
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ton.email@universite.fr

export PYTHONUNBUFFERED=1

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(dirname $(dirname $(realpath $0)))}"
cd "$PROJECT_DIR"

echo "============================================"
echo "  SLURM Job ID     : $SLURM_JOB_ID"
echo "  Node              : $SLURM_NODELIST"
echo "  GPU               : $CUDA_VISIBLE_DEVICES"
echo "  Date              : $(date)"
echo "============================================"

# Environnement
eval "$(conda shell.bash hook)"
conda activate deeplearning

mkdir -p outputs/slurm outputs/pseudo_labels

echo ""
echo ">>> Génération des pseudo-labels (Teacher DINOv2-Giant)"
echo ">>> Images : indoor (NYU + SUN RGB-D)"
echo ""

python scripts/generate_pseudo_labels.py \
    --teacher_weights outputs/teacher/checkpoints/best_model.pt \
    --images_dir datasets/real_unlabeled/indoor/images \
    --output_dir outputs/pseudo_labels \
    --batch_size 8 \
    --num_workers $SLURM_CPUS_PER_TASK

EXIT_CODE=$?

echo ""
echo "  Terminé à : $(date) | Exit code : $EXIT_CODE"
exit $EXIT_CODE
