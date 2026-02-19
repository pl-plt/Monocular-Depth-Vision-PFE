#!/bin/bash
# ============================================================
# SLURM Job — Phase 4.3 : Scale-up Student (200k)
# ============================================================
# Soumission :   sbatch scripts/slurm/train_phase43.slurm
# Reprend depuis le meilleur checkpoint de Phase 4.2
# ============================================================

#SBATCH --job-name=dav2-train-p43
#SBATCH --output=outputs/slurm/train_phase43_%j.out
#SBATCH --error=outputs/slurm/train_phase43_%j.out
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
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

mkdir -p outputs/slurm outputs/checkpoints outputs/logs

# Vérifier que le checkpoint Phase 4.2 existe
CHECKPOINT="outputs/checkpoints/best_model.pt"
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERREUR : Checkpoint introuvable : $CHECKPOINT"
    echo "    → Lancer d'abord Phase 4.2 (train_phase42.slurm)"
    exit 1
fi

echo ""
echo ">>> Lancement entraînement Phase 4.3 (200k images, reprise depuis Phase 4.2)"
echo ""

python scripts/train.py \
    --images_dir datasets/real_unlabeled/sa1b_200k/images \
    --pseudo_labels_dir outputs/pseudo_labels_200k \
    --output_dir outputs \
    --resume "$CHECKPOINT" \
    --epochs 15 \
    --batch_size 16 \
    --lr 5e-5 \
    --weight_decay 0.01 \
    --gradient_clip 1.0 \
    --num_workers $SLURM_CPUS_PER_TASK \
    --seed 42

EXIT_CODE=$?

echo ""
echo "  Terminé à : $(date) | Exit code : $EXIT_CODE"
exit $EXIT_CODE
