#!/bin/bash
# ============================================================
# SLURM Job — Phase 4.4 v4 : Reprise Student v3 (cosine reset)
# ============================================================
# Reprend depuis checkpoint_epoch_020.pt (v3, epoch 20).
# Nouveau cycle cosine : LR 5e-5 → 1e-6 sur 20 epochs.
# Le scheduler est réinitialisé (reset_scheduler) : T_max=20,
# eta_min=1e-6, LR repart exactement à 5e-5 dès l'epoch 0.
#
# Prérequis :
#   outputs/v3/checkpoints/checkpoint_epoch_020.pt doit exister.
#
# Soumission :   sbatch scripts/slurm/train_student_v4.slurm
# Monitoring :   squeue -u $USER
# Logs :         tail -f outputs/slurm/train_student_v4_<JOB_ID>.out
# ============================================================

#SBATCH --job-name=dav2-student-v4
#SBATCH --output=outputs/slurm/train_student_v4_%j.out
#SBATCH --error=outputs/slurm/train_student_v4_%j.out
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=arcadia-slurm-node-4
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=pierlouis.pillet@telecom-sudparis.eu

export PYTHONUNBUFFERED=1

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(dirname $(dirname $(realpath $0)))}"
cd "$PROJECT_DIR"

echo "============================================"
echo "  SLURM Job ID     : $SLURM_JOB_ID"
echo "  Node              : $SLURM_NODELIST"
echo "  GPU               : $CUDA_VISIBLE_DEVICES"
echo "  Date              : $(date)"
echo "  Run               : Student v4 (cosine reset, LR 5e-5 → 1e-6, 20 epochs)"
echo "============================================"

eval "$(conda shell.bash hook)"
conda activate deeplearning

python -c "
import torch
print(f'PyTorch       : {torch.__version__}')
print(f'CUDA dispo    : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU           : {torch.cuda.get_device_name(0)}')
    print(f'VRAM          : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    print(f'AMP support   : {torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7}')
"

CHECKPOINT="outputs/v3/checkpoints/checkpoint_epoch_020.pt"
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERREUR : Checkpoint introuvable : $CHECKPOINT"
    echo "Vérifier que l'entraînement v3 a bien produit ce fichier."
    exit 1
fi
echo ">>> Checkpoint source : $CHECKPOINT"

mkdir -p outputs/slurm outputs/v4/checkpoints outputs/v4/logs

echo ""
echo ">>> Student v4 — LR reset 5e-5 → 1e-6, T_max=20, depuis checkpoint v3 epoch 20"
echo ">>> reset_scheduler=True : nouveau cycle cosine complet, best_val_loss réinitialisé"
echo ""

python scripts/train.py \
    --images_dir datasets/real_unlabeled/sa1b/images \
    --pseudo_labels_dir outputs/pseudo_labels_v3 \
    --output_dir outputs/v4 \
    --resume "$CHECKPOINT" \
    --reset_scheduler \
    --epochs 20 \
    --lr 5e-5 \
    --eta_min 1e-6 \
    --weight_decay 0.01 \
    --batch_size 64 \
    --gradient_clip 1.0 \
    --lambda_ssi 0.5 \
    --alpha_gm 0.5 \
    --top_k_masking 0.1 \
    --early_stopping_patience 5 \
    --num_workers $SLURM_CPUS_PER_TASK \
    --seed 42 \
    --amp

EXIT_CODE=$?

echo ""
echo "============================================"
echo "  Terminé à : $(date)"
echo "  Exit code  : $EXIT_CODE"
if [ $EXIT_CODE -eq 0 ]; then
    echo "  Meilleur modèle : outputs/v4/checkpoints/best_model.pt"
    echo "  Évaluation      : sbatch scripts/slurm/evaluate.slurm (modifier --checkpoint)"
fi
echo "============================================"

exit $EXIT_CODE
