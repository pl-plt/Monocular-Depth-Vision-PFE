#!/bin/bash
# ============================================================
# SLURM Job — Phase 3 : Entraînement Teacher sur synthétiques
# ============================================================
# Le Teacher = DINOv2-Giant (backbone frozen) + DPT decoder (trainable)
# Entraîné sur Hypersim / Virtual KITTI 2 avec GT depth
#
# Soumission :   sbatch scripts/slurm/train_teacher.slurm
# Monitoring :   squeue -u $USER
# Logs :         tail -f outputs/slurm/train_teacher_<JOB_ID>.out
# ============================================================

#SBATCH --job-name=dav2-teacher
#SBATCH --output=outputs/slurm/train_teacher_%j.out
#SBATCH --error=outputs/slurm/train_teacher_%j.out
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=arcadia-slurm-node-2

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

# Vérifier PyTorch + GPU
python -c "
import torch
print(f'PyTorch       : {torch.__version__}')
print(f'CUDA dispo    : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU           : {torch.cuda.get_device_name(0)}')
    print(f'VRAM          : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

mkdir -p outputs/slurm outputs/teacher/checkpoints outputs/teacher/logs

echo ""
echo ">>> Entraînement Teacher (DINOv2-Giant + DPT decoder)"
echo ">>> Dataset : synthétiques (Hypersim)"
echo ">>> Backbone frozen, decoder trainable"
echo ""

# Auto-resume : reprendre depuis le meilleur checkpoint s'il existe
RESUME_ARG=""
CKPT_PATH="outputs/teacher/checkpoints/best_model.pt"
if [ -f "$CKPT_PATH" ]; then
    echo ">>> Checkpoint trouvé : $CKPT_PATH → reprise automatique"
    RESUME_ARG="--resume $CKPT_PATH"
else
    echo ">>> Aucun checkpoint trouvé → entraînement from scratch"
fi
echo ""

python scripts/train_teacher.py \
    --dataset_dir datasets/synthetic/hypersim \
    --output_dir outputs/teacher \
    --epochs 25 \
    --batch_size 4 \
    --lr 1e-4 \
    --weight_decay 0.01 \
    --gradient_clip 1.0 \
    --lambda_ssi 0.5 \
    --alpha_gm 0.5 \
    --top_k_masking 0.1 \
    --num_workers $SLURM_CPUS_PER_TASK \
    --seed 42 \
    $RESUME_ARG

EXIT_CODE=$?

echo ""
echo "============================================"
echo "  Terminé à : $(date)"
echo "  Exit code  : $EXIT_CODE"
echo "============================================"

exit $EXIT_CODE
