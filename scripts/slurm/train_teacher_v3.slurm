#!/bin/bash
# ============================================================
# SLURM Job — Teacher v3 : Entraînement / reprise (corrigé)
# ============================================================
# Reprise automatique : si outputs/teacher_v3/checkpoints/best_model.pt
# existe, l'entraînement reprend depuis cet epoch (LR, early-stopping,
# etc. restaurés). Sinon, repart from scratch.
#
# IMPORTANT (historique) : Le Teacher v2 était incompatible avec le fix
# DPT fusion order :
# la fusion partait du plus grand feature map (36×36 effectif),
# tous les pseudo-labels v2 sont donc de mauvaise qualité.
# Ce run repart à zéro avec l'architecture corrigée.
#
# Corrections appliquées (v3) :
#   - DPT fusion order : bottom-up (18×18 → 296×296) ✓
#   - Scheduler resume : reset base_lrs avant step() ✓
#   - Val transform séparé (pas de data augmentation sur val) ✓
#   - L_gm en log-space ✓
#   - sqrt clamp stabilisé ✓
#   - AMP activé (2× plus rapide sur H100 NVL) ✓
#
# Soumission :   sbatch scripts/slurm/train_teacher_v3.slurm
# Monitoring :   squeue -u $USER
# Logs :         tail -f outputs/slurm/train_teacher_v3_<JOB_ID>.out
# ============================================================

#SBATCH --job-name=dav2-teacher-v3
#SBATCH --output=outputs/slurm/train_teacher_v3_%j.out
#SBATCH --error=outputs/slurm/train_teacher_v3_%j.out
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=08:00:00
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
echo "  Run               : Teacher v3 (from scratch, AMP)"
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
    print(f'AMP support   : {torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7}')
"

mkdir -p outputs/slurm outputs/teacher_v3/checkpoints outputs/teacher_v3/logs

# Auto-resume : reprendre depuis le meilleur checkpoint s'il existe
RESUME_ARG=""
CKPT_PATH="outputs/teacher_v3/checkpoints/best_model.pt"
if [ -f "$CKPT_PATH" ]; then
    echo ">>> Checkpoint trouvé : $CKPT_PATH → reprise automatique"
    RESUME_ARG="--resume $CKPT_PATH"
else
    echo ">>> Aucun checkpoint trouvé → entraînement from scratch"
fi

echo ""
echo ">>> Entraînement Teacher v3 (DINOv2-Giant + DPT decoder corrigé)"
echo ">>> Datasets : Hypersim (indoor) + Virtual KITTI 2 (outdoor)"
echo ">>> Backbone frozen, decoder trainable, AMP activé"
echo ""

# Vérifier les datasets disponibles
HYPERSIM_COUNT=$(find datasets/synthetic/hypersim/images -name '*.png' 2>/dev/null | wc -l)
VKITTI2_COUNT=$(find datasets/synthetic/vkitti2/images -name '*.jpg' 2>/dev/null | wc -l)
echo "Datasets synthétiques :"
echo "  Hypersim : $HYPERSIM_COUNT images"
echo "  VKitti2  : $VKITTI2_COUNT images"
echo ""

if [ "$HYPERSIM_COUNT" -eq 0 ] && [ "$VKITTI2_COUNT" -eq 0 ]; then
    echo "ERREUR : Aucun dataset trouvé. Vérifier datasets/synthetic/."
    exit 1
fi

python scripts/train_teacher.py \
    --dataset_dirs datasets/synthetic/hypersim datasets/synthetic/vkitti2 \
    --output_dir outputs/teacher_v3 \
    --epochs 30 \
    --batch_size 16 \
    --lr 1e-4 \
    --weight_decay 0.01 \
    --gradient_clip 1.0 \
    --lambda_ssi 0.5 \
    --alpha_gm 0.5 \
    --top_k_masking 0.1 \
    --num_workers $SLURM_CPUS_PER_TASK \
    --seed 42 \
    --amp \
    $RESUME_ARG

EXIT_CODE=$?

echo ""
echo "============================================"
echo "  Terminé à : $(date)"
echo "  Exit code  : $EXIT_CODE"
if [ $EXIT_CODE -eq 0 ]; then
    echo "  Checkpoint : outputs/teacher_v3/checkpoints/best_model.pt"
    echo "  Prochaine étape : sbatch scripts/slurm/generate_pseudo_labels_v3.slurm"
fi
echo "============================================"

exit $EXIT_CODE
