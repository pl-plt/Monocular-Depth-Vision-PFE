#!/bin/bash
# ============================================================
# SLURM Job — Phase 4.2 v3 : Entraînement Student (from scratch)
# ============================================================
# À lancer APRÈS generate_pseudo_labels_v3.slurm.
# Entraîne le Student (DINOv2-Small + DPT) sur les pseudo-labels
# générés par le Teacher v3 (architecture corrigée).
#
# AUCUNE reprise depuis checkpoint : les pseudo-labels v2 et les
# checkpoints v2 sont incompatibles avec les corrections v3.
#
# Prérequis :
#   outputs/pseudo_labels_v3/ doit contenir les .npy générés.
#
# Soumission :   sbatch scripts/slurm/train_student_v3.slurm
# Monitoring :   squeue -u $USER
# Logs :         tail -f outputs/slurm/train_student_v3_<JOB_ID>.out
# ============================================================

#SBATCH --job-name=dav2-student-v3
#SBATCH --output=outputs/slurm/train_student_v3_%j.out
#SBATCH --error=outputs/slurm/train_student_v3_%j.out
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
echo "  Run               : Student v3 (from scratch, AMP)"
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

# Vérifier que les pseudo-labels v3 sont disponibles
PSEUDO_DIR="outputs/pseudo_labels_v3"
PSEUDO_COUNT=$(find "$PSEUDO_DIR" -name "*.npy" 2>/dev/null | wc -l)
if [ "$PSEUDO_COUNT" -eq 0 ]; then
    echo "ERREUR : Aucun pseudo-label trouvé dans $PSEUDO_DIR"
    echo "Lancer d'abord : sbatch scripts/slurm/generate_pseudo_labels_v3.slurm"
    exit 1
fi
echo ">>> Pseudo-labels v3 : $PSEUDO_COUNT fichiers dans $PSEUDO_DIR"

SA1B_COUNT=$(find datasets/real_unlabeled/sa1b/images -maxdepth 1 \( -name "*.jpg" -o -type l \) 2>/dev/null | wc -l)
echo ">>> Images SA-1B     : $SA1B_COUNT"

mkdir -p outputs/slurm outputs/v3/checkpoints outputs/v3/logs

echo ""
echo ">>> Entraînement Student v3 (DINOv2-Small + DPT decoder corrigé)"
echo ">>> AUCUNE reprise depuis checkpoint (from scratch)"
echo ""

python scripts/train.py \
    --images_dir datasets/real_unlabeled/sa1b/images \
    --pseudo_labels_dir outputs/pseudo_labels_v3 \
    --output_dir outputs/v3 \
    --epochs 30 \
    --batch_size 64 \
    --lr 1e-4 \
    --weight_decay 0.01 \
    --gradient_clip 1.0 \
    --lambda_ssi 0.5 \
    --alpha_gm 0.5 \
    --top_k_masking 0.1 \
    --num_workers $SLURM_CPUS_PER_TASK \
    --seed 42 \
    --amp

EXIT_CODE=$?

echo ""
echo "============================================"
echo "  Terminé à : $(date)"
echo "  Exit code  : $EXIT_CODE"
if [ $EXIT_CODE -eq 0 ]; then
    echo "  Meilleur modèle : outputs/v3/checkpoints/best_model.pt"
    echo "  Évaluation      : sbatch scripts/slurm/evaluate.slurm (modifier --checkpoint)"
fi
echo "============================================"

exit $EXIT_CODE
