#!/bin/bash
# ============================================================
# SLURM Job — Phase 4.2 : Entraînement initial Student (50k)
# ============================================================
# Soumission :   sbatch scripts/slurm/train_phase42.slurm
# Monitoring :   squeue -u $USER
# Annulation :   scancel <JOB_ID>
# Logs :         tail -f outputs/slurm/train_phase42_<JOB_ID>.out
# ============================================================

#SBATCH --job-name=dav2-train-p42
#SBATCH --output=outputs/slurm/train_phase42_%j.out
#SBATCH --error=outputs/slurm/train_phase42_%j.out
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=arcadia-slurm-node-4
#SBATCH --mail-type=BEGIN,END,FAIL          # Notifications par mail
#SBATCH --mail-user=pierlouis.pillet@telecom-sudparis.eu # <-- MODIFIER

export PYTHONUNBUFFERED=1

# ====================
# Configuration
# ====================
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(dirname $(dirname $(realpath $0)))}"
cd "$PROJECT_DIR"

echo "============================================"
echo "  SLURM Job ID     : $SLURM_JOB_ID"
echo "  Node              : $SLURM_NODELIST"
echo "  GPU               : $CUDA_VISIBLE_DEVICES"
echo "  Working dir       : $(pwd)"
echo "  Date              : $(date)"
echo "============================================"

# ====================
# Environnement
# ====================
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

# Créer les dossiers de sortie
mkdir -p outputs/slurm
mkdir -p outputs/checkpoints
mkdir -p outputs/logs

# ====================
# Lancement
# ====================
echo ""
echo ">>> Lancement entraînement Phase 4.2"
echo ""

python scripts/train.py \
    --images_dir datasets/real_unlabeled/sa1b/images \
    --pseudo_labels_dir outputs/pseudo_labels \
    --output_dir outputs \
    --epochs 20 \
    --batch_size 64 \
    --lr 1e-4 \
    --weight_decay 0.01 \
    --gradient_clip 1.0 \
    --lambda_ssi 0.5 \
    --alpha_gm 0.5 \
    --top_k_masking 0.1 \
    --num_workers $SLURM_CPUS_PER_TASK \
    --seed 42 \
    --resume outputs/checkpoints/best_model.pt # <-- Reprendre depuis le meilleur checkpoint (optionnel)

EXIT_CODE=$?

echo ""
echo "============================================"
echo "  Terminé à : $(date)"
echo "  Exit code  : $EXIT_CODE"
echo "============================================"

exit $EXIT_CODE
