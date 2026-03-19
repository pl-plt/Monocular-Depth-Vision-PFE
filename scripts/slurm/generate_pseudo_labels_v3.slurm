#!/bin/bash
# ============================================================
# SLURM Job — Phase 4.1 v3 : Génération de pseudo-labels
# ============================================================
# À lancer APRÈS train_teacher_v3.slurm.
# Génère les pseudo-labels avec le Teacher v3 (architecture corrigée).
#
# Prérequis :
#   outputs/teacher_v3/checkpoints/best_model.pt doit exister.
#
# Soumission :   sbatch scripts/slurm/generate_pseudo_labels_v3.slurm
# Monitoring :   squeue -u $USER
# Logs :         tail -f outputs/slurm/pseudo_labels_v3_<JOB_ID>.out
# ============================================================

#SBATCH --job-name=dav2-pseudo-v3
#SBATCH --output=outputs/slurm/pseudo_labels_v3_%j.out
#SBATCH --error=outputs/slurm/pseudo_labels_v3_%j.out
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
echo "  Run               : Pseudo-labels v3 (Teacher v3)"
echo "============================================"

# Environnement
eval "$(conda shell.bash hook)"
conda activate deeplearning

# Diagnostic GPU
python -c "
import torch
print(f'PyTorch       : {torch.__version__}')
print(f'CUDA dispo    : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU           : {torch.cuda.get_device_name(0)}')
    print(f'VRAM          : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# Vérifier que le Teacher v3 est disponible
TEACHER_CKPT="outputs/teacher_v3/checkpoints/best_model.pt"
if [ ! -f "$TEACHER_CKPT" ]; then
    echo "ERREUR : Teacher v3 introuvable : $TEACHER_CKPT"
    echo "Lancer d'abord : sbatch scripts/slurm/train_teacher_v3.slurm"
    exit 1
fi
echo ">>> Teacher v3 trouvé : $TEACHER_CKPT"

mkdir -p outputs/slurm outputs/pseudo_labels_v3

# Construire le dossier SA-1B symlinks (si pas déjà fait)
COMBINED_DIR="datasets/real_unlabeled/sa1b/images"
if [ ! -d "$COMBINED_DIR" ]; then
    echo "ERREUR : Répertoire SA-1B introuvable : $COMBINED_DIR"
    exit 1
fi

SA1B_COUNT=$(find "$COMBINED_DIR" -maxdepth 1 \( -name "*.jpg" -o -type l \) 2>/dev/null | wc -l)
echo ""
echo ">>> Génération des pseudo-labels avec Teacher v3"
echo ">>> Images SA-1B : $SA1B_COUNT"
echo ">>> Output : outputs/pseudo_labels_v3/"
echo ""

python scripts/generate_pseudo_labels.py \
    --teacher_weights "$TEACHER_CKPT" \
    --images_dir "$COMBINED_DIR" \
    --output_dir outputs/pseudo_labels_v3 \
    --batch_size 4 \
    --half \
    --num_workers $SLURM_CPUS_PER_TASK

EXIT_CODE=$?

PSEUDO_COUNT=$(find outputs/pseudo_labels_v3 -name "*.npy" 2>/dev/null | wc -l)
echo ""
echo "============================================"
echo "  Terminé à    : $(date)"
echo "  Exit code    : $EXIT_CODE"
echo "  Pseudo-labels générés : $PSEUDO_COUNT"
if [ $EXIT_CODE -eq 0 ]; then
    echo "  Prochaine étape : sbatch scripts/slurm/train_student_v3.slurm"
fi
echo "============================================"

exit $EXIT_CODE
