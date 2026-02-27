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
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=arcadia-slurm-node-2
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

# Diagnostic GPU — avorter si pas de CUDA
echo ""
echo ">>> Diagnostic GPU :"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  ✗ nvidia-smi introuvable"
python -c "import torch; print(f'  PyTorch: {torch.__version__}'); print(f'  CUDA dispo: {torch.cuda.is_available()}'); print(f'  GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else '  ✗ PAS DE GPU — le job va être très lent !')"
echo ""

mkdir -p outputs/slurm outputs/pseudo_labels

# Combiner NYU + SA-1B dans un dossier unifié via symlinks
COMBINED_DIR="datasets/real_unlabeled/combined/images"
mkdir -p "$COMBINED_DIR"

# Vérifier si les symlinks existent déjà
EXISTING=$(find "$COMBINED_DIR" -maxdepth 1 -type l 2>/dev/null | wc -l)

if [ "$EXISTING" -gt 0 ]; then
    echo ">>> Symlinks déjà présents ($EXISTING liens), skip la création"
    NYU_COUNT=$(find "$COMBINED_DIR" -maxdepth 1 -name "nyu_*.png" | wc -l)
    SA1B_COUNT=$(find "$COMBINED_DIR" -maxdepth 1 -name "sa_*.jpg" | wc -l)
else
    echo ">>> Création du dossier combiné (NYU + SA-1B)"

    # Symlinks NYU (rapide, ~795 fichiers)
    NYU_COUNT=0
    if [ -d "datasets/real_unlabeled/indoor/images" ]; then
        NYU_SRC="$(realpath datasets/real_unlabeled/indoor/images)"
        find "$NYU_SRC" -maxdepth 1 -name "*.png" -exec ln -sf {} "$COMBINED_DIR/" \;
        NYU_COUNT=$(find "$COMBINED_DIR" -maxdepth 1 -name "nyu_*.png" | wc -l)
    fi
    echo "    NYU  : $NYU_COUNT images"

    # Symlinks SA-1B (rapide avec find, évite la boucle bash lente)
    SA1B_COUNT=0
    if [ -d "datasets/real_unlabeled/sa1b/images" ]; then
        SA1B_SRC="$(realpath datasets/real_unlabeled/sa1b/images)"
        find "$SA1B_SRC" -maxdepth 1 -name "*.jpg" -exec ln -sf {} "$COMBINED_DIR/" \;
        SA1B_COUNT=$(find "$COMBINED_DIR" -maxdepth 1 -name "sa_*.jpg" | wc -l)
    fi
    echo "    SA-1B: $SA1B_COUNT images"
fi

TOTAL=$(find "$COMBINED_DIR" -maxdepth 1 -type l | wc -l)
echo "    NYU: $NYU_COUNT | SA-1B: $SA1B_COUNT | Total: $TOTAL symlinks"

echo ""
echo ">>> Génération des pseudo-labels (Teacher DINOv2-Giant)"
echo ">>> Images : NYU ($NYU_COUNT) + SA-1B ($SA1B_COUNT) = $TOTAL"
echo ""

python scripts/generate_pseudo_labels.py \
    --teacher_weights outputs/teacher/checkpoints/best_model.pt \
    --images_dir "$COMBINED_DIR" \
    --output_dir outputs/pseudo_labels \
    --batch_size 2 \
    --half \
    --num_workers $SLURM_CPUS_PER_TASK

EXIT_CODE=$?

echo ""
echo "  Terminé à : $(date) | Exit code : $EXIT_CODE"
exit $EXIT_CODE
