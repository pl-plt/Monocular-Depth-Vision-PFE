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

# Vérifier que le dataset est présent
NYU_DIR="datasets/real_depth/nyudepthv2"
if [ ! -d "$NYU_DIR/images" ] || [ "$(ls $NYU_DIR/images/ 2>/dev/null | wc -l)" -lt 654 ]; then
    echo ""
    echo "  ⚠ NYU test set introuvable ou incomplet dans $NYU_DIR"
    echo "     Lancer d'abord : sbatch scripts/slurm/download_nyu_test.slurm"
    exit 1
fi

NYU_COUNT=$(ls "$NYU_DIR/images/" | wc -l)
echo ""
echo "  NYU test : $NYU_COUNT images"
echo ""

mkdir -p outputs/slurm outputs/evaluation

python scripts/evaluate.py \
    --checkpoint outputs/v2/checkpoints/best_model.pt \
    --nyu_dir "$NYU_DIR" \
    --batch_size 8 \
    --output_dir outputs/evaluation

EXIT_CODE=$?

echo ""
echo "============================================"
echo "  Terminé à : $(date)"
echo "  Exit code  : $EXIT_CODE"
echo "  Résultats  : outputs/evaluation/benchmark_results.json"
echo "============================================"

exit $EXIT_CODE
