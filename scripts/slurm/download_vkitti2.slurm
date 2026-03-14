#!/bin/bash
# ============================================================
# SLURM Job — Téléchargement Virtual KITTI 2
# ============================================================
# Virtual KITTI 2 : 5 scènes outdoor, ~21k images avec depth GT.
# Complément outdoor au dataset Hypersim (indoor).
#
# Downloads :
#   vkitti_2.0.3_rgb.tar    7.01 GB
#   vkitti_2.0.3_depth.tar  7.58 GB
# Output final (après extraction + conversion) : ~5-8 GB
#
# Soumission :   sbatch scripts/slurm/download_vkitti2.slurm
# Logs :         tail -f outputs/slurm/download_vkitti2_<JOB_ID>.out
# ============================================================

#SBATCH --job-name=dav2-vkitti2
#SBATCH --output=outputs/slurm/download_vkitti2_%j.out
#SBATCH --error=outputs/slurm/download_vkitti2_%j.out
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

export PYTHONUNBUFFERED=1

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(dirname $(dirname $(realpath $0)))}"
cd "$PROJECT_DIR"

echo "============================================"
echo "  SLURM Job ID     : $SLURM_JOB_ID"
echo "  Node              : $SLURM_NODELIST"
echo "  Date              : $(date)"
echo "============================================"

# Environnement
eval "$(conda shell.bash hook)"
conda activate deeplearning

# Vérifier espace disque (~35 GB pendant téléchargement + extraction)
echo ""
echo "Espace disque :"
df -h . | head -2
echo ""

mkdir -p outputs/slurm datasets/raw/vkitti2 datasets/synthetic/vkitti2

python scripts/download_vkitti2.py \
    --output_dir datasets/synthetic/vkitti2 \
    --raw_dir datasets/raw/vkitti2 \
    --resume

EXIT_CODE=$?

echo ""
echo "============================================"
echo "  Terminé à : $(date)"
echo "  Exit code  : $EXIT_CODE"
echo ""
echo "  Images VKitti2 : $(find datasets/synthetic/vkitti2/images -name '*.*' 2>/dev/null | wc -l)"
echo "  Depth  VKitti2 : $(find datasets/synthetic/vkitti2/depth -name '*.npy' 2>/dev/null | wc -l)"
echo "============================================"

exit $EXIT_CODE
