#!/bin/bash
#SBATCH --job-name=meu-s1-desc-only-all-both-best-logbarrier-centx5-fixedpatch
#SBATCH --account=gisr107588
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-06:00:00
#SBATCH --chdir=/projects/prjs1722/oai_baselines/meunet_raw
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

echo "set no memory constraint"

set -euo pipefail

module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
source ~/venvs/oai-baseline/bin/activate

python -u train/train_stage1_desc_only_logbarrier_fixedpatch.py \
  --config configs/oai_stage1_desc_only_all_both_best_logbarrier_centx5_fixedpatch.yaml

echo "=== Done MeUNet Stage 1 RAW — descriptor-only, experiment centroid focus ==="
echo "dataset: fixed per-case sampling (1 STD + 1 EXP patch reused for entire training)"

