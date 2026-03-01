#!/bin/bash
#SBATCH --job-name=meu-s1-desc-only-all-both-ESP100-maxEXP
#SBATCH --account=gisr107588
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-06:00:00
#SBATCH --chdir=/projects/prjs1722/oai_baselines/meunet_raw
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
source ~/venvs/oai-baseline/bin/activate

python -u train/train_stage1_desc_only.py \
  --config configs/oai_stage1_desc_only_all_both_ESP100_maxEXP.yaml

echo "=== Done MeUNet Stage 1 RAW — descriptor-only (all classes, EXP + STD, Early stop patience = 100, Maximised Patches [384,384,160]) ==="
