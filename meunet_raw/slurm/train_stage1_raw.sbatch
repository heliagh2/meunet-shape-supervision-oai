#!/bin/bash
#SBATCH --job-name=meu-s1-raw
#SBATCH --account=gisr107588
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --chdir=/projects/prjs1722/oai_baselines/meunet_raw
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

echo "=== MeUNet Stage 1 RAW training ==="
echo "whoami=$(whoami) host=$(hostname) date=$(date)"

module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

source ~/venvs/oai-baseline/bin/activate

python -u -m train.train_stage1 --config configs/oai_stage1_raw.yaml

echo "=== Done Stage 1 RAW training ==="
