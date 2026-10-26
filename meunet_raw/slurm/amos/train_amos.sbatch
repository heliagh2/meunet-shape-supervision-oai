#!/bin/bash
# Train one AMOS22 config. Needs prep_amos.sbatch to have run once.
#
#   cd /home/isimion/meunet-shape-supervision-oai/meunet_raw && mkdir -p logs
#   sbatch -J amos-A2.1  slurm/amos/train_amos.sbatch config/amos/AMOS_A2_1_distribution.yaml
#   sbatch -J amos-A1.11 slurm/amos/train_amos.sbatch config/amos/AMOS_A1_11_concentration.yaml
#   sbatch -J amos-A0    slurm/amos/train_amos.sbatch config/amos/AMOS_A0_concentration.yaml
#SBATCH --job-name=amos-train
#SBATCH --account=gisr107588
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --chdir=/home/isimion/meunet-shape-supervision-oai/meunet_raw
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

CONFIG=${1:?usage: sbatch slurm/amos/train_amos.sbatch config/amos/<name>.yaml}

module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
source ~/venvs/oai-baseline/bin/activate

nvidia-smi --query-gpu=name,memory.total --format=csv

python -u train/train_stage1_desc_only_logbarrier_fixedpatch.py --config "$CONFIG"

echo "=== Done AMOS run: $CONFIG ==="
