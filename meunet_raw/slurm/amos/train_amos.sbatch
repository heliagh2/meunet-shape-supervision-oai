#!/bin/bash
# Train one AMOS22 config. Needs prep_amos.sbatch to have run once.
#
#   cd /home/isimion/meunet-shape-supervision-oai/meunet_raw && mkdir -p logs
#   sbatch -J amos-A2.1  slurm/amos/train_amos.sbatch config/amos/AMOS_A2_1_distribution.yaml
#   sbatch -J amos-A1.11 slurm/amos/train_amos.sbatch config/amos/AMOS_A1_11_concentration.yaml
#   sbatch -J amos-A0    slurm/amos/train_amos.sbatch config/amos/AMOS_A0_concentration.yaml
#SBATCH --job-name=amos-train
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --chdir=/home/isimion1/meunet-shape-supervision-oai/meunet_raw
#SBATCH --output=logs/train_amos-%j.out
#SBATCH --error=logs/train_amos-%j.err

set -euo pipefail

CONFIG=${1:?usage: sbatch slurm/amos/train_amos.sbatch config/amos/<name>.yaml}

module purge
source ~/envs/medshape/bin/activate

export WANDB_ENTITY="i-simion-university-of-amsterdam"
source ~/.secrets/wandb.env

python -u train/train_stage1_desc_only_logbarrier_fixedpatch.py --config "$CONFIG"

echo "=== Done AMOS run: $CONFIG ==="
