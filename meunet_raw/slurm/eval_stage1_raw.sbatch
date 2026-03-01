#!/bin/bash
#SBATCH --job-name=meu-s1-eval-raw
#SBATCH --account=gisr107588
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --chdir=/projects/prjs1722/oai_baselines/meunet_raw
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

echo "=== Stage-1 eval start ==="
echo "whoami=$(whoami) host=$(hostname) date=$(date)"

source ~/venvs/oai-baseline/bin/activate

GT_DIR=/projects/prjs1722/oai_baselines/nnunet/nnUNet_raw/Dataset701_OAIKnee/labelsTs
PRED_DIR=/projects/prjs1722/oai_baselines/meunet_raw/results/stage1/test_preds_raw_fixed
OUT_CSV=/projects/prjs1722/oai_baselines/meunet_raw/results/stage1/meunet_stage1_raw_test.csv

python -u eval/eval_dice_oai.py \
  --gt_dir "$GT_DIR" \
  --pred_dir "$PRED_DIR" \
  --out_csv "$OUT_CSV"

echo "=== Stage-1 eval done ==="