#!/bin/bash
#SBATCH --job-name=meu-s1-eval-desc-all-both-esp100-maxEXP-600ep
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

echo "=== Stage-1 eval start for Descriptor only all classes ESP=100, maximised EXP patches, using both STD and EXP patches, max 600epochs==="
echo "whoami=$(whoami) host=$(hostname) date=$(date)"

source ~/venvs/oai-baseline/bin/activate

GT_DIR=/projects/prjs1722/oai_baselines/nnunet/nnUNet_raw/Dataset701_OAIKnee/labelsTs
PRED_DIR=/projects/prjs1722/oai_baselines/meunet_raw/results/stage1_desc_only_all_both_ESP100_maxEXP_600ep/test_preds_desc_all_both_ESP100_maxEXP_600ep
OUT_CSV=/projects/prjs1722/oai_baselines/meunet_raw/results/stage1_desc_only_all_both_ESP100_maxEXP_600ep/meunet_stage1_desc_only_all_both_ESP100_maxEXP_600ep_test.csv

python -u eval/eval_dice_oai.py \
  --gt_dir "$GT_DIR" \
  --pred_dir "$PRED_DIR" \
  --out_csv "$OUT_CSV"

echo "=== Stage-1 eval done for Descriptor only all classes ESP=100, maximised EXP patches, using both STD and EXP patches, max 600epochs==="