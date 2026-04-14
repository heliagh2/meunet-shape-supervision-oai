#!/bin/bash
#SBATCH --job-name=meu-s1-eval-desc-all-both-best-logbarrier-centx5-fixedpatch
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

echo "=== Stage-1 eval start for Descriptor only all classes, experiment with logbarrier (both patches) centroid focus==="
echo "whoami=$(whoami) host=$(hostname) date=$(date)"
echo "dataset: fixed per-case sampling (1 STD + 1 EXP patch reused for entire training)"

source ~/venvs/oai-baseline/bin/activate

GT_DIR=/projects/prjs1722/oai_baselines/nnunet/nnUNet_raw/Dataset701_OAIKnee/labelsTs
PRED_DIR=/projects/prjs1722/oai_baselines/meunet_raw/results/stage1_desc_only_all_both_best_logbarrier_centx5_fixedpatch/test_preds_desc_all_both_best_logbarrier_centx5_fixedpatch
OUT_CSV=/projects/prjs1722/oai_baselines/meunet_raw/results/stage1_desc_only_all_both_best_logbarrier_centx5_fixedpatch/meunet_stage1_desc_only_all_both_best_logbarrier_centx5_fixedpatch_test.csv

python -u eval/eval_dice_oai.py \
  --gt_dir "$GT_DIR" \
  --pred_dir "$PRED_DIR" \
  --out_csv "$OUT_CSV"

echo "=== Stage-1 eval done for Descriptor only all classes, experiment with logbarrier (both patches) centroid focus==="
echo "dataset: fixed per-case sampling (1 STD + 1 EXP patch reused for entire training)"
