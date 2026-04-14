#!/bin/bash
#SBATCH --job-name=meu-s1-test-desc-only-all-both-best-logbarrier-centx5-fixedpatch
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --gres=gpu:titan_rtx:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --chdir=/home/isimion/meunet-shape-supervision-oai/meunet_raw
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

echo "=== Stage-1 raw inference start Descriptor Only all classes, experiment with logbarrier (both patches) centroid focus==="
echo "whoami=$(whoami) host=$(hostname) date=$(date)"
echo "dataset: fixed per-case sampling (1 STD + 1 EXP patch reused for entire training)"

module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

source ~/venvs/oai-baseline/bin/activate

python eval/infer_stage1_oai.py \
  --config config/oai_stage1_desc_only_all_both_best_logbarrier_centx5_fixedpatch.yaml \
  --stage1_ckpt /home/isimion/meunet-shape-supervision-oai/oai_baselines/meunet_raw/results/stage1_desc_only_all_both_best_logbarrier_centx5_fixedpatch/checkpoint_best.pt \
  --images_dir /ivi/zfs/s0/original_homes/isimion/knee_dataset/OAI_ZIB_DESS \
  --output_dir results/stage1_desc_only_all_both_best_logbarrier_centx5_fixedpatch/test_preds_desc_all_both_best_logbarrier_centx5_fixedpatch \
  --device cuda

echo "=== Stage-1 raw inference done Descriptor Only all classes, experiment with logbarrier (both patches) centroid focus==="
echo "dataset: fixed per-case sampling (1 STD + 1 EXP patch reused for entire training)"
