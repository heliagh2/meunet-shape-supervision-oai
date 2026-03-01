#!/bin/bash
#SBATCH --job-name=meu-s1-test-desc-only-all-both-esp100
#SBATCH --account=gisr107588
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --chdir=/projects/prjs1722/oai_baselines/meunet_raw
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

echo "=== Stage-1 raw inference start Descriptor Only all classes for EXP + STD patches with ESP 100==="
echo "whoami=$(whoami) host=$(hostname) date=$(date)"

module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

source ~/venvs/oai-baseline/bin/activate

python eval/infer_stage1_oai.py \
  --config configs/oai_stage1_desc_only_all_both_ESP100.yaml \
  --stage1_ckpt /projects/prjs1722/oai_baselines/meunet_raw/results/stage1_desc_only_all_both_ESP100/checkpoint_best.pt \
  --images_dir /projects/prjs1722/oai_baselines/nnunet/nnUNet_raw/Dataset701_OAIKnee/imagesTs \
  --output_dir results/stage1_desc_only_all_both_ESP100/test_preds_desc_all_both_ESP100 \
  --device cuda

echo "=== Stage-1 raw inference done Descriptor Only all classes for EXP + STD patches with ESP 100==="
