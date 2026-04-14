#!/bin/bash
#SBATCH --job-name=meu-s1-test-desc-only-all-both-esp100-minSTD2-maxEXP-600ep
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

echo "=== Stage-1 raw inference start Descriptor Only all classes for EXP + STD patches with ESP 100 and with Minimised STD (200,200,200) patches and Maximised EXP (400,400,400) patches and 600 epochs==="
echo "whoami=$(whoami) host=$(hostname) date=$(date)"

module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

source ~/venvs/oai-baseline/bin/activate

python eval/infer_stage1_oai.py \
  --config config/oai_stage1_desc_only_all_both_ESP100_minSTD2_maxEXP_600ep.yaml \
  --stage1_ckpt /home/isimion/meunet-shape-supervision-oai/oai_baselines/meunet_raw/results/stage1_desc_only_all_both_ESP100_minSTD2_maxEXP_600ep/checkpoint_best.pt \
  --images_dir /ivi/zfs/s0/original_homes/isimion/knee_dataset/OAI_ZIB_DESS \
  --output_dir results/stage1_desc_only_all_both_ESP100_minSTD2_maxEXP_600ep/test_preds_desc_all_both_ESP100_minSTD2_maxEXP_600ep \
  --device cuda

echo "=== Stage-1 raw inference done Descriptor Only all classes for EXP + STD patches with ESP 100 and Minimised STD (200,200,200) patches and Maximised EXP (400,400,400) patches and 600 epochs==="
