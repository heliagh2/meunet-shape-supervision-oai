#!/bin/bash
# AMOS22 one-off: preprocess to the canonical grid, then print the descriptor
# statistics the 30% runs will be constrained by (incl. the oracle check).
# CPU only. Run once, before any train_amos.sbatch.
#
#   cd /home/isimion/meunet-shape-supervision-oai/meunet_raw && mkdir -p logs
#   sbatch slurm/amos/prep_amos.sbatch
#
# PREP_DIR must be writable and must match images_dir/labels_dir in
# config/amos/*.yaml. ~300 CT cases -> roughly 4 GB.
#SBATCH --job-name=amos-prep
#SBATCH --account=gisr107588
#SBATCH --partition=genoa
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --chdir=/home/isimion/meunet-shape-supervision-oai/meunet_raw
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

AMOS_ROOT=/projects/0/prjs1400/AMOS/amos22
META_CSV=/projects/0/prjs1400/AMOS/labeled_data_meta_0000_0599.csv
PREP_DIR=/projects/0/prjs1400/AMOS/amos22_meunet_ct

module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
source ~/venvs/oai-baseline/bin/activate

# splits are committed in nnunet/nnUNet_preprocessed/Dataset702_AMOS_CT/ and are
# deterministic; pass --splits_dir to regenerate them.
python -u data/preprocess_amos.py \
  --amos_root "$AMOS_ROOT" \
  --meta_csv "$META_CSV" \
  --out_dir "$PREP_DIR" \
  --modality ct \
  --workers "${SLURM_CPUS_PER_TASK:-8}"

python -u tools/descriptor_stats.py \
  --config config/amos/AMOS_A2_1_distribution.yaml \
  --oracle \
  --out "$PREP_DIR/descriptor_stats_ann30_seed0.json"

echo "=== AMOS preprocessing + descriptor stats done: $PREP_DIR ==="
