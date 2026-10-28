#!/bin/bash
# Descriptor statistics for one AMOS config, no training: the annotated-case bank
# (per class, STD and EXP patch), the distribution-matcher targets + oracle check
# (A2.1 configs), and the barrier_t needed to keep every per-case volume /
# centroid / spread constraint out of the log-barrier dead zone (t > 1/sqrt(band)).
#
#   sbatch slurm/amos/stats_amos.sbatch config/amos/AMOS_A1_11_concentration.yaml
#
# Output: logs/stats_amos-<job>.out, and the full numbers as JSON in
# <preprocessed dir>/descriptor_stats_<config name>.json
#SBATCH --job-name=amos-stats
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --chdir=/home/isimion1/meunet-shape-supervision-oai/meunet_raw
#SBATCH --output=logs/stats_amos-%j.out
#SBATCH --error=logs/stats_amos-%j.err

set -euo pipefail

CONFIG=${1:?usage: sbatch slurm/amos/stats_amos.sbatch config/amos/<name>.yaml}
PREP_DIR=/projects/0/prjs1400/AMOS/amos22_meunet_ct

module purge
source ~/envs/medshape/bin/activate

python -u tools/descriptor_stats.py \
  --config "$CONFIG" \
  --oracle \
  --out "$PREP_DIR/descriptor_stats_$(basename "${CONFIG%.yaml}").json"
