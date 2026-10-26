#!/bin/bash
# One-off check of the grid -> native mapping used by eval_amos.sbatch.
# NO model involved: takes the preprocessed GROUND TRUTH of the test cases, maps it
# back to each scan's original grid, and scores it against the original AMOS
# labels. Large organs (liver, spleen, kidneys, stomach) should be ~0.95+; small
# ones (adrenals, esophagus) lower because of the coarser grid -- that is the
# resampling ceiling for native-space Dice, not a bug.
#
#   sbatch slurm/amos/sanity_amos.sbatch config/amos/AMOS_A1_11_concentration.yaml
#
# Any AMOS config works (only its data paths/splits are read). Result also saved
# as <workdir>/sanity_gt_roundtrip_dice.csv
#SBATCH --job-name=amos-sanity
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --chdir=/home/isimion1/meunet-shape-supervision-oai/meunet_raw
#SBATCH --output=logs/sanity_amos-%j.out
#SBATCH --error=logs/sanity_amos-%j.err

set -euo pipefail

CONFIG=${1:?usage: sbatch slurm/amos/sanity_amos.sbatch config/amos/<name>.yaml}
AMOS_ROOT=/projects/0/prjs1400/AMOS/amos22
LABELS="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"

module purge
source ~/envs/medshape/bin/activate

read -r WORKDIR LABELS_DIR SPLITS_DIR < <(python - "$CONFIG" <<'EOF'
import sys, yaml
from pathlib import Path
c = yaml.safe_load(open(sys.argv[1]))
print(c["workdir"], c["labels_dir"], Path(c["splits_file"]).parent)
EOF
)
TEST_JSON="$SPLITS_DIR/splits_test_final.json"
SAN=$(mktemp -d)
trap 'rm -rf "$SAN"' EXIT
mkdir -p "$WORKDIR"

echo "=== SANITY: preprocessed GT -> native vs original labels (no model) ==="
python - "$TEST_JSON" "$LABELS_DIR" "$SAN/gt_grid" <<'EOF'
import sys, json, shutil
from pathlib import Path
ids = json.load(open(sys.argv[1]))["test"]
out = Path(sys.argv[3]); out.mkdir(parents=True)
for s in ids:
    shutil.copy(Path(sys.argv[2]) / f"{s}.nii.gz", out / f"{s}.nii.gz")
print(f"copied {len(ids)} preprocessed GT label maps")
EOF
python -u eval/amos_to_native.py --pred_dir "$SAN/gt_grid" --amos_root "$AMOS_ROOT" --out_dir "$SAN/gt_native" > /dev/null
python -u eval/eval_dice_oai.py --gt_dir "$AMOS_ROOT/labelsVa" --pred_dir "$SAN/gt_native" \
  --labels $LABELS --cases_json "$TEST_JSON" --out_csv "$WORKDIR/sanity_gt_roundtrip_dice.csv"

echo "=== Done SANITY ==="
