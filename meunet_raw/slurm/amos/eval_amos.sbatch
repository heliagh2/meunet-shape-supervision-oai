#!/bin/bash
# Evaluate one trained AMOS22 run on the held-out test set (official imagesVa CT,
# 100 cases, listed in Dataset702_AMOS_CT/splits_test_final.json).
#
#   sbatch -J eval-A2.1 slurm/amos/eval_amos.sbatch config/amos/AMOS_A2_1_distribution.yaml
#   sbatch -J eval-A2.1-budget slurm/amos/eval_amos.sbatch config/amos/AMOS_A2_1_distribution.yaml checkpoint_budget_best.pt
#   SANITY=1 sbatch slurm/amos/eval_amos.sbatch config/amos/AMOS_A0_concentration.yaml   # + mapping check (run once)
#
# Steps:
#   1. whole-volume inference on the canonical grid (as for OAI: logit1, argmax)
#   2. Dice on the canonical grid       -> <workdir>/test_<ckpt>_dice_grid.csv
#   3. map predictions back to each scan's original grid
#   4. Dice vs the ORIGINAL AMOS labels -> <workdir>/test_<ckpt>_dice_native.csv
#      (native is the number to report; grid is the number comparable to val Dice)
#
# SANITY=1 first maps the preprocessed GROUND TRUTH back to native space and
# scores it against the original labels. Dice there should be ~0.95+ for large
# organs; anything much lower means the grid->native mapping is broken and
# step 4 cannot be trusted. Only needs doing once.
#SBATCH --job-name=amos-eval
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --chdir=/home/isimion1/meunet-shape-supervision-oai/meunet_raw
#SBATCH --output=logs/eval_amos-%j.out
#SBATCH --error=logs/eval_amos-%j.err

set -euo pipefail

CONFIG=${1:?usage: sbatch slurm/amos/eval_amos.sbatch config/amos/<name>.yaml [checkpoint_best.pt]}
CKPT_NAME=${2:-checkpoint_best.pt}
AMOS_ROOT=/projects/0/prjs1400/AMOS/amos22
LABELS="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"

module purge
source ~/envs/medshape/bin/activate

# everything else comes from the config, so eval can never point at a different
# grid / split than the run was trained on
read -r WORKDIR IMAGES_DIR LABELS_DIR SPLITS_DIR < <(python - "$CONFIG" <<'EOF'
import sys, yaml
from pathlib import Path
c = yaml.safe_load(open(sys.argv[1]))
print(c["workdir"], c["images_dir"], c["labels_dir"], Path(c["splits_file"]).parent)
EOF
)
TEST_JSON="$SPLITS_DIR/splits_test_final.json"
TAG="test_${CKPT_NAME%.pt}"
PRED_GRID="$WORKDIR/${TAG}_preds_grid"
PRED_NATIVE="$WORKDIR/${TAG}_preds_native"
echo "config=$CONFIG ckpt=$WORKDIR/$CKPT_NAME test=$TEST_JSON"

if [[ "${SANITY:-0}" == "1" ]]; then
  echo "=== SANITY: preprocessed GT -> native vs original labels ==="
  SAN=$(mktemp -d)
  python - "$TEST_JSON" "$LABELS_DIR" "$SAN/gt_grid" <<'EOF'
import sys, json, shutil
from pathlib import Path
ids = json.load(open(sys.argv[1]))["test"]
out = Path(sys.argv[3]); out.mkdir(parents=True)
for s in ids:
    shutil.copy(Path(sys.argv[2]) / f"{s}.nii.gz", out / f"{s}.nii.gz")
EOF
  python -u eval/amos_to_native.py --pred_dir "$SAN/gt_grid" --amos_root "$AMOS_ROOT" --out_dir "$SAN/gt_native" > /dev/null
  python -u eval/eval_dice_oai.py --gt_dir "$AMOS_ROOT/labelsVa" --pred_dir "$SAN/gt_native" \
    --labels $LABELS --cases_json "$TEST_JSON" --out_csv "$WORKDIR/sanity_gt_roundtrip_dice.csv"
  rm -rf "$SAN"
fi

echo "=== 1. inference ($CKPT_NAME) ==="
python -u eval/infer_stage1_oai.py \
  --config "$CONFIG" \
  --stage1_ckpt "$WORKDIR/$CKPT_NAME" \
  --images_dir "$IMAGES_DIR" \
  --output_dir "$PRED_GRID" \
  --device cuda
# add --amp above if whole-volume inference runs out of memory

echo "=== 2. Dice on the canonical grid ==="
python -u eval/eval_dice_oai.py --gt_dir "$LABELS_DIR" --pred_dir "$PRED_GRID" \
  --labels $LABELS --cases_json "$TEST_JSON" --out_csv "$WORKDIR/${TAG}_dice_grid.csv"

echo "=== 3. predictions -> original scan grids ==="
python -u eval/amos_to_native.py --pred_dir "$PRED_GRID" --amos_root "$AMOS_ROOT" --out_dir "$PRED_NATIVE"

echo "=== 4. Dice vs original AMOS labels (report this) ==="
python -u eval/eval_dice_oai.py --gt_dir "$AMOS_ROOT/labelsVa" --pred_dir "$PRED_NATIVE" \
  --labels $LABELS --cases_json "$TEST_JSON" --out_csv "$WORKDIR/${TAG}_dice_native.csv"

echo "=== Done AMOS eval: $CONFIG ($CKPT_NAME) ==="
