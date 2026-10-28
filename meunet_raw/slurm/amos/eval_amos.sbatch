#!/bin/bash
# Evaluate one trained AMOS22 run on the held-out test set (official imagesVa CT,
# 100 cases, listed in Dataset702_AMOS_CT/splits_test_final.json), with BOTH
# output heads of MEUNet3D by default:
#
#   std  DEFAULT head, the one used until now: logit1 (head1, full resolution) on
#        the whole scan. Same head as validation Dice / checkpoint_best.pt.
#   exp  OTHER head: logit2 (head2, half resolution) on the scan resized like the
#        EXP training patch, logits upsampled back to the scan grid.
#
#   sbatch -J eval-A2.1 slurm/amos/eval_amos.sbatch config/amos/AMOS_A2_1_distribution.yaml
#   sbatch -J eval-A2.1-budget slurm/amos/eval_amos.sbatch config/amos/AMOS_A2_1_distribution.yaml checkpoint_budget_best.pt
#   sbatch -J eval-A1.11-std slurm/amos/eval_amos.sbatch config/amos/AMOS_A1_11_concentration.yaml checkpoint_best.pt std
#
# 3rd arg HEADS restricts to one head ("std" or "exp"); default "std exp".
#
# Steps, per head:
#   1. whole-volume inference on the canonical grid (argmax)
#   2. Dice on the canonical grid       -> <workdir>/test_<ckpt>[_exp]_dice_grid.csv
#   3. map predictions back to each scan's original grid
#   4. Dice vs the ORIGINAL AMOS labels -> <workdir>/test_<ckpt>[_exp]_dice_native.csv
#      (native is the number to report; grid is the number comparable to val Dice)
# std outputs keep their old names (no suffix); exp outputs get _exp. At the end a
# per-class native Dice table compares the heads side by side.
#
# The grid -> native mapping is checked separately, once: slurm/amos/sanity_amos.sbatch
#SBATCH --job-name=amos-eval
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --chdir=/home/isimion1/meunet-shape-supervision-oai/meunet_raw
#SBATCH --output=logs/eval_amos-%j.out
#SBATCH --error=logs/eval_amos-%j.err

set -euo pipefail

CONFIG=${1:?usage: sbatch slurm/amos/eval_amos.sbatch config/amos/<name>.yaml [checkpoint_best.pt] [\"std exp\"|std|exp]}
CKPT_NAME=${2:-checkpoint_best.pt}
HEADS=${3:-std exp}
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
echo "config=$CONFIG ckpt=$WORKDIR/$CKPT_NAME heads=$HEADS test=$TEST_JSON"

head_desc() {
  case "$1" in
    std) echo "std head (default, used until now: logit1, full res)" ;;
    exp) echo "exp head (other head: logit2, half res, EXP-style input)" ;;
    *)   echo "unknown head '$1' (expected std or exp)" >&2; exit 1 ;;
  esac
}
tag_of() {
  local t="test_${CKPT_NAME%.pt}"
  if [ "$1" = std ]; then echo "$t"; else echo "${t}_$1"; fi
}

for HEAD in $HEADS; do
  DESC=$(head_desc "$HEAD")
  TAG=$(tag_of "$HEAD")
  PRED_GRID="$WORKDIR/${TAG}_preds_grid"
  PRED_NATIVE="$WORKDIR/${TAG}_preds_native"
  echo
  echo "################ $DESC ################"

  echo "=== [$HEAD] 1. inference ($CKPT_NAME) ==="
  python -u eval/infer_stage1_oai.py \
    --config "$CONFIG" \
    --stage1_ckpt "$WORKDIR/$CKPT_NAME" \
    --images_dir "$IMAGES_DIR" \
    --output_dir "$PRED_GRID" \
    --head "$HEAD" \
    --device cuda
  # add --amp above if whole-volume inference runs out of memory

  echo "=== [$HEAD] 2. Dice on the canonical grid ==="
  python -u eval/eval_dice_oai.py --gt_dir "$LABELS_DIR" --pred_dir "$PRED_GRID" \
    --labels $LABELS --cases_json "$TEST_JSON" --out_csv "$WORKDIR/${TAG}_dice_grid.csv"

  echo "=== [$HEAD] 3. predictions -> original scan grids ==="
  python -u eval/amos_to_native.py --pred_dir "$PRED_GRID" --amos_root "$AMOS_ROOT" --out_dir "$PRED_NATIVE"

  echo "=== [$HEAD] 4. Dice vs original AMOS labels (report this) ==="
  python -u eval/eval_dice_oai.py --gt_dir "$AMOS_ROOT/labelsVa" --pred_dir "$PRED_NATIVE" \
    --labels $LABELS --cases_json "$TEST_JSON" --out_csv "$WORKDIR/${TAG}_dice_native.csv"
done

echo
echo "################ native Dice per head ($CKPT_NAME) ################"
CSVS=()
for HEAD in $HEADS; do CSVS+=("$HEAD=$WORKDIR/$(tag_of "$HEAD")_dice_native.csv"); done
python - "$LABELS_DIR/../dataset.json" "${CSVS[@]}" <<'EOF'
import csv, json, os, sys
import numpy as np
names = {}
if os.path.exists(sys.argv[1]):
    names = {int(k): v for k, v in json.load(open(sys.argv[1])).get("labels", {}).items()}
heads, cols = [], {}
for arg in sys.argv[2:]:
    h, path = arg.split("=", 1)
    rows = list(csv.DictReader(open(path)))
    keys = [k for k in rows[0] if k.startswith("dice_c")]
    cols[h] = {int(k[6:]): np.nanmean([float(r[k]) if r[k] != "" else np.nan for r in rows]) for k in keys}
    heads.append(h)
label = {"std": "std (default)", "exp": "exp (other)"}
print(f"{'class':<22}" + "".join(f"{label.get(h, h):>15}" for h in heads))
for c in sorted(cols[heads[0]]):
    name = f"{c}:{names.get(c, '?')}"
    print(f"{name:<22}" + "".join(f"{cols[h][c]:>15.3f}" for h in heads))
print(f"{'meanFGDice':<22}" + "".join(f"{np.nanmean(list(cols[h].values())):>15.3f}" for h in heads))
EOF

echo "=== Done AMOS eval: $CONFIG ($CKPT_NAME, heads: $HEADS) ==="
