#!/usr/bin/env bash
# FINAL SUBMISSION PIPELINE — one entry point, three stages, fully offline.
#
#   stage 1  score    27B 4bit + LoRA adapter, 24 teacher-forced candidates per video
#   stage 2  dhash    64-bit frame hashes of the test split (CPU, ~1 min)
#   stage 3  nd2      near-dup post-processing -> submission.csv
#
# Measured on 1x RTX 3090 24GB (40-sample dry run, offline, this exact code path):
# 20 min for 40 including a ~5 min model load = 31.1 s/sample, peak VRAM 22837 MB / 24576.
# The load is a fixed cost, so 819 samples extrapolate to <= 7.1 h against the 24 h limit.
# Stage 1 is the only GPU stage and is resumable: if SCORES already exists it is reused,
# so a crash in stage 2/3 costs seconds, not hours.
#
# Usage:
#   ./run_inference.sh                                  # defaults below
#   ./run_inference.sh --test-csv /path/test.csv --image-root /path/test --out sub.csv
#   ./run_inference.sh --limit 8                        # quick end-to-end check
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TEST_CSV=""
IMAGE_ROOT=""
OUT="$HERE/submission.csv"
SCORES="$HERE/work/scores.json"
TEST_DHASH="$HERE/work/test_dhash.npz"
LIMIT=""
PY="${PYTHON:-python}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

while [ $# -gt 0 ]; do
  case "$1" in
    --test-csv)    TEST_CSV="$2"; shift 2;;
    --image-root)  IMAGE_ROOT="$2"; shift 2;;
    --out)         OUT="$2"; shift 2;;
    --scores)      SCORES="$2"; shift 2;;
    --limit)       LIMIT="$2"; shift 2;;
    --python)      PY="$2"; shift 2;;
    -h|--help)     sed -n '2,20p' "$0"; exit 0;;
    *) echo "unknown option: $1" >&2; exit 2;;
  esac
done

[ -n "$TEST_CSV" ]   || { echo "ERROR: --test-csv is required (columns Id,Input_1..Input_4)" >&2; exit 2; }
[ -n "$IMAGE_ROOT" ] || { echo "ERROR: --image-root is required (dir holding <Id>/<frame>)" >&2; exit 2; }
[ -e "$HERE/weights/base_model/config.json" ] || {
  echo "ERROR: base model missing at $HERE/weights/base_model" >&2
  echo "       see weights/BASE_MODEL.md" >&2; exit 2; }

mkdir -p "$HERE/work"
export PYTHONPATH="$HERE/code${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="$GPU"

echo "=============================================================="
echo " frame-ordering final inference"
echo "   test csv    : $TEST_CSV"
echo "   image root  : $IMAGE_ROOT"
echo "   GPU         : $CUDA_VISIBLE_DEVICES"
echo "   output      : $OUT"
echo "=============================================================="

# ---------------- stage 1: GPU scoring ----------------
if [ -s "$SCORES" ]; then
  echo "[1/3] scores already present at $SCORES -- reusing (delete it to force a rescore)"
else
  echo "[1/3] scoring 24 candidates per video (the long stage)"
  T0=$(date +%s)
  # shellcheck disable=SC2086
  "$PY" "$HERE/code/score_frames.py" \
    --model-dir "$HERE/weights/base_model" \
    --adapter   "$HERE/weights/adapter" \
    --train-csv "$TEST_CSV" \
    --image-root "$IMAGE_ROOT" \
    --load-4bit --prompt-a0 \
    --max-pixels 262144 --candidate-batch 8 \
    ${LIMIT:+--limit $LIMIT} \
    --output "$SCORES" || { echo "STAGE 1 FAILED" >&2; exit 1; }
  T1=$(date +%s)
  echo "[1/3] done in $(( (T1-T0)/60 )) min"
fi

# ---------------- stage 2: test dhash (CPU) ----------------
echo "[2/3] hashing test frames"
"$PY" "$HERE/code/build_dhash.py" \
  --csv "$TEST_CSV" --image-root "$IMAGE_ROOT" \
  --out "$TEST_DHASH" --key te_h || { echo "STAGE 2 FAILED" >&2; exit 1; }

# ---------------- stage 3: near-dup 2.0 -> submission ----------------
echo "[3/3] near-dup 2.0 post-processing"
"$PY" "$HERE/code/nd2_postprocess.py" \
  --scores "$SCORES" \
  --test-dhash "$TEST_DHASH" \
  --train-index "$HERE/assets/train_index_dhash.npz" \
  --train-labels "$HERE/assets/train_labels.csv" \
  --config "$HERE/assets/nd2_calib_best.json" \
  --also-pure "$HERE/work/submission_plain_argmax.csv" \
  --out "$OUT" || { echo "STAGE 3 FAILED" >&2; exit 1; }

echo
echo "SUBMISSION WRITTEN: $OUT"
"$PY" - "$OUT" <<'EOF'
import ast, csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
bad = [r["Id"] for r in rows if sorted(ast.literal_eval(r["Answer"])) != [1, 2, 3, 4]]
print(f"  rows: {len(rows)} | valid permutations: {len(rows)-len(bad)}/{len(rows)}")
if bad:
    print(f"  INVALID: {bad[:5]}"); sys.exit(1)
print("  all rows are valid permutations of [1,2,3,4]")
EOF
