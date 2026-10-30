#!/usr/bin/env bash
# Run AdaGround on one benchmark with one backbone, then compose the final answers and score them.
#
# usage: bash scripts/run_adaground.sh BENCHMARK MODEL_SIZE MODEL_PATH DATA_DIR OUTPUT_DIR [IDS_FILE]
#   BENCHMARK   sspro | osworldg | ssv2 | uivision
#   MODEL_SIZE  8b | 32b        selects the vLLM engine settings used in the paper
#   MODEL_PATH  Qwen3-VL-8B-Instruct or Qwen3-VL-32B-Instruct-FP8 (local directory or Hugging Face id)
#   DATA_DIR    benchmark root directory (see "Data Preparation" in the README)
#   OUTPUT_DIR  stage outputs, predictions.json and the score report are written here
#   IDS_FILE    optional JSON list of sample ids, to run a subset
# environment:
#   PYTHON        python interpreter (default: python)
#   GPU_MEM_UTIL  overrides the vLLM memory fraction (default 0.32 for 8b, 0.62 for 32b)
#
# Stages whose output already exists are skipped, so an interrupted run resumes where it stopped.
set -euo pipefail

if [ $# -lt 5 ]; then
  sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
  exit 1
fi
BENCHMARK=$1; MODEL_SIZE=$2; MODEL_PATH=$3; DATA_DIR=$4; OUT=$5; IDS_FILE=${6:-}
PYTHON=${PYTHON:-python}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

case $BENCHMARK in sspro|osworldg|ssv2|uivision) ;; *) echo "unknown benchmark: $BENCHMARK" >&2; exit 1;; esac
case $MODEL_SIZE in
  8b)  ENG="--gpu_mem_util ${GPU_MEM_UTIL:-0.32} --chunk 32 --max_num_seqs 16";;
  32b) ENG="--gpu_mem_util ${GPU_MEM_UTIL:-0.62} --chunk 16 --max_num_seqs 8";;
  *) echo "MODEL_SIZE must be 8b or 32b" >&2; exit 1;;
esac

mkdir -p "$OUT"
COMMON="--dataset $BENCHMARK --data_dir $DATA_DIR --model_path $MODEL_PATH --family qwen3 --factor 32 \
--max_pixels 8847360 --eta 8 --dmin 512 --dmax 1024 --mv_vside 1.0 --max_model_len 32768 --tp 1 $ENG"
[ -n "$IDS_FILE" ] && COMMON="$COMMON --ids_file $IDS_FILE"

stage() {  # stage NAME OUTPUT_FILE ARGS...
  local name=$1 done_file=$2; shift 2
  if [ -s "$OUT/$done_file" ]; then echo "[skip] $name ($done_file exists)"; return; fi
  echo "[run ] $name"
  (cd "$ROOT" && $PYTHON -m adaground.infer $COMMON "$@")
}

# 1. fixed 4x crop chain c_4x (its first pass is the single-pass baseline)
stage c4x c4x.json --output "$OUT/c4x.json" --mode zoom --b1 1.0 --b2 4.0 --crop_wh 1288,728
# 2. existence query e(S, I), used for abstention on OSWorld-G
if [ "$BENCHMARK" = osworldg ]; then
  stage exist exist.json --output "$OUT/exist.json" --mode exist --b1 1.0
fi
# 3. CSG: instruction program and header grounding for spreadsheet cells
stage csg csg_cell.json --output "$OUT/csg.json" --mode program --retry_from "$OUT/c4x.json" --retry_b3 9.0 \
  --retry_min_side 192
# 4. TNZ chain c_TNZ (target span tau = 3 tokens)
stage tnz tnz_t3.json --output "$OUT/tnz_t3.json" --mode tsn --tsn_patch 32 --tsn_tok_target 3 --tsn_ctx 16 \
  --tsn_err_tokens 4 --tsn_kmax 4 --tsn_loose 2 --tsn_depth 2 --auto_px 2200000 --ad_b1_min 0.25 --retry_vb 4.0 \
  --retry_min_side 192
# 5. verifier arbitration between c_TNZ and c_4x
stage arb arb_t3.json --output "$OUT/arb_t3.json" --mode choose2 --choose_from "$OUT/tnz_t3.json,$OUT/c4x.json" \
  --choose_method verify --b2 4.0 --mv_tau 14 --mv_k 3 --choose_tok sum --choose_default first
# 6. negative-feedback re-grounding (theta_r = 0.99)
stage rg rg_t3_tau0.99.json --output "$OUT/rg_t3.json" --mode retry --retry_from "$OUT/arb_t3.json" --b1 1.0 \
  --retry_b3 9.0 --retry_min_side 192 --retry_vb 4.0 --retry_taus 0.99

# 7. compose (CSG header rule, abstention on OSWorld-G) and score
EXTRA=""
[ "$BENCHMARK" = osworldg ] && EXTRA="--osworldg_labels $DATA_DIR/OSWorld-G_refined.json"
(cd "$ROOT" && $PYTHON -m adaground.compose --run_dir "$OUT" --benchmark "$BENCHMARK" $EXTRA \
  --out "$OUT/predictions.json") | tee "$OUT/results.txt"
