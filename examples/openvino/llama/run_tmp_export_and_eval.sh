#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="/home/user/Documents/nncf/src/"

LLAMA_CHECKPOINT=/home/user/Documents/executorch/Llama-3-8B/original/consolidated.00.pth
LLAMA_PARAMS=/home/user/Documents/executorch/Llama-3-8B/original/params.json
LLAMA_TOKENIZER=/home/user/Documents/executorch/Llama-3-8B/original/tokenizer.model

# TASKS="lambada_openai"
TASKS="wikitext"
SEQ_LENGTH="2048"
LIMIT=1000
CALIBRATION_DATA="Once upon a time"

TMP_EXPORT="$(mktemp --suffix=.pt2 /tmp/ov_llama_compressed_XXXXXX)"
cleanup() {
  rm -f "$TMP_EXPORT"
}
trap cleanup EXIT

echo "[1/2] Exporting compressed model to temporary file: $TMP_EXPORT"
ET_COMPRESSED_EXPORT_PATH="$TMP_EXPORT" \
python -m executorch.extension.llm.export.export_llm \
  --config llama3_2_ov_4wo.yaml \
  +backend.openvino.device="CPU" \
  +base.model_class="llama3_2" \
  +base.checkpoint="$LLAMA_CHECKPOINT" \
  +base.params="$LLAMA_PARAMS" \
  +base.tokenizer_path="$LLAMA_TOKENIZER"

MODEL_BYTES="$(stat -c%s "$TMP_EXPORT")"
MODEL_MIB="$(awk "BEGIN {printf \"%.2f\", $MODEL_BYTES / 1024 / 1024}")"

echo ""
echo "============================================"
echo "  Exported model size: ${MODEL_MIB} MiB"
echo "  (${MODEL_BYTES} bytes)"
echo "============================================"
echo ""

export CUDA_VISIBLE_DEVICES=0,1
    # python eval_compressed_export.py
EVAL_CMD=(
    accelerate launch --multi_gpu --num_processes 2 eval_compressed_export.py
  --exported_program "$TMP_EXPORT"
  --tokenizer_path "$LLAMA_TOKENIZER"
  --tasks "$TASKS"
  --seq_length "$SEQ_LENGTH"
  --use_kv_cache
  --enable_dynamic_shape
  --generate_full_logits
  --limit "$LIMIT"
  --cuda
  # --split_model_across_gpus
)

echo "[2/2] Evaluating temporary exported program"
"${EVAL_CMD[@]}"

echo "Done. Temporary exported program removed: $TMP_EXPORT"
