#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

PIPELINE_DIR=${PIPELINE_DIR:-/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline}
CLASSIFIER_DIR=${CLASSIFIER_DIR:-$PIPELINE_DIR/silver_classifier}
CLASSIFIED_DIR=${CLASSIFIED_DIR:-$CLASSIFIER_DIR/classified_country_files}
CONTENT_OUTPUT_DIR=${CONTENT_OUTPUT_DIR:-/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/content_classification/final_gpt51}
CONTENT_MODEL=${CONTENT_MODEL:-gpt-5.1}
CONTENT_MAX_CHARS=${CONTENT_MAX_CHARS:-6000}
CONTENT_SAVE_EVERY=${CONTENT_SAVE_EVERY:-25}

mkdir -p "$CONTENT_OUTPUT_DIR"
cd "$REPO_ROOT"

python3 -u content-classification/scripts/00_verify_final_corpus.py \
  --classifier-output-dir "$CLASSIFIER_DIR" \
  --classified-dir "$CLASSIFIED_DIR"

run_classifier() {
  local variable=$1
  echo
  echo "[$(date -Is)] START: $variable"
  python3 -u content-classification/scripts/classify_content.py \
    --variable "$variable" \
    --source classified \
    --classified-dir "$CLASSIFIED_DIR" \
    --classifier-output-dir "$CLASSIFIER_DIR" \
    --output-dir "$CONTENT_OUTPUT_DIR" \
    --model "$CONTENT_MODEL" \
    --max-chars "$CONTENT_MAX_CHARS" \
    --min-words 1 \
    --save-every "$CONTENT_SAVE_EVERY" \
    --retry-errors
  echo "[$(date -Is)] FINISHED: $variable"
}

run_classifier victim_visibility
run_classifier corruption_frame
run_classifier case_location
run_classifier accused_actor_visibility

python3 -u content-classification/scripts/merge_content_labels.py \
  --input-dir "$CONTENT_OUTPUT_DIR"

echo
echo "[$(date -Is)] CONTENT CLASSIFICATION COMPLETE"
echo "Output: $CONTENT_OUTPUT_DIR/political_corruption_content_categories_final.csv.gz"
