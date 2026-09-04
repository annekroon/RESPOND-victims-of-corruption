#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

DATA_ROOT="${DATA_ROOT:-/home/akroon/data/1t_storage/RESPOND-victims-of-corruption}"
PIPELINE_DIR="${PIPELINE_DIR:-$DATA_ROOT/political_corruption_pipeline}"
TOPIC_ROOT="${TOPIC_ROOT:-$DATA_ROOT/topic_classification}"
CLASSIFIED_DIR="${CLASSIFIED_DIR:-$PIPELINE_DIR/silver_classifier/classified_country_files}"
MODEL="${LLMPROXY_MODEL:-gpt-5.1}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-1}"
PER_COUNTRY_YEAR="${PER_COUNTRY_YEAR:-50}"
TARGET_TOPICS="${TARGET_TOPICS:-none}"
MIN_TOPIC_SIZE="${MIN_TOPIC_SIZE:-40}"
HDBSCAN_MIN_SAMPLES="${HDBSCAN_MIN_SAMPLES:-10}"
CLUSTERER="${CLUSTERER:-hdbscan-stability}"
CLUSTER_SELECTION_METHOD="${CLUSTER_SELECTION_METHOD:-leaf}"

if [[ "$CLUSTERER" == "hdbscan-stability" ]]; then
  CLUSTER_SPEC="hdbscan_stability_v1"
elif [[ "$CLUSTERER" == "hdbscan" ]]; then
  CLUSTER_SPEC="${CLUSTERER}_${CLUSTER_SELECTION_METHOD}_${TARGET_TOPICS}_min${MIN_TOPIC_SIZE}_ms${HDBSCAN_MIN_SAMPLES}"
else
  CLUSTER_SPEC="${CLUSTERER}_${TARGET_TOPICS}"
fi

SAMPLE_PATH="${SAMPLE_PATH:-$TOPIC_ROOT/political_corruption_descriptive_country_year_sample_${PER_COUNTRY_YEAR}.csv.gz}"
ABSTRACTION_PATH="${ABSTRACTION_PATH:-$TOPIC_ROOT/political_corruption_descriptive_country_year_sample_${PER_COUNTRY_YEAR}_english_abstracts.csv.gz}"
BERTOPIC_DIR="${BERTOPIC_DIR:-$TOPIC_ROOT/bertopic_political_corruption_descriptive_${CLUSTER_SPEC}_sample_${PER_COUNTRY_YEAR}}"
ABSTRACTION_COMPLETE="$ABSTRACTION_PATH.complete.json"

mkdir -p "$TOPIC_ROOT"

echo "Stability-selected descriptive BERTopic model"
echo "=============================================="
echo "Classifier inputs: $CLASSIFIED_DIR"
echo "Country-year sample: $SAMPLE_PATH"
echo "English abstractions: $ABSTRACTION_PATH"
echo "BERTopic output: $BERTOPIC_DIR"
echo "LLM: $MODEL"
echo "Maximum per country-year: $PER_COUNTRY_YEAR"
echo "Target topics: $TARGET_TOPICS"
echo "Minimum topic size: $MIN_TOPIC_SIZE"
echo "HDBSCAN minimum samples: $HDBSCAN_MIN_SAMPLES"
echo "Clusterer: $CLUSTERER"
echo "Cluster selection method: $CLUSTER_SELECTION_METHOD"
echo

if [[ ! -f "$SAMPLE_PATH" ]]; then
  echo "1/6 Creating a verified sample of up to $PER_COUNTRY_YEAR articles per country-year..."
  python3 topic_classification/scripts/01_create_stratified_topic_sample.py \
    --source classified \
    --classified-dir "$CLASSIFIED_DIR" \
    --classifier-output-dir "$PIPELINE_DIR/silver_classifier" \
    --political-only \
    --per-country-year "$PER_COUNTRY_YEAR" \
    --output-name "$(basename "$SAMPLE_PATH")" \
    --output-dir "$(dirname "$SAMPLE_PATH")" \
    --overwrite
else
  echo "1/6 Reusing existing verified country-year sample: $SAMPLE_PATH"
fi

echo
if [[ -f "$ABSTRACTION_COMPLETE" ]] && python3 - "$ABSTRACTION_PATH" <<'PY'
import sys
from pathlib import Path
from topic_classification.provenance import validate_verified_topic_sample

validate_verified_topic_sample(Path(sys.argv[1]))
PY
then
  echo "2/6 Reusing completed and verified abstractions: $ABSTRACTION_PATH"
else
  echo "2/6 Creating or resuming language-neutral English abstractions..."
  python3 -u topic_classification/scripts/02_create_descriptive_abstractions.py \
    --input "$SAMPLE_PATH" \
    --output "$ABSTRACTION_PATH" \
    --model "$MODEL" \
    --max-chars 4000 \
    --save-every 10 \
    --retry-errors
fi

echo
if [[ -f "$BERTOPIC_DIR/run_manifest.json" ]] && python3 - "$BERTOPIC_DIR" <<'PY'
import sys
from pathlib import Path
from topic_classification.provenance import validate_topic_model_outputs

validate_topic_model_outputs(Path(sys.argv[1]))
PY
then
  echo "3/6 Reusing verified descriptive BERTopic model: $BERTOPIC_DIR"
else
  echo "3/6 Selecting and fitting BERTopic to usable abstractions..."
  TMPDIR="${TMPDIR:-/home/akroon/data/1t_storage/tmp}" \
  HF_HOME="${HF_HOME:-/home/akroon/data/1t_storage/huggingface_cache}" \
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" \
  python3 topic_classification/scripts/03_fit_descriptive_bertopic.py \
    --sample "$ABSTRACTION_PATH" \
    --output-dir "$BERTOPIC_DIR" \
    --text-column topic_description_english \
    --status-column topic_abstraction_status \
    --include-status usable \
    --clusterer "$CLUSTERER" \
    --cluster-selection-method "$CLUSTER_SELECTION_METHOD" \
    --min-topic-size "$MIN_TOPIC_SIZE" \
    --hdbscan-min-samples "$HDBSCAN_MIN_SAMPLES" \
    --nr-topics "$TARGET_TOPICS" \
    --overwrite
fi

echo
if [[ -f "$BERTOPIC_DIR/topic_labels_run_manifest.json" ]] && python3 - "$BERTOPIC_DIR" <<'PY'
import sys
from pathlib import Path
from topic_classification.provenance import validate_topic_label_outputs

validate_topic_label_outputs(Path(sys.argv[1]))
PY
then
  echo "4/6 Reusing verified topic labels: $BERTOPIC_DIR/topic_labels_llm.csv"
else
  echo "4/6 Labelling the compact descriptive topics with $MODEL..."
  python3 topic_classification/scripts/04_label_descriptive_topics_with_llm.py \
    --bertopic-dir "$BERTOPIC_DIR" \
    --text-column topic_description_english \
    --examples-per-topic 10 \
    --model "$MODEL" \
    --overwrite
fi

echo
echo "5/6 Building direct-topic visualizations..."
python3 topic_classification/scripts/05_build_topic_visualizations.py \
  --bertopic-dir "$BERTOPIC_DIR" \
  --analysis-level fine-grained \
  --top-n 12

echo
echo "6/6 Building the direct topic table, diagnostics, and manual-review packet..."
python3 topic_classification/scripts/06_build_descriptive_topic_review.py \
  --bertopic-dir "$BERTOPIC_DIR" \
  --examples-per-topic 10

echo
echo "Candidate build complete. No manuscript files were uploaded automatically."
echo "Inspect this file first:"
echo "$BERTOPIC_DIR/00_LATEST_DESCRIPTIVE_TOPIC_BUILD.txt"
echo "Then review:"
echo "$BERTOPIC_DIR/descriptive_outputs/descriptive_topic_manual_review.csv"
