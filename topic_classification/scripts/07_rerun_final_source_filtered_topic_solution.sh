#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

UPLOAD_TABLES=0
for arg in "$@"; do
  case "$arg" in
    --upload-tables)
      UPLOAD_TABLES=1
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      echo "Usage: $0 [--upload-tables]" >&2
      exit 2
      ;;
  esac
done

DATA_ROOT="${DATA_ROOT:-/home/akroon/data/1t_storage/RESPOND-victims-of-corruption}"
PIPELINE_DIR="${PIPELINE_DIR:-$DATA_ROOT/political_corruption_pipeline}"
TOPIC_ROOT="${TOPIC_ROOT:-$DATA_ROOT/topic_classification}"
CLASSIFIED_DIR="${CLASSIFIED_DIR:-$PIPELINE_DIR/silver_classifier/classified_country_files_source_filtered}"

SAMPLE_PATH="${SAMPLE_PATH:-$TOPIC_ROOT/political_corruption_source_filtered_country_year_sample_200.csv.gz}"
BERTOPIC_DIR="${BERTOPIC_DIR:-$TOPIC_ROOT/bertopic_political_corruption_source_filtered_200_min10}"
MODEL="${LLMPROXY_MODEL:-gpt-5.1}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-1}"

mkdir -p "$TOPIC_ROOT" "$BERTOPIC_DIR"

echo "Project root: $PROJECT_ROOT"
echo "Source-filtered classified directory: $CLASSIFIED_DIR"
echo "Sample: $SAMPLE_PATH"
echo "BERTopic output: $BERTOPIC_DIR"
echo

echo "1/5 Creating source-filtered political-corruption topic sample..."
python3 topic_classification/scripts/01_create_stratified_topic_sample.py \
  --source classified \
  --classified-dir "$CLASSIFIED_DIR" \
  --political-only \
  --per-country-year 200 \
  --output-name "$(basename "$SAMPLE_PATH")" \
  --output-dir "$(dirname "$SAMPLE_PATH")"

echo
echo "2/5 Fitting multilingual BERTopic final specification..."
TMPDIR="${TMPDIR:-/home/akroon/data/1t_storage/tmp}" \
HF_HOME="${HF_HOME:-/home/akroon/data/1t_storage/huggingface_cache}" \
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/home/akroon/data/1t_storage/huggingface_cache}" \
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" \
python3 topic_classification/scripts/02_fit_multilingual_bertopic.py \
  --sample "$SAMPLE_PATH" \
  --output-dir "$BERTOPIC_DIR" \
  --min-topic-size 10 \
  --nr-topics auto

echo
echo "3/5 Labelling fine-grained BERTopic topics with $MODEL..."
python3 topic_classification/scripts/03_label_topics_with_llm.py \
  --bertopic-dir "$BERTOPIC_DIR" \
  --model "$MODEL"

echo
echo "4/5 Grouping fine-grained topics into higher-order topics with $MODEL..."
python3 topic_classification/scripts/04_group_topics_with_llm.py \
  --bertopic-dir "$BERTOPIC_DIR" \
  --model "$MODEL" \
  --target-groups 12

echo
echo "5/5 Executing inspection notebook to regenerate CSV and LaTeX tables..."
mkdir -p "$BERTOPIC_DIR/inspection_notebooks"
if command -v jupyter >/dev/null 2>&1; then
  TOPIC_DIR="$BERTOPIC_DIR" jupyter nbconvert \
    --to notebook \
    --execute topic_classification/notebooks/01_inspect_topic_results.ipynb \
    --output 01_inspect_topic_results.executed.ipynb \
    --output-dir "$BERTOPIC_DIR/inspection_notebooks"
else
  TOPIC_DIR="$BERTOPIC_DIR" python3 -m jupyter nbconvert \
    --to notebook \
    --execute topic_classification/notebooks/01_inspect_topic_results.ipynb \
    --output 01_inspect_topic_results.executed.ipynb \
    --output-dir "$BERTOPIC_DIR/inspection_notebooks"
fi

echo
echo "Regenerated final tables:"
ls -lh "$BERTOPIC_DIR/inspection_tables/latex"/table_*.tex

if [[ "$UPLOAD_TABLES" -eq 1 ]]; then
  echo
  echo "Uploading LaTeX tables to Research Drive..."
  python3 topic_classification/scripts/06_publish_topic_archive_to_webdav.py \
    --bertopic-dir "$BERTOPIC_DIR" \
    --sample "$SAMPLE_PATH" \
    --upload-latex-tables \
    --tables-only
fi

echo
echo "Done."
echo "Final source-filtered topic directory: $BERTOPIC_DIR"
