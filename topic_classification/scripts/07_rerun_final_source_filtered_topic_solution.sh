#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

UPLOAD_TABLES=0
UPLOAD_ARCHIVE=0
for arg in "$@"; do
  case "$arg" in
    --upload-tables)
      UPLOAD_TABLES=1
      ;;
    --upload-archive)
      UPLOAD_ARCHIVE=1
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      echo "Usage: $0 [--upload-tables] [--upload-archive]" >&2
      exit 2
      ;;
  esac
done

DATA_ROOT="${DATA_ROOT:-/home/akroon/data/1t_storage/RESPOND-victims-of-corruption}"
PIPELINE_DIR="${PIPELINE_DIR:-$DATA_ROOT/political_corruption_pipeline}"
TOPIC_ROOT="${TOPIC_ROOT:-$DATA_ROOT/topic_classification}"
CLASSIFIED_DIR="${CLASSIFIED_DIR:-$PIPELINE_DIR/silver_classifier/classified_country_files}"

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

echo "1/7 Creating source-filtered political-corruption topic sample..."
python3 topic_classification/scripts/01_create_stratified_topic_sample.py \
  --source classified \
  --classified-dir "$CLASSIFIED_DIR" \
  --classifier-output-dir "$PIPELINE_DIR/silver_classifier" \
  --political-only \
  --per-country-year 200 \
  --output-name "$(basename "$SAMPLE_PATH")" \
  --output-dir "$(dirname "$SAMPLE_PATH")" \
  --overwrite

echo
echo "2/7 Fitting multilingual BERTopic final specification..."
TMPDIR="${TMPDIR:-/home/akroon/data/1t_storage/tmp}" \
HF_HOME="${HF_HOME:-/home/akroon/data/1t_storage/huggingface_cache}" \
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" \
python3 topic_classification/scripts/02_fit_multilingual_bertopic.py \
  --sample "$SAMPLE_PATH" \
  --output-dir "$BERTOPIC_DIR" \
  --min-topic-size 10 \
  --nr-topics auto \
  --overwrite

echo
echo "3/7 Labelling fine-grained BERTopic topics with $MODEL..."
python3 topic_classification/scripts/03_label_topics_with_llm.py \
  --bertopic-dir "$BERTOPIC_DIR" \
  --model "$MODEL" \
  --overwrite

echo
echo "4/7 Grouping fine-grained topics into six higher-order topics with $MODEL..."
python3 topic_classification/scripts/04_group_topics_with_llm.py \
  --bertopic-dir "$BERTOPIC_DIR" \
  --model "$MODEL" \
  --target-groups 6 \
  --overwrite

echo
echo "5/7 Building higher-order topic visualizations..."
python3 topic_classification/scripts/05_build_topic_visualizations.py \
  --bertopic-dir "$BERTOPIC_DIR" \
  --analysis-level higher-order \
  --top-n 6

echo
echo "6/7 Building CSV and LaTeX analysis tables..."
python3 topic_classification/scripts/_impl/build_topic_tables.py \
  --bertopic-dir "$BERTOPIC_DIR"

echo
echo "7/7 Validating the complete chain and writing manuscript metadata..."
python3 topic_classification/scripts/_impl/finalize_topic_outputs.py \
  --bertopic-dir "$BERTOPIC_DIR"

echo
echo "Regenerated manuscript-facing topic files:"
ls -lh "$BERTOPIC_DIR/inspection_tables/latex"/table_*.tex
ls -lh "$BERTOPIC_DIR/inspection_tables/latex"/topic_model_manuscript_values.tex

if [[ "$UPLOAD_TABLES" -eq 1 ]]; then
  echo
  echo "Uploading LaTeX tables to Research Drive..."
  python3 topic_classification/scripts/06_publish_topic_archive_to_webdav.py \
    --bertopic-dir "$BERTOPIC_DIR" \
    --sample "$SAMPLE_PATH" \
    --upload-latex-tables \
    --tables-only
fi

if [[ "$UPLOAD_ARCHIVE" -eq 1 ]]; then
  echo
  echo "Creating and uploading the full reproducibility archive..."
  python3 topic_classification/scripts/06_publish_topic_archive_to_webdav.py \
    --bertopic-dir "$BERTOPIC_DIR" \
    --sample "$SAMPLE_PATH" \
    --upload
fi

echo
echo "Done."
echo "Final source-filtered topic directory: $BERTOPIC_DIR"
