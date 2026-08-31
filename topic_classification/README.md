# Exploratory Topic Modelling

This folder contains the separate, inductive BERTopic analysis of articles
already classified as political corruption. It is appendix material and a
sensitizing step for later article-level codebook development. Topic clusters
are not validated measures of victim visibility, corruption type, case
location, or accused actors.

## Scientific Role

The workflow asks two descriptive questions:

1. Which fine-grained themes recur in the final political-corruption corpus?
2. How are those themes distributed across countries and time?

The final input is the completed source-filtered classifier run. Source
screening follows the project rule established in Part 1: only country-source
pairs explicitly marked `No` in `conventional_journalism` are removed. Rows
marked `Yes`, `Review`, or without a decision are retained.

BERTopic discovers fine-grained clusters from a balanced country-year sample.
GPT-5.1 gives those clusters readable inductive labels. The researchers define
six broad interpretive groupings, after which GPT-5.1 assigns each discovered
cluster to exactly one group. These six groups are summaries of the inductive
solution, not six statistically estimated topics.

## Production Spine

| Step | Script | Output |
|---|---|---|
| 01 | `scripts/01_create_stratified_topic_sample.py` | Verified country-year sample and strata diagnostics |
| 02 | `scripts/02_fit_multilingual_bertopic.py` | Fine-grained BERTopic model and document assignments |
| 03 | `scripts/03_label_topics_with_llm.py` | GPT-5.1 labels, evidence, and audit records |
| 04 | `scripts/04_group_topics_with_llm.py` | Six-group assignments and audit record |
| 05 | `scripts/05_build_topic_visualizations.py` | Weighted country/time visualizations |
| 06 | `scripts/_impl/build_topic_tables.py` | Weighted analysis CSVs and LaTeX tables |
| 07 | `scripts/_impl/finalize_topic_outputs.py` | Final checks, generated values, and publication manifest |
| Publish | `scripts/06_publish_topic_archive_to_webdav.py` | Canonical manuscript files and full archive |

`scripts/07_rerun_final_source_filtered_topic_solution.sh` is the maintained
one-command entry point for steps 01-07. Do not assemble a production run from
old notebook cells or shell-history fragments.

`notebooks/01_inspect_topic_results.ipynb` is an optional interactive view. It
writes to `inspection_notebook_exports/` and cannot overwrite the production
tables.

## Required Upstream State

Political-classifier step 06 must have completed successfully. The topic
sampler requires and verifies all of these files:

```text
/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/
  political_corruption_pipeline/silver_classifier/
    classifier_run_manifest.json
    classified_country_summary.csv
    selected_threshold.txt
    classified_country_files/*_classified.csv.gz
```

The checks cover the classifier threshold, country set, row and positive
counts, source-decision provenance, file hashes, and the canonical
`cleaned_deduped_source_filtered` input directory. A stale or partial
classifier run stops the topic workflow.

## Final Specification

- Corpus: final source-filtered articles predicted as political corruption.
- Sampling: up to 200 articles per non-empty `country x year` stratum.
- Weights: inverse stratum sampling probabilities retained as
  `analysis_weight`.
- Embeddings: `intfloat/multilingual-e5-large`.
- BERTopic: `min_topic_size = 10`, `nr_topics = auto`, random seed 42.
- Fine-grained labels: GPT-5.1, temperature 0, with representative texts and
  keywords retained in the audit output.
- Interpretation: six documented higher-order groups; every non-outlier topic
  is assigned exactly once.

Counts are deliberately not hardcoded here. The finalizer derives them from
the current output and writes LaTeX macros and a latest-build index.

## Install

From the repository root:

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -r topic_classification/requirements-topic.txt
```

On `annecuda`, keep caches and temporary files on the large disk:

```bash
export TMPDIR=/home/akroon/data/1t_storage/tmp
export HF_HOME=/home/akroon/data/1t_storage/huggingface_cache
```

The UvA proxy currently permits GPT-5.1 for this project. GPT-5.6-terra was
tested but the project key was denied access, so it must not be selected for a
production run unless a fresh access test succeeds and the model change is
documented.

## Run Everything

After pulling the current repository on the server:

```bash
cd ~/RESPOND-victims-of-corruption

export DATA_ROOT=/home/akroon/data/1t_storage/RESPOND-victims-of-corruption
export TMPDIR=/home/akroon/data/1t_storage/tmp
export HF_HOME=/home/akroon/data/1t_storage/huggingface_cache
export CUDA_VISIBLE_DEVICES=1
export LLMPROXY_MODEL=gpt-5.1

mkdir -p "$DATA_ROOT/topic_classification/logs"

nohup bash topic_classification/scripts/07_rerun_final_source_filtered_topic_solution.sh \
  --upload-tables \
  --upload-archive \
  > "$DATA_ROOT/topic_classification/logs/final_topic_model.log" 2>&1 &

echo $! > "$DATA_ROOT/topic_classification/logs/final_topic_model.pid"
```

This is a deliberate clean rebuild. The sample and known outputs in the final
BERTopic directory are overwritten only by the numbered production scripts.
Human annotations and political-classifier outputs are not modified.

Monitor it with:

```bash
export DATA_ROOT=/home/akroon/data/1t_storage/RESPOND-victims-of-corruption
tail -f "$DATA_ROOT/topic_classification/logs/final_topic_model.log"

PID=$(cat "$DATA_ROOT/topic_classification/logs/final_topic_model.pid")
ps -p "$PID" -o pid,etime,%cpu,%mem,cmd
```

Successful completion ends with `Done.` and creates
`topic_model_output_manifest.json`.

## Resume Only GPT Labelling

Step 03 checkpoints after every topic. If only the LLM-label step fails, resume
it without `--overwrite`:

```bash
export TOPIC_DIR=/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/topic_classification/bertopic_political_corruption_source_filtered_200_min10

python3 topic_classification/scripts/03_label_topics_with_llm.py \
  --bertopic-dir "$TOPIC_DIR" \
  --model gpt-5.1
```

Then rerun steps 04-07, or restart the complete runner if a fully clean rebuild
is preferred. Do not use `--overwrite` when resuming a valid label checkpoint.

## Local Outputs

```text
/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/topic_classification/
  political_corruption_source_filtered_country_year_sample_200.csv.gz
  political_corruption_source_filtered_country_year_sample_200_strata.csv
  political_corruption_source_filtered_country_year_sample_200_run_manifest.json
  bertopic_political_corruption_source_filtered_200_min10/
    topic_info.csv
    document_topics.csv.gz
    topic_model/
    run_manifest.json
    topic_labels_llm.csv
    topic_labels_llm_audit.jsonl
    topic_labels_run_manifest.json
    topic_groups_llm.csv
    topic_groups_llm_audit.json
    topic_group_summaries_llm.csv
    topic_groups_run_manifest.json
    visualizations/
    inspection_tables/
    topic_model_build_summary.json
    topic_model_output_manifest.json
    00_LATEST_TOPIC_MODEL_BUILD.txt
```

Open `00_LATEST_TOPIC_MODEL_BUILD.txt` first. It records the current classifier
N and threshold, topic sample size, inlier/outlier counts, Git commits, hashes,
and exact manuscript paths.

## Manuscript And Archive Outputs

Canonical Research Drive manuscript files:

```text
ASCOR-FMG-5580-RESPOND-news-data (Projectfolder)/
  victims-of-corruption-paper/output/
    00_LATEST_TOPIC_MODEL_BUILD.txt
    topic_model_output_manifest.json
    tables/topic_models/
      topic_model_manuscript_values.tex
      table_topic_higher_order_summary.tex
```

The manuscript includes only the compact higher-order summary. The complete
fine-grained inventory is retained in the reproducibility archive:

```text
ASCOR-FMG-5580-RESPOND-news-data (Projectfolder)/
  victims-of-corruption-paper/derived_data/topic_classification/
```

Use `--upload-full-inventory` with the publisher only when the long inventory
is explicitly wanted in the manuscript table folder.

## Manual Commands

The one-command runner is preferred. Individual scripts expose `--help` for
development and diagnosis. The final order is fixed: sample, model, labels,
groups, visualizations, inspection tables, finalization, publication. Each
stage records hashes and refuses inputs from a different upstream run.

## Interpretation Rules

- Report weighted distributions because the sample is stratified.
- Treat BERTopic topic IDs as run-specific and unstable.
- Keep outliers visible in diagnostics but outside substantive group shares.
- Describe GPT labels and assignments as LLM-assisted interpretation.
- Do not present topic clusters or the six groups as validated corruption-type
  classifications.
- Use the archived prompts, examples, raw responses, model name, and manifests
  when auditing a published table.
