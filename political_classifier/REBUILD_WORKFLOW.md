# Political-Classifier Rebuild Workflow

This is the canonical, reproducible run order for Part 1: constructing the
source-eligible corruption-query corpus, training and validating the political-
corruption classifier, scoring the corpus, and producing manuscript outputs.

Run all commands from the repository root on `annecuda`.

## Analysis Contract

```text
NewsAPI corruption-keyword article retrieval
-> clean and exact-deduplicate within country
-> remove country-source pairs with conventional_journalism == No
-> draw a fresh country-balanced silver-training sample
-> GPT-5.1 silver labels
-> exclude all overlap with the human benchmark
-> threshold calibration on 40% of the eligible human benchmark
-> one held-out performance estimate on the remaining 60%
-> score every source-filtered country file
-> final political-corruption corpus
```

Relative attention uses a separate denominator:

```text
NewsAPI weekly total-news counts
-> aggregate the same weekly bins for numerator and denominator
-> political-corruption articles / all news articles
```

The numbered Python scripts are the production workflow. The optional
comparison notebook is a read-only inspection view and is never required to
create a production file.

## Before You Run

Create or activate the server environment, install the declared dependencies,
and save the exact installed versions with the run:

```bash
cd ~/RESPOND-victims-of-corruption
python3 -m pip install -r requirements.txt
python3 -m pip freeze > environment-lock.txt
```

`config_local.py` or environment variables must provide the Research Drive and
UvA LLM proxy credentials. Never commit that file or the credentials.

Use storage-backed cache/temp folders for model runs:

```bash
mkdir -p /home/akroon/data/1t_storage/tmp
mkdir -p /home/akroon/data/1t_storage/huggingface_cache
export TMPDIR=/home/akroon/data/1t_storage/tmp
export HF_HOME=/home/akroon/data/1t_storage/huggingface_cache
```

For a deliberate clean rebuild, first inspect and then execute the generated-
output reset. It preserves the reviewed source workbook, human annotations, UK
benchmark supplement, cleaned country files, and content-codebook samples:

```bash
python3 political_classifier/tools/reset_rebuild_outputs.py
python3 political_classifier/tools/reset_rebuild_outputs.py \
  --confirm DELETE_OLD_POLITICAL_CLASSIFIER_OUTPUTS
```

The first command is a dry run. Read its list before running the confirmed
deletion.

## Step 00: Download The Reviewed Source Workbook

This uses the WebDAV API, not the unreliable mounted WebDAV folder.

```bash
python3 political_classifier/scripts/00_download_source_workbook.py --overwrite
```

The workbook is used as an exclusion list. Only rows whose final
`conventional_journalism` value is exactly `No` are removed. `Yes`, unresolved,
blank, and unmatched country-source pairs are retained; unmatched pairs are
reported in the source-filter audit.

## Step 01: Clean And Exact-Deduplicate Within Country

Run this when the raw NewsAPI exports or cleaning rules change. It downloads
each country file to a local cache, processes it in chunks, and writes country
outputs atomically. Exact URI and normalized-text duplicates are removed within
country. Risky prefix-based near-deduplication is disabled in production.

```bash
nohup python3 -u political_classifier/scripts/01_clean_dedupe_data.py \
  --overwrite \
  > 01_clean_dedupe_data.log 2>&1 &
```

Monitor:

```bash
tail -f 01_clean_dedupe_data.log
```

Important outputs include:

```text
political_corruption_pipeline/{country}_cleaned_deduped.csv.gz
political_corruption_pipeline/all_countries_cleaned_deduped_minimal.csv.gz
political_corruption_pipeline/clean_dedupe_audit.csv
political_corruption_pipeline/denominator_country_*.csv
```

## Step 02: Build The Full Source-Filtered Corpus

```bash
python3 political_classifier/scripts/02_create_source_filtered_corpus.py \
  --overwrite
```

The ranked normalized domains without workbook decisions are written to
`source_filter_missing_sources.csv`. They are retained under the exclusion-list
policy, while every country-source pair explicitly marked `No` is removed.
Adding later reviewed decisions to the canonical workbook and rerunning step 02
updates the corpus deterministically.

Outputs:

```text
political_corruption_pipeline/cleaned_deduped_source_filtered/
  {country}_cleaned_deduped_source_filtered.csv.gz
  all_countries_cleaned_deduped_source_filtered_minimal.csv.gz
political_corruption_pipeline/source_inclusion/
  cleaned_source_filter_output_summary.csv
  cleaned_source_filter_decision_summary_by_country.csv
  source_filter_missing_sources.csv
```

## Step 03: Draw A Fresh Silver-Training Candidate Set

The script samples 500 articles per country from the source-filtered corpus. It
automatically excludes URI and normalized-text overlap with both parts of the
human benchmark and saves an overlap audit.

```bash
python3 political_classifier/scripts/03_prepare_classifier_training_sample.py \
  --overwrite \
  --country-targets Bulgaria:500,France:500,Hungary:500,Italy:500,Netherlands:500,Serbia:500,Sweden:500,Ukraine:500,United_Kingdom:500
```

## Step 04: Create Fresh GPT-5.1 Silver Labels

```bash
nohup python3 -u political_classifier/scripts/04_label_silver_batch.py \
  --input /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/active_learning/silver_training_source_filtered_for_annotation.csv \
  --output /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/active_learning/silver_training_source_filtered_with_llm_suggestions.csv \
  --model gpt-5.1 \
  --max-chars 3000 \
  --save-every 10 \
  --overwrite \
  > 04_label_silver_batch.log 2>&1 &
```

Monitor:

```bash
tail -f 04_label_silver_batch.log
```

The output has a run manifest and append-only JSONL audit. Invalid labels and
failed requests are not accepted as training labels. A successful run also
writes `silver_training_source_filtered_with_llm_suggestions.csv.complete.json`;
step 05 refuses a partial CSV without this checksum-verified marker.

If the job was interrupted, resume it without `--overwrite` and retry any saved
errors:

```bash
nohup python3 -u political_classifier/scripts/04_label_silver_batch.py \
  --input /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/active_learning/silver_training_source_filtered_for_annotation.csv \
  --output /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/active_learning/silver_training_source_filtered_with_llm_suggestions.csv \
  --model gpt-5.1 \
  --max-chars 3000 \
  --save-every 10 \
  --retry-errors \
  > 04_label_silver_batch.log 2>&1 &
```

## Step 05: Calibrate And Evaluate The Classifier

This is the statistical checkpoint. The pre-specified production model is
E5-large plus class-balanced logistic regression. The human benchmark is
source-filtered to match the production population, then split once by country
and class: 40% for threshold calibration and 60% for held-out reporting. A hard
assertion stops the run if any silver/human URI or text overlap remains.

```bash
python3 political_classifier/scripts/05_compare_models.py \
  --embedding-models intfloat/multilingual-e5-large \
  --threshold-calibration-fraction 0.40 \
  --batch-size 32 \
  --extra-human-validation /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/active_learning/uk_human_validation_reviewed.csv
```

Inspect these files before full scoring:

```text
classifier_comparison/classifier_comparison_run_manifest.json
classifier_comparison/human_benchmark_split.csv
classifier_comparison/selected_threshold.txt
classifier_comparison/best_model_results.csv
classifier_comparison/country_results_for_best_silver_thresholds.csv
classifier_comparison/all_threshold_results.csv
```

The threshold table is calibration-only. The selected silver-model row in
`best_model_results.csv` is held-out performance. Human five-fold CV rows are
diagnostics, not the selected model's held-out estimate.

## Step 06A: Refit And Validate The Final Model

First run step 06 without corpus scoring. It verifies that the silver files,
source workbook, split, model, and threshold exactly match step 05.

```bash
python3 political_classifier/scripts/06_train_final_classifier.py \
  --extra-human-validation /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/active_learning/uk_human_validation_reviewed.csv \
  --batch-size 32
```

Review the printed held-out report and:

```text
silver_classifier/classifier_run_manifest.json
silver_classifier/country_validation_results.csv
silver_classifier/threshold_validation_results.csv
```

`threshold_validation_results.csv` is still the calibration partition; country
validation is held-out.

## Step 06B: Score The Full Source-Filtered Corpus

Run only after accepting step 07. Existing classified country files are not
silently mixed with a new run, so a deliberate full rebuild uses
`--overwrite-classifications`.

```bash
CUDA_VISIBLE_DEVICES=1 nohup python3 -u political_classifier/scripts/06_train_final_classifier.py \
  --extra-human-validation /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/active_learning/uk_human_validation_reviewed.csv \
  --batch-size 32 \
  --score-corpus \
  --overwrite-classifications \
  > 06_score_final_corpus.log 2>&1 &
```

Monitor:

```bash
tail -f 06_score_final_corpus.log
```

Canonical final files:

```text
political_corruption_pipeline/silver_classifier/classified_country_files/
  {country}_classified.csv.gz
political_corruption_pipeline/silver_classifier/classified_country_summary.csv
```

## Step 07: Rebuild Attention, Tables, Figure 1, And LaTeX Values

```bash
python3 political_classifier/scripts/07_build_attention_outputs.py
```

This runs the script-based attention analysis and then builds all manuscript
artifacts from saved CSVs. No notebook execution is required. It fails if
classifier input, source-filtered counts, attention counts, model, or threshold
disagree.

Generated LaTeX tables use `\scriptsize` and `adjustbox` with `max width`; they
are never enlarged to the full text width. The manuscript preamble needs:

```tex
\usepackage{booktabs}
\usepackage{adjustbox}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning}
```

The generated values file is:

```text
attention_tables/latex/political_corruption_manuscript_values.tex
```

Both Method files input this file so corpus Ns, the threshold, split sizes, and
held-out metrics are not copied by hand.

## Step 08: Upload Current Manuscript Outputs

```bash
python3 political_classifier/scripts/08_upload_outputs.py
```

The uploader requires the fresh step-07 build manifest and refuses stale
outputs.

## Step 09: Archive An Immutable Reproducibility Snapshot

```bash
python3 political_classifier/scripts/09_archive_derived_data.py \
  --groups source_inclusion cleaned_deduped silver_training_data validation_data classifier_comparison classifier_outputs attention_outputs
```

Each run is stored under `derived_data/political_classifier/runs/<timestamp>_<git>`
with SHA-256 checksums. It does not overwrite a previous run.

To restore the newest archived snapshot on a fresh machine:

```bash
python3 political_classifier/tools/restore_derived_data_from_webdav.py
```

To restore one named snapshot:

```bash
python3 political_classifier/tools/restore_derived_data_from_webdav.py \
  --archive-version YYYYMMDDTHHMMSSZ_COMMIT
```

## What To Rerun

| Change | Start again at |
|---|---|
| Reviewed source workbook | `00`, then `02` through `09` |
| Raw NewsAPI article exports or cleaning rules | `01` through `09` |
| Silver sampling, prompt, labels, or LLM model | `03` through `09` |
| Human benchmark or UK supplement | `03` through `09` |
| Classifier model or threshold grid | `05` through `09` |
| Attention aggregation/plotting only | `07`, then `08` and `09` |
| LaTeX upload only, after a current build | `08` |

Human annotation files and codebook-development samples are not generated
classifier outputs. Preserve them when cleaning or rebuilding the pipeline.
