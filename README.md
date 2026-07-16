# RESPOND Victims of Corruption

This repository contains the analysis workflow for the RESPOND victims-of-corruption paper.

The project is organized in parts. **Part 1 identifies articles that are primarily about political corruption** and produces the cleaned corpus, classifier outputs, attention figures, and reproducibility archive. Later analyses, such as victim identification and corruption-domain classification, should live in separate folders rather than being added to the political-classifier sequence.

Topic modelling is a separate exploratory step. It is deliberately inductive and is mainly intended for appendix material and for inspiration when developing substantive coding variables later, especially victim identification and corruption-type/domain categories. It is not the final supervised coding workflow for those variables.

Raw news collection code is maintained in the related RESPOND media repository:

```text
https://github.com/annekroon/RESPOND_media/tree/main/data-collection/news-collection/news-api
```

## Repository Map

| Path | Purpose |
|---|---|
| `political_classifier/` | Part 1 workflow: clean/dedupe, silver labels, classifier comparison, final scoring, attention tables |
| `topic_classification/` | Inductive political-corruption topic discovery for appendix/exploratory interpretation and future coding-frame development |
| `content-classification/` | Article-level GPT 5.1 zero-shot coding of victim visibility, corruption frames, case scope, and accused actors |
| `config.py` | Shared non-secret paths and defaults |
| `config_local.example.py` | Template for ignored local credentials |
| `dataloader.py` | Shared data-loading helpers |
| `extract_cpi_from_transparency.py` | Preferred helper to download official Transparency International CPI full-results files into country-year scores |
| `extract_cpi_from_pdfs.py` | Fallback helper to parse CPI PDF reports from Research Drive/SURF into country-year scores |
| `upload_cpi_to_webdav.py` | Helper to upload extracted CPI country-year scores and extraction logs to Research Drive/SURF |
| `rd_io.py`, `rd_utils.py` | Research Drive/WebDAV helpers |
| `requirements.txt` | Python dependencies |
| `src/` | Older exploratory scripts kept for provenance |

Research Drive credentials and UvA LLM proxy tokens belong in ignored `config_local.py`, not in git:

```bash
cp config_local.example.py config_local.py
```

## Part 1: Political-Corruption Classifier

See:

```text
political_classifier/README.md
```

### Order Of Execution

Run the Part 1 political-corruption workflow from the repository root on
`annecuda`. The numbered scripts are the reproducible backbone; notebooks are
for inspection and sanity checks.

First update the repository:

```bash
cd ~/RESPOND-victims-of-corruption
git pull
```

If `git pull` is blocked by local notebook outputs, stash those outputs first:

```bash
git stash push -m "local notebook outputs before pull" -- \
  political_classifier/notebooks/02_inspect_classifier_comparison.ipynb \
  political_classifier/notebooks/03_analyze_political_corruption_attention.ipynb

git pull
```

Then run the rebuild in this order:

```bash
python3 political_classifier/scripts/00_download_source_workbook.py --overwrite
python3 political_classifier/scripts/01_clean_dedupe_data.py --overwrite
python3 political_classifier/scripts/02_prepare_classifier_training_sample.py \
  --overwrite \
  --country-targets Bulgaria:500,France:500,Hungary:500,Italy:500,Netherlands:500,Serbia:500,Sweden:500,Ukraine:500,United_Kingdom:500
```

Label the fresh source-filtered silver set with the UvA LLM proxy:

```bash
nohup python3 -u political_classifier/scripts/03_label_silver_batch.py \
  --input /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/active_learning/silver_training_source_filtered_for_annotation.csv \
  --output /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/active_learning/silver_training_source_filtered_with_llm_suggestions.csv \
  --max-chars 3000 \
  --overwrite \
  > llm_silver_training_source_filtered.log 2>&1 &

tail -f llm_silver_training_source_filtered.log
```

After the LLM labelling finishes, compare and inspect the classifier:

```bash
python3 political_classifier/scripts/04_compare_models.py \
  --embedding-models intfloat/multilingual-e5-large \
  --batch-size 32 \
  --extra-human-validation /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/active_learning/uk_human_validation_reviewed.csv
```

Inspect the generated validation tables in:

```text
political_classifier/notebooks/02_inspect_classifier_comparison.ipynb
```

Only after accepting classifier performance, run the expensive full-corpus
scoring step:

```bash
TMPDIR=/home/akroon/data/1t_storage/tmp \
HF_HOME=/home/akroon/data/1t_storage/huggingface_cache \
TRANSFORMERS_CACHE=/home/akroon/data/1t_storage/huggingface_cache \
CUDA_VISIBLE_DEVICES=1 \
nohup python3 -u political_classifier/scripts/05_train_final_classifier.py \
  --extra-human-validation /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/active_learning/uk_human_validation_reviewed.csv \
  --score-corpus \
  > silver_classifier_final_scoring.log 2>&1 &

tail -f silver_classifier_final_scoring.log
```

After full scoring finishes, rebuild, upload, and archive outputs:

```bash
python3 political_classifier/scripts/06_build_attention_outputs.py
python3 political_classifier/scripts/07_upload_outputs.py
python3 political_classifier/scripts/08_archive_derived_data.py \
  --groups source_inclusion cleaned_deduped silver_training_data classifier_comparison classifier_outputs attention_outputs
```

Final selected political-corruption classifier:

| Item | Value |
|---|---|
| Training labels | LLM silver-labelled training set |
| Classifier | Balanced logistic regression |
| Embeddings | `intfloat/multilingual-e5-large` |
| Decision threshold | `0.40` |
| Human validation set | Original 452 rows + 50 manually reviewed UK supplement rows |
| Validation rows | `502` |
| Political-corruption support | `141` |
| Political precision | `0.731` |
| Political recall | `0.809` |
| Political F1 | `0.768` |
| Accuracy | `0.863` |
| Macro F1 | `0.835` |
| Weighted F1 | `0.865` |

The main derived outputs are stored on `annecuda` under:

```text
/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/
```

The final political-corruption analytical sample applies an additional
source-inclusion screen. The reviewed workbook is expected at:

```text
/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/source_inclusion/political_corruption_all_sources_classified.xlsx
```

Only sources marked `Yes` in the `conventional_journalism` column are retained
for final political-corruption attention tables and figures.

The reproducibility archive for expensive-to-recreate derived data is stored on
Research Drive/WebDAV under:

```text
ASCOR-FMG-5580-RESPOND-news-data (Projectfolder)/victims-of-corruption-paper/derived_data/political_classifier/
```

Create/update that archive from `annecuda` with:

```bash
python3 political_classifier/scripts/08_archive_derived_data.py
```

Restore the archived derived data into the expected local folder with:

```bash
python3 political_classifier/scripts/restore_derived_data_from_webdav.py
```

The archive contains cleaned/deduplicated files, denominator tables,
silver-labelled training data, validation supplements, classifier outputs,
attention outputs, and a `derived_data_manifest.json` with file sizes,
checksums, destination paths, and the git commit used for the archive.

## CPI / Corruption Perceptions Index

The preferred CPI source is Transparency International's official yearly CPI
page and linked full-results spreadsheet/archive. This is more reproducible than
parsing report PDFs because the score and rank columns are structured data:

```bash
python3 extract_cpi_from_transparency.py \
  --years 2018 2019 2020 2021 2022 2023 2024 2025 \
  --output output/cpi_country_year_scores.csv
```

By default the website extractor keeps only the project countries listed in
`config.py` and drops any selected-country year unless all project countries
were recovered. It writes:

```text
output/cpi_country_year_scores.csv
output/cpi_country_year_scores_extraction_log.csv
```

The output contains `year`, `country`, `cpi_score`, `cpi_rank`, `source_url`,
`source_file`, and `extraction_method`. CPI scores are the Transparency
International values from 0 to 100; ranks are stored separately. Use the log to
verify the official source file used for each year.

The CPI PDF reports are still stored on Research Drive/SURF under:

```text
ASCOR-FMG-5580-RESPOND-news-data (Projectfolder)/victims-of-corruption-paper/CPI/
```

Use the PDF parser only as a fallback/audit route:

```bash
python3 extract_cpi_from_pdfs.py \
  --source webdav \
  --output output/cpi_country_year_scores.csv
```

For the PDF parser, use `--min-selected-countries` only if you intentionally
want to relax the completeness threshold, `--allow-partial-years` only for
debugging PDF layouts, and `--country-scope all` only if you need every
country/territory from the CPI PDFs.

Upload the extracted scores and extraction log back to Research Drive/SURF with:

```bash
python3 upload_cpi_to_webdav.py
```

By default this uploads:

```text
output/cpi_country_year_scores.csv
output/cpi_country_year_scores_extraction_log.csv
```

to:

```text
ASCOR-FMG-5580-RESPOND-news-data (Projectfolder)/victims-of-corruption-paper/derived_data/cpi/
```

For local PDFs instead of WebDAV:

```bash
python3 extract_cpi_from_pdfs.py \
  --source local \
  --local-dir /path/to/CPI \
  --output output/cpi_country_year_scores.csv
```

## Current Main Commands

From the repo root on `annecuda`:

```bash
cd ~/RESPOND-victims-of-corruption
git pull
```

The reproducible Part 1 political-corruption pipeline is:

```bash
python3 political_classifier/scripts/00_download_source_workbook.py --overwrite
python3 political_classifier/scripts/01_clean_dedupe_data.py --overwrite
python3 political_classifier/scripts/02_prepare_classifier_training_sample.py --overwrite
nohup python3 -u political_classifier/scripts/03_label_silver_batch.py --overwrite \
  > llm_silver_training_source_filtered.log 2>&1 &
python3 political_classifier/scripts/04_compare_models.py \
  --embedding-models intfloat/multilingual-e5-large \
  --batch-size 32 \
  --extra-human-validation /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/active_learning/uk_human_validation_reviewed.csv
```

After accepting classifier performance, score the cleaned/source-filtered
corpus:

```bash
TMPDIR=/home/akroon/data/1t_storage/tmp \
HF_HOME=/home/akroon/data/1t_storage/huggingface_cache \
TRANSFORMERS_CACHE=/home/akroon/data/1t_storage/huggingface_cache \
CUDA_VISIBLE_DEVICES=1 \
nohup python3 -u political_classifier/scripts/05_train_final_classifier.py \
  --extra-human-validation /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/active_learning/uk_human_validation_reviewed.csv \
  --score-corpus \
  > silver_classifier_final_scoring.log 2>&1 &
```

Then rebuild and upload attention outputs:

```bash
python3 political_classifier/scripts/06_build_attention_outputs.py
python3 political_classifier/scripts/07_upload_outputs.py
python3 political_classifier/scripts/08_archive_derived_data.py \
  --groups source_inclusion cleaned_deduped silver_training_data classifier_comparison classifier_outputs attention_outputs
```

## Part 2: Topic Classification And Discovery

See:

```text
topic_classification/README.md
```

This workflow starts with reproducible random samples across country and year
among articles classified as political corruption, then fits multilingual
BERTopic models, labels topics with GPT 5.1 through the UvA LLM proxy, and
uses a notebook to inspect topic tables, example articles, and interactive
country/time visualizations. The topic outputs are intended as an inductive
appendix/discovery step before building final substantive variables. They can
help inspire later coding of victim visibility, victim type, corruption domain,
and case scope, but they are not treated as final measurement of those
variables.
