# Political-Classifier Rebuild Workflow

This is the reproducible rebuild path for Part 1 of the paper: identifying
articles that are primarily about political corruption and producing the final
attention tables/figures.

Run commands from the repository root on `annecuda`.

## Principle

The final analytical sample is:

```text
raw corruption-query corpus from NewsAPI
-> clean and deduplicate
-> source-inclusion filter
-> political-corruption classifier
-> final political-corruption corpus
```

The relative-attention denominator follows a separate NewsAPI route:

```text
NewsAPI total-news count request
-> country-week/month total-news denominator
```

The source-inclusion rule is strict: keep only country/source pairs where
`conventional_journalism == Yes` in
`political_corruption_all_sources_classified.xlsx`. Everything else is
excluded, including `No`, missing, and ambiguous values.

## Numbered Script Spine

The numbered scripts are the reproducible pipeline. The notebooks are optional
inspection views and are not required to build production tables or figures.

```bash
python3 political_classifier/scripts/00_download_source_workbook.py --overwrite
python3 political_classifier/scripts/01_clean_dedupe_data.py --overwrite
python3 political_classifier/scripts/02_create_source_filtered_corpus.py --overwrite
python3 political_classifier/scripts/03_prepare_classifier_training_sample.py --overwrite
python3 political_classifier/scripts/04_label_silver_batch.py --overwrite
python3 political_classifier/scripts/05_compare_models.py
python3 political_classifier/scripts/06_train_final_classifier.py --score-corpus
python3 political_classifier/scripts/07_build_attention_outputs.py
python3 political_classifier/scripts/08_upload_outputs.py
python3 political_classifier/scripts/09_archive_derived_data.py
```

For long-running steps, use `nohup` as shown below.

## 0. Optional Clean Slate

This removes generated outputs only. It preserves cleaned input files if you do
not want to recreate them, total-news denominators, the source workbook, and
annotation/review files.

First run the dry run:

```bash
python3 political_classifier/scripts/reset_rebuild_outputs.py
```

If the listed paths are old generated outputs you want to delete, run:

```bash
python3 political_classifier/scripts/reset_rebuild_outputs.py \
  --confirm DELETE_OLD_POLITICAL_CLASSIFIER_OUTPUTS
```

## 1. Download The Source Workbook

Do not copy from the mounted WebDAV folder; the mount can throw I/O errors.
Download through the WebDAV API:

```bash
python3 political_classifier/scripts/00_download_source_workbook.py --overwrite
```

Output:

```text
/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/source_inclusion/political_corruption_all_sources_classified.xlsx
```

## 2. Clean And Deduplicate The Corruption-Query Corpus

This loads `*_news.csv` files directly from Research Drive/WebDAV, normalizes
text, removes duplicate-source rows, removes empty/short text, parses dates,
and performs URI, exact-text, and near-exact-text deduplication.

```bash
python3 political_classifier/scripts/01_clean_dedupe_data.py --overwrite
```

Key outputs:

```text
{country}_cleaned_deduped.csv.gz
all_countries_cleaned_deduped_minimal.csv.gz
denominator_country_total.csv
denominator_country_year.csv
denominator_country_month.csv
denominator_country_week.csv
denominator_country_year_source.csv
```

## 3. Create The Full Source-Filtered Corpus

This starts from the existing cleaned/deduplicated country files and removes
sources where `conventional_journalism != Yes`.

```bash
python3 political_classifier/scripts/02_create_source_filtered_corpus.py --overwrite
```

Key outputs:

```text
cleaned_deduped_source_filtered/{country}_cleaned_deduped_source_filtered.csv.gz
cleaned_deduped_source_filtered/all_countries_cleaned_deduped_source_filtered_minimal.csv.gz
source_inclusion/cleaned_source_filter_output_summary.csv
source_inclusion/cleaned_source_filter_decision_summary_by_country.csv
```

## 4. Prepare The Source-Filtered Classifier Training Sample

This samples from the full source-filtered corpus. It does not use previous
silver labels or provisional classifier scores.

```bash
python3 political_classifier/scripts/03_prepare_classifier_training_sample.py \
  --overwrite \
  --country-targets Bulgaria:500,France:500,Hungary:500,Italy:500,Netherlands:500,Serbia:500,Sweden:500,Ukraine:500,United_Kingdom:500
```

Output:

```text
active_learning/silver_training_source_filtered_for_annotation.csv
```

## 5. Label The Fresh Silver Set

This calls the UvA LLM proxy. Run it with `nohup` because it can take hours.

```bash
nohup python3 -u political_classifier/scripts/04_label_silver_batch.py \
  --input /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/active_learning/silver_training_source_filtered_for_annotation.csv \
  --output /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/active_learning/silver_training_source_filtered_with_llm_suggestions.csv \
  --max-chars 3000 \
  --overwrite \
  > llm_silver_training_source_filtered.log 2>&1 &
```

Monitor:

```bash
tail -f llm_silver_training_source_filtered.log
```

Output:

```text
active_learning/silver_training_source_filtered_with_llm_suggestions.csv
```

## 6. Compare Classifier Models

This validates the source-filtered silver-trained classifier against the fixed
human validation benchmark. The source filter is not applied to that benchmark,
which preserves its original country composition and UK supplement.

```bash
python3 political_classifier/scripts/05_compare_models.py \
  --embedding-models intfloat/multilingual-e5-large \
  --batch-size 32 \
  --extra-human-validation /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/active_learning/uk_human_validation_reviewed.csv
```

Outputs:

```text
classifier_comparison/all_threshold_results.csv
classifier_comparison/best_model_results.csv
classifier_comparison/country_results_for_best_silver_thresholds.csv
classifier_comparison/validation_prediction_comparison.csv
```

Optionally inspect the saved outputs with:

```text
political_classifier/notebooks/02_inspect_classifier_comparison.ipynb
```

## 7. Score The Final Corpus

Only run this after accepting the validation results. This is the expensive
full-corpus scoring step.

```bash
TMPDIR=/home/akroon/data/1t_storage/tmp \
HF_HOME=/home/akroon/data/1t_storage/huggingface_cache \
TRANSFORMERS_CACHE=/home/akroon/data/1t_storage/huggingface_cache \
CUDA_VISIBLE_DEVICES=1 \
nohup python3 -u political_classifier/scripts/06_train_final_classifier.py \
  --extra-human-validation /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/active_learning/uk_human_validation_reviewed.csv \
  --score-corpus \
  > silver_classifier_final_scoring.log 2>&1 &
```

Monitor:

```bash
tail -f silver_classifier_final_scoring.log
```

## 8. Rebuild Manuscript And Attention Outputs

The command-line entry point executes a clean copy of the attention notebook
outside the repository and then regenerates all classifier tables, the
country-level corpus table, and Figure 1 from saved pipeline summaries:

```bash
python3 political_classifier/scripts/07_build_attention_outputs.py
```

The build checks that the comparison and final-scoring thresholds match, that
the classified input equals the source-filtered corpus, and that attention
counts match the final classified corpus. A mismatch stops the build with the
step that needs to be rerun.

You can optionally inspect interactively in:

```text
political_classifier/notebooks/03_analyze_political_corruption_attention.ipynb
```

The attention outputs use the classified political-corruption corpus as the
numerator and the separate NewsAPI total-news count series as the denominator.

Outputs:

```text
attention_tables/
attention_tables/latex/
attention_figures/
manuscript_tables/
```

If the attention CSVs and figures are already current and only the LaTeX tables
or Figure 1 need regeneration:

```bash
python3 political_classifier/scripts/07_build_attention_outputs.py --skip-notebook
```

## 9. Upload Manuscript Outputs

```bash
python3 political_classifier/scripts/08_upload_outputs.py
```

This uploads classifier manuscript tables plus attention tables/figures.

## 10. Archive Derived Data

```bash
python3 political_classifier/scripts/09_archive_derived_data.py \
  --groups source_inclusion cleaned_deduped silver_training_data classifier_comparison classifier_outputs attention_outputs
```

This updates the Research Drive reproducibility archive and manifest.

## What To Rerun When Something Changes

| Change | Rerun |
|---|---|
| Source workbook changes | Steps 1, 3-10 |
| Raw corruption-query data changes | Steps 2-10 |
| Silver prompt/model changes | Steps 5-10 |
| Classifier model/threshold changes | Steps 6-10 |
| Only attention plotting code changes | Steps 8-9 |
| Only Overleaf tables need reupload | Step 9 |
