# Political-Classifier Rebuild Workflow

This is the reproducible rebuild path for Part 1 of the paper: identifying
articles that are primarily about political corruption and producing the final
attention tables/figures.

Run commands from the repository root on `annecuda`.

## Principle

The final analytical sample is:

```text
cleaned corruption-query corpus
-> source-inclusion filter
-> political-corruption classifier
-> final political-corruption corpus
```

The source-inclusion rule is strict: keep only country/source pairs where
`conventional_journalism == Yes`. Everything else is excluded, including `No`,
missing, and ambiguous values.

## 0. Inputs And Clean Slate

The cleaned/deduplicated country files must already exist here:

```text
/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/
```

Required source-inclusion workbook:

```text
/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/source_inclusion/political_corruption_all_sources_classified.xlsx
```

If needed, copy it from the Research Drive mount:

```bash
mkdir -p /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/source_inclusion

cp "/home/akroon/webdav/ASCOR-FMG-5580-RESPOND-news-data (Projectfolder)/victims-of-corruption-paper/political_corruption_all_sources_classified.xlsx" \
  /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/source_inclusion/
```

Before rebuilding, remove old generated outputs. First run the dry run:

```bash
python3 political_classifier/scripts/reset_rebuild_outputs.py
```

If the listed paths are the old outputs you want to delete, run:

```bash
python3 political_classifier/scripts/reset_rebuild_outputs.py \
  --confirm DELETE_OLD_POLITICAL_CLASSIFIER_OUTPUTS
```

This preserves the cleaned/deduplicated country files, total-news denominators,
source-inclusion workbook, and validation review files.

## 1. Create A Fresh Silver-Training Input

This samples from the cleaned corpus after applying the source-inclusion
workbook. It does not use old silver labels.

```bash
python3 political_classifier/scripts/create_source_filtered_silver_seed.py \
  --overwrite \
  --country-targets Bulgaria:500,France:500,Hungary:500,Italy:500,Netherlands:500,Serbia:500,Sweden:500,Ukraine:500,United_Kingdom:500
```

Output:

```text
active_learning/silver_training_source_filtered_for_annotation.csv
```

## 2. Label The Fresh Silver Set

```bash
nohup python3 -u political_classifier/scripts/label_silver_batch.py \
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

## 3. Compare Classifier Models

This validates the source-filtered silver-trained classifier against the
source-filtered human validation universe.

```bash
python3 political_classifier/scripts/compare_models.py \
  --embedding-models intfloat/multilingual-e5-large \
  --batch-size 32 \
  --extra-human-validation /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/active_learning/uk_human_validation_reviewed.csv
```

Outputs:

```text
classifier_comparison/all_threshold_results.csv
classifier_comparison/best_model_results.csv
classifier_comparison/country_validation_results.csv
classifier_comparison/validation_prediction_comparison.csv
```

Then rerun:

```text
political_classifier/notebooks/02_inspect_classifier_comparison.ipynb
```

This regenerates manuscript classifier tables in:

```text
manuscript_tables/
```

## 4. Score Or Filter The Corpus

If classifier predictions already exist and only the source screen changed, use
the cheap post-filter:

```bash
python3 political_classifier/scripts/filter_classified_outputs.py
```

Output:

```text
silver_classifier/classified_country_files_source_filtered/
source_inclusion/classified_source_filter_output_summary.csv
source_inclusion/classified_source_filter_decision_summary_by_country.csv
```

If the classifier itself changed and you need fresh corpus scores, rerun full
scoring instead:

```bash
TMPDIR=/home/akroon/data/1t_storage/tmp \
HF_HOME=/home/akroon/data/1t_storage/huggingface_cache \
TRANSFORMERS_CACHE=/home/akroon/data/1t_storage/huggingface_cache \
CUDA_VISIBLE_DEVICES=1 \
nohup python3 -u political_classifier/scripts/train_final_classifier.py \
  --extra-human-validation /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/active_learning/uk_human_validation_reviewed.csv \
  --score-corpus \
  > silver_classifier_final_scoring.log 2>&1 &
```

Monitor:

```bash
tail -f silver_classifier_final_scoring.log
```

## 5. Rebuild Attention Tables And Figures

Rerun from section 3 onward:

```text
political_classifier/notebooks/03_analyze_political_corruption_attention.ipynb
```

The notebook applies the source-inclusion filter before creating the final
attention numerator. The denominator remains the total-news count series from
NewsAPI.

Outputs:

```text
attention_tables/
attention_tables/latex/
attention_figures/
```

## 6. Upload Manuscript Outputs

```bash
python3 political_classifier/scripts/upload_manuscript_tables.py
python3 political_classifier/scripts/upload_attention_outputs.py
```

## 7. Archive Derived Data

```bash
python3 political_classifier/scripts/archive_derived_data_to_webdav.py \
  --groups source_inclusion silver_training_data classifier_comparison classifier_outputs attention_outputs
```

This updates the Research Drive reproducibility archive and manifest.

## What To Rerun When Something Changes

| Change | Rerun |
|---|---|
| Source workbook changes | Steps 1-7 |
| Silver prompt/model changes | Steps 2-7 |
| Classifier model/threshold changes | Steps 3-7 |
| Only attention plotting code changes | Steps 5-6 |
| Only Overleaf tables need reupload | Step 6 |
