# Part 1: Political-Corruption Classifier

This folder contains the workflow for identifying which cleaned news articles are primarily about political corruption. It is deliberately separate from later victim-visibility analyses.

## Folder Structure

| Path | Purpose |
|---|---|
| `notebooks/01_clean_dedupe_data.ipynb` | Load raw country files, clean text, deduplicate, and write denominator tables |
| `notebooks/02_inspect_classifier_comparison.ipynb` | Inspect saved classifier comparison outputs and generate manuscript tables |
| `notebooks/03_analyze_political_corruption_attention.ipynb` | Analyze relative attention to political corruption over time |
| `scripts/create_silver_label_batch.py` | Create targeted silver-label batches |
| `scripts/label_silver_batch.py` | Send a silver-label batch to the UvA LLM proxy |
| `scripts/compare_models.py` | Reproducible classifier comparison and validation |
| `scripts/train_final_classifier.py` | Train/evaluate the final classifier and optionally score all cleaned country files |
| `scripts/upload_manuscript_tables.py` | Upload generated LaTeX tables to Research Drive |
| `scripts/create_uk_silver_batch.py` | Create the UK calibration silver-label batch |
| `scripts/create_uk_validation_review_batch.py` | Create a small UK human-validation review file from LLM-labelled UK cases |
| `scripts/merge_uk_validation_annotations.py` | Save a merged annotation file with the reviewed UK supplement |
| `tools/annotation_interface.py` | Streamlit UI for manual review |
| `archive/` | Older notebook versions kept for provenance |

## Final Classifier Decision

The final political-corruption classifier uses the UK-calibrated combined silver-label training set. This choice is based on the comparison run with the manually reviewed UK validation supplement.

| Metric | Value |
|---|---|
| Label source | `silver_combined_with_uk_calibration` |
| Training rows | `5,480` |
| Embedding model | `intfloat/multilingual-e5-large` |
| Classifier | Balanced logistic regression |
| Threshold | `0.40` |
| Validation rows | `502` |
| Political-corruption support | `141` |
| Accuracy | `0.871` |
| Political precision | `0.750` |
| Political recall | `0.809` |
| Political F1 | `0.778` |
| Macro F1 | `0.843` |
| Weighted F1 | `0.872` |
| Predicted positive rate on validation | `0.303` |

Compared with the original batches 1+2 model, the UK-calibrated model improves overall political F1 and improves UK-specific F1 in the reviewed UK validation supplement.

## Workflow

Run commands from the repository root on `annecuda`.

### 1. Clean And Dedupe

Run:

```text
political_classifier/notebooks/01_clean_dedupe_data.ipynb
```

This creates cleaned compressed country files and denominator tables under:

```text
/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/
```

Important outputs include:

```text
{country}_cleaned_deduped.csv.gz
all_countries_cleaned_deduped_minimal.csv.gz
denominator_country_year.csv
denominator_country_month.csv
denominator_country_week.csv
```

### 2. Create And Label Silver Data

Create a targeted silver-label batch:

```bash
python3 political_classifier/scripts/create_silver_label_batch.py \
  --country-targets Sweden:540,United_Kingdom:540,Ukraine:540,Netherlands:420,Serbia:360,Hungary:180,Bulgaria:180,Italy:120,France:120
```

Label a batch with the UvA LLM proxy:

```bash
nohup python3 -u political_classifier/scripts/label_silver_batch.py \
  --input /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/active_learning/active_learning_batch_2_for_annotation.csv \
  --output /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/active_learning/active_learning_batch_2_with_llm_suggestions.csv \
  --max-chars 3000 \
  > llm_silver_label_batch_2.log 2>&1 &
```

### 3. UK Calibration And Validation Supplement

The original balanced human validation set had low UK positive support. A UK calibration batch was therefore created from already classified UK articles and labelled with the same LLM prompt:

```bash
python3 political_classifier/scripts/create_uk_silver_batch.py
```

A smaller UK review file was then sampled from the translated/LLM-labelled UK cases:

```bash
python3 political_classifier/scripts/create_uk_validation_review_batch.py \
  --target-n 50
```

Review it with Streamlit:

```bash
RESPOND_ANNOTATION_INPUT=/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/active_learning/uk_human_validation_review_batch.csv \
RESPOND_ANNOTATION_OUTPUT=/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/active_learning/uk_human_validation_reviewed.csv \
python3 -m streamlit run political_classifier/tools/annotation_interface.py \
  --server.address 127.0.0.1 \
  --server.port 8501
```

If working through an SSH tunnel:

```bash
ssh -L 8501:127.0.0.1:8501 akroon@145.38.195.108
```

Then open:

```text
http://localhost:8501
```

To save a merged copy of the annotation file on Research Drive:

```bash
python3 political_classifier/scripts/merge_uk_validation_annotations.py
```

The original annotation file is not overwritten.

### 4. Compare Candidate Models

Run the final comparison:

```bash
python3 political_classifier/scripts/compare_models.py \
  --embedding-models intfloat/multilingual-e5-large \
  --batch-size 32 \
  --extra-human-validation /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/active_learning/uk_human_validation_reviewed.csv
```

For a broader embedding comparison:

```bash
python3 political_classifier/scripts/compare_models.py \
  --embedding-models \
  sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
  sentence-transformers/paraphrase-multilingual-mpnet-base-v2 \
  sentence-transformers/LaBSE \
  intfloat/multilingual-e5-base \
  intfloat/multilingual-e5-large \
  BAAI/bge-m3 \
  --batch-size 32 \
  --extra-human-validation /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/active_learning/uk_human_validation_reviewed.csv
```

Main outputs:

```text
/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/classifier_comparison_uk_calibration/
```

Then inspect:

```text
political_classifier/notebooks/02_inspect_classifier_comparison.ipynb
```

This notebook generates manuscript-ready LaTeX classifier tables.

### 5. Train And Score Final Classifier

Small test run:

```bash
TMPDIR=/home/akroon/data/1t_storage/tmp \
HF_HOME=/home/akroon/data/1t_storage/huggingface_cache \
TRANSFORMERS_CACHE=/home/akroon/data/1t_storage/huggingface_cache \
CUDA_VISIBLE_DEVICES=1 \
python3 political_classifier/scripts/train_final_classifier.py \
  --extra-human-validation /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/active_learning/uk_human_validation_reviewed.csv \
  --score-corpus \
  --countries Serbia
```

Full scoring run:

```bash
TMPDIR=/home/akroon/data/1t_storage/tmp \
HF_HOME=/home/akroon/data/1t_storage/huggingface_cache \
TRANSFORMERS_CACHE=/home/akroon/data/1t_storage/huggingface_cache \
CUDA_VISIBLE_DEVICES=1 \
nohup python3 -u political_classifier/scripts/train_final_classifier.py \
  --extra-human-validation /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/active_learning/uk_human_validation_reviewed.csv \
  --score-corpus \
  > silver_classifier_uk_calibrated_scoring.log 2>&1 &
```

Monitor:

```bash
tail -f silver_classifier_uk_calibrated_scoring.log
```

Outputs are written to:

```text
/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/silver_classifier_uk_calibrated/
```

### 6. Attention Over Time

After full scoring, run:

```text
political_classifier/notebooks/03_analyze_political_corruption_attention.ipynb
```

Relative political-corruption attention is defined as:

```text
political_corruption_articles / total_news_articles
```

The denominator is total news coverage, not the corruption-query corpus. Weekly total-news files are read from:

```text
ASCOR-FMG-5580-RESPOND-news-data (Projectfolder)/weekly_counts_total_coverage/
```

The notebook writes attention tables, LaTeX summaries, and figures under:

```text
/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/attention_tables/
/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/attention_figures/
```

## Manuscript Tables

The comparison notebook writes classifier validation tables to:

```text
/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/manuscript_tables/
```

Upload them to Research Drive:

```bash
python3 political_classifier/scripts/upload_manuscript_tables.py
```
