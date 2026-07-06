# RESPOND Victims of Corruption

This repository contains a reproducible workflow for building a political-corruption classifier for the RESPOND multilingual news corpus.

The code used to collect the raw news data is maintained in the related RESPOND media repository:

```text
https://github.com/annekroon/RESPOND_media/tree/main/data-collection/news-collection/news-api
```

The final recommended classifier is:

- **Training labels:** LLM-generated silver labels from batches 1 and 2
- **Model:** multilingual sentence embeddings + balanced logistic regression
- **Embedding model:** `intfloat/multilingual-e5-large`
- **Decision threshold:** `0.40`
- **Validation set:** 452 manually labelled articles
- **Validation performance:** political-corruption precision `0.766`, recall `0.792`, F1 `0.779`, accuracy `0.881`

The cleaned corpus and derived outputs are stored on `annecuda` under:

```text
/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/
```

## Setup

Clone/pull the repository on `annecuda`:

```bash
cd ~/RESPOND-victims-of-corruption
git pull
```

Install Python dependencies in the environment used for the notebooks/scripts:

```bash
pip3 install --user -r requirements.txt
```

Research Drive credentials and the UvA LLM proxy token should go in ignored `config_local.py`, not in `config.py`. Start from:

```bash
cp config_local.example.py config_local.py
```

## Main Execution Order

Run the numbered files in this order when rebuilding the workflow. Files without numbers are shared helpers and are not meant to be run directly.

| Step | File | Run when |
|---|---|---|
| 1 | `01_load_clean_dedupe_data.ipynb` | Clean/dedupe raw corpus and create denominator tables |
| 2 | `02_create_targeted_silver_label_batch.py` | Create a targeted silver-label CSV |
| 3 | `03_label_silver_label_batch.py` | Label a silver-label CSV with the UvA LLM proxy |
| 4 | `04_compare_classifier_models.py` + `04_compare_classifier_models.ipynb` | Run reproducible model comparison, then inspect tables/plots |
| 5 | `05_train_final_classifier.py` | Train the final combined silver-label classifier and classify the full corpus |
| 6 | `06_upload_manuscript_tables.py` | Upload generated LaTeX manuscript tables to Research Drive |
| 7 | `07_analyze_political_corruption_attention.ipynb` | Build weekly/monthly attention tables and plot political-corruption attention over time |
| 8 | `08_create_uk_silver_label_batch.py` | Create a UK-focused calibration batch for classifier auditing/improvement |
| 9 | `09_create_uk_human_validation_review_batch.py` | Create a manually reviewable UK validation supplement from LLM-labelled UK cases |

## Shared Helper Files

| File | Purpose |
|---|---|
| `config.py` | Shared paths and non-secret defaults |
| `dataloader.py` | Loads raw and annotated data from Research Drive/WebDAV |
| `rd_io.py`, `rd_utils.py` | Research Drive/WebDAV I/O helpers |
| `requirements.txt` | Python dependencies |
| `miscellaneous/annotation_interface.py` | Optional Streamlit annotation UI; not needed for the final scripted workflow |
| `miscellaneous/llmproxy_label_batch_notebook.ipynb` | Older notebook version of LLM batch labelling; kept for reference |

## Workflow

### 1. Clean And Dedupe The Corpus

Run the notebook:

```text
01_load_clean_dedupe_data.ipynb
```

This creates cleaned compressed files such as:

```text
{country}_cleaned_deduped.csv.gz
all_countries_cleaned_deduped_minimal.csv.gz
denominator_country_year.csv
denominator_country_month.csv
denominator_country_week.csv
```

### 2. Create LLM Silver Labels

Batch 1 was the initial LLM-labelled sample. Batch 2 was a targeted follow-up sample focused on weaker countries and boundary cases.

To create a targeted follow-up batch:

```bash
python3 02_create_targeted_silver_label_batch.py \
  --country-targets Sweden:540,United_Kingdom:540,Ukraine:540,Netherlands:420,Serbia:360,Hungary:180,Bulgaria:180,Italy:120,France:120
```

To label that batch with the UvA LLM proxy:

```bash
nohup python3 -u 03_label_silver_label_batch.py \
  --input /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/active_learning/active_learning_batch_2_for_annotation.csv \
  --output /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/active_learning/active_learning_batch_2_with_llm_suggestions.csv \
  --max-chars 3000 \
  > llm_silver_label_batch_2.log 2>&1 &
```

Monitor with:

```bash
tail -f llm_silver_label_batch_2.log
```

### 3. Optional Manual Review Interface

Start Streamlit on `annecuda`:

```bash
streamlit run miscellaneous/annotation_interface.py --server.address 0.0.0.0 --server.port 8501
```

From your laptop, open an SSH tunnel:

```bash
ssh -N -L 8501:127.0.0.1:8501 akroon@145.38.195.108
```

Then open:

```text
http://127.0.0.1:8501
```

### 4. Compare Candidate Classifiers

Run the reproducible comparison script:

```bash
python3 04_compare_classifier_models.py
```

To compare several multilingual sentence-embedding models in one run:

```bash
python3 04_compare_classifier_models.py \
  --embedding-models \
  sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
  sentence-transformers/paraphrase-multilingual-mpnet-base-v2 \
  sentence-transformers/LaBSE \
  intfloat/multilingual-e5-base
```

For a longer weekend run, you can add heavier models:

```bash
python3 04_compare_classifier_models.py \
  --embedding-models \
  sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
  sentence-transformers/paraphrase-multilingual-mpnet-base-v2 \
  sentence-transformers/LaBSE \
  intfloat/multilingual-e5-base \
  intfloat/multilingual-e5-large \
  BAAI/bge-m3 \
  --batch-size 32
```

These are sentence-embedding models, so they are directly comparable with the current logistic-regression classifier. Raw RoBERTa/XLM-R models are not included here because they are not sentence-embedding classifiers by themselves; comparing them properly would require a separate fine-tuning setup.

This writes:

```text
/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/classifier_comparison/
```

Main comparison tables:

| Output | Meaning |
|---|---|
| `best_model_results.csv` | Best threshold per candidate model |
| `all_threshold_results.csv` | Full threshold sweep for every candidate |
| `country_results_for_best_silver_thresholds.csv` | Country-level validation results |
| `validation_prediction_comparison.csv` | Row-level validation predictions |

The current comparison supports using the combined silver batches for the final classifier because it is transparent, uses all available silver labels, and performs almost identically to the best single-batch variant.

Then open the inspection notebook:

```text
04_compare_classifier_models.ipynb
```

The notebook does not redo the expensive model comparison. It reads the saved CSV files, displays the non-truncated final table, plots threshold trade-offs, shows country-level F1 scores, and prints the final scoring command for the selected model.

It also writes manuscript/appendix LaTeX tables to:

```text
/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/manuscript_tables/
```

Generated tables include the main political-corruption classifier comparison, the full appendix comparison, country-level validation metrics, and the threshold sweep for the selected final political-corruption classifier. Model-level tables include accuracy, political-class precision/recall/F1, macro F1, and weighted F1. The generated tables use `booktabs` and wrap wide tabular content in `\resizebox{\textwidth}{!}{...}`, so the manuscript preamble should include `\usepackage{booktabs}` and `\usepackage{graphicx}`.

To upload these generated tables to Research Drive/WebDAV:

```bash
python3 06_upload_manuscript_tables.py
```

By default this uploads local tables from:

```text
/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/manuscript_tables/
```

to:

```text
ASCOR-FMG-5580-RESPOND-news-data (Projectfolder)/victims-of-corruption-paper/output/tables/
```

To check the GPU on `annecuda` before a longer comparison run:

```bash
nvidia-smi
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv
python3 - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)
PY
```

### 5. Final Full-Corpus Classification

The final decision from the comparison notebook is:

| Decision | Value |
|---|---|
| Training data | Combined silver-label batches 1 and 2 |
| Classifier | Balanced logistic regression |
| Embeddings | `intfloat/multilingual-e5-large` |
| Threshold | `0.40` |
| Validation precision | `0.766` |
| Validation recall | `0.792` |
| Validation political-corruption F1 | `0.779` |
| Validation accuracy | `0.881` |

Run a small end-to-end test first:

```bash
TMPDIR=/home/akroon/data/1t_storage/tmp \
HF_HOME=/home/akroon/data/1t_storage/huggingface_cache \
TRANSFORMERS_CACHE=/home/akroon/data/1t_storage/huggingface_cache \
CUDA_VISIBLE_DEVICES=1 \
python3 05_train_final_classifier.py \
  --score-corpus \
  --countries Serbia
```

Then classify every cleaned country file:

```bash
TMPDIR=/home/akroon/data/1t_storage/tmp \
HF_HOME=/home/akroon/data/1t_storage/huggingface_cache \
TRANSFORMERS_CACHE=/home/akroon/data/1t_storage/huggingface_cache \
CUDA_VISIBLE_DEVICES=1 \
nohup python3 -u 05_train_final_classifier.py \
  --score-corpus \
  > silver_classifier_final_scoring.log 2>&1 &
```

Monitor:

```bash
tail -f silver_classifier_final_scoring.log
```

Final outputs are written to:

```text
/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/silver_classifier/
```

Important files:

| Output | Meaning |
|---|---|
| `human_validation_predictions.csv` | Predictions on the 452 manually labelled validation rows |
| `threshold_validation_results.csv` | Threshold sweep on the human validation set |
| `country_validation_results.csv` | Country-level validation metrics |
| `classified_country_summary.csv` | Full-corpus predicted counts by country |
| `classified_country_year_counts.csv` | Full-corpus predicted counts by country-year |
| `classified_country_files/{country}_classified.csv.gz` | Row-level predictions for each cleaned country file |

## Final Model Reporting

Suggested concise wording:

> We trained a multilingual sentence-embedding classifier using LLM-generated silver labels from two targeted silver-label batches. The classifier used `intfloat/multilingual-e5-large` embeddings and a class-balanced logistic regression model. Candidate models were compared against a held-out manually annotated validation set of 452 articles. The final model was trained on the combined silver-label batches and used a decision threshold of 0.40, selected from validation-set threshold sweeps to balance precision and recall for the political-corruption class. On the validation set, the final classifier achieved precision = 0.766, recall = 0.792, and F1 = 0.779 for the political-corruption class.

## Political-Corruption Attention Analysis

After the full corpus has been classified, run:

```text
07_analyze_political_corruption_attention.ipynb
```

The notebook loads total-news denominator files:

```text
total_news_coverage_week.csv
total_news_coverage_month.csv
```

from:

```text
/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/
```

If these local files do not exist, the notebook first looks for mounted WebDAV files such as `Bulgaria_weekly_count.csv` and `UK_weekly_count.csv` in:

```text
/home/akroon/webdav/ASCOR-FMG-5580-RESPOND-news-data (Projectfolder)/weekly_counts_total_coverage/
```

If the mounted folder is unavailable, it falls back to Research Drive/WebDAV API access and caches local copies:

```text
ASCOR-FMG-5580-RESPOND-news-data (Projectfolder)/weekly_counts_total_coverage/
```

These files must count **all news coverage** by country-period, not just the corruption-query corpus. Expected columns are:

```text
country
week or month
total_news_articles
```

The count column may also be named `total_articles`, `total_coverage`, `n_articles`, or `count`; the notebook standardizes it to `total_news_articles`.

The notebook also loads the classified political-corruption numerator:

```text
silver_classifier/classified_country_files/{country}_classified.csv.gz
```

It writes derived weekly/monthly attention tables to:

```text
/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/attention_tables/
```

and publication-style figures to:

```text
/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/attention_figures/
```

The notebook also writes LaTeX summary tables to:

```text
/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/attention_tables/latex/
```

These include country summaries, yearly summaries, country-year relative-attention matrices, and peak months of political-corruption attention. Figures include monthly/weekly country trend lines, small multiples, stacked absolute volume, a country-year heatmap, an average-attention country ranking, and an indexed monthly attention plot.

Relative attention is defined as the share of **total news coverage** classified as political corruption in a given country-period:

```text
political_corruption_articles / total_news_articles
```

The cleaned corruption-query denominator files are only used in an optional diagnostic cell and are not the main denominator.

## UK Classifier Calibration Batch

The United Kingdom had the weakest country-level validation performance and very low positive support in the manual validation set. To audit and potentially improve the classifier, create a UK-focused third silver-label batch from the final classified UK output:

```bash
python3 08_create_uk_silver_label_batch.py
```

This writes:

```text
/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/active_learning/uk_calibration_batch_for_annotation.csv
```

The default target is 1,500 UK articles sampled from:

```text
40% threshold-boundary cases, probability 0.30--0.50
30% high predicted positives, probability >= 0.60
20% high predicted negatives, probability <= 0.20
10% random checks
```

Label the batch with the same LLM prompt as earlier silver-label batches:

```bash
nohup python3 -u 03_label_silver_label_batch.py \
  --input /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/active_learning/uk_calibration_batch_for_annotation.csv \
  --output /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/active_learning/uk_calibration_batch_with_llm_suggestions.csv \
  --max-chars 3000 \
  > llm_uk_calibration_batch.log 2>&1 &
```

Monitor with:

```bash
tail -f llm_uk_calibration_batch.log
```

Afterwards, compare models trained with and without this UK batch before changing the final classifier. The comparison script includes the UK calibration labels by default when the file exists, but keeps the baseline and UK-calibrated training sets separate in the output:

- `silver_combined`: original silver-label batches 1 and 2.
- `silver_combined_with_uk_calibration`: original silver-label batches plus the UK calibration batch.

The new comparison experiment is written to:

```text
/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/classifier_comparison_uk_calibration/
```

Run:

```bash
python3 04_compare_classifier_models.py \
  --embedding-models intfloat/multilingual-e5-large \
  --batch-size 32
```

Then inspect:

```text
04_compare_classifier_models.ipynb
```

If the UK-calibrated model improves UK performance without damaging overall validation, train/evaluate it into a separate output folder:

```bash
TMPDIR=/home/akroon/data/1t_storage/tmp \
HF_HOME=/home/akroon/data/1t_storage/huggingface_cache \
TRANSFORMERS_CACHE=/home/akroon/data/1t_storage/huggingface_cache \
CUDA_VISIBLE_DEVICES=1 \
python3 05_train_final_classifier.py \
  --include-uk-calibration
```

With `--include-uk-calibration`, `05_train_final_classifier.py` appends the UK calibration batch and writes validation/model outputs to:

```text
/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/silver_classifier_uk_calibrated/
```

For a full UK-calibrated corpus scoring run:

```bash
TMPDIR=/home/akroon/data/1t_storage/tmp \
HF_HOME=/home/akroon/data/1t_storage/huggingface_cache \
TRANSFORMERS_CACHE=/home/akroon/data/1t_storage/huggingface_cache \
CUDA_VISIBLE_DEVICES=1 \
nohup python3 -u 05_train_final_classifier.py \
  --include-uk-calibration \
  --score-corpus \
  > silver_classifier_uk_calibrated_scoring.log 2>&1 &
```

## UK Human Validation Supplement

The UK calibration batch did not improve the combined classifier, but it is still useful as a pool of translated, LLM-labelled UK cases for manual validation. To create a smaller review file from those labelled cases:

```bash
python3 09_create_uk_human_validation_review_batch.py \
  --target-n 300
```

This writes:

```text
/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/active_learning/uk_human_validation_review_batch.csv
```

Review it in the Streamlit interface:

```bash
RESPOND_ANNOTATION_INPUT=/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/active_learning/uk_human_validation_review_batch.csv \
RESPOND_ANNOTATION_OUTPUT=/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/active_learning/uk_human_validation_reviewed.csv \
streamlit run miscellaneous/annotation_interface.py \
  --server.address 0.0.0.0 \
  --server.port 8501
```

Rows with a completed `human_final_label` can then be appended to the human validation benchmark:

```bash
python3 04_compare_classifier_models.py \
  --embedding-models intfloat/multilingual-e5-large \
  --batch-size 32 \
  --extra-human-validation /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/active_learning/uk_human_validation_reviewed.csv
```

This does not retrain on the reviewed UK rows. It evaluates the existing candidate models against the original human validation set plus the new manually reviewed UK supplement.
