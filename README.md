# RESPOND Victims of Corruption

This repository contains a reproducible workflow for building a political-corruption classifier for the RESPOND multilingual news corpus.

The final recommended classifier is:

- **Training labels:** LLM-generated silver labels from active-learning batches 1 and 2
- **Model:** multilingual sentence embeddings + balanced logistic regression
- **Embedding model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Decision threshold:** `0.30`
- **Validation set:** 452 manually labelled articles
- **Validation performance:** political-corruption F1 about `0.667`

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
| 2 | `02_llmproxy_active_learning_suggestions.ipynb` | Optional notebook version of LLM labelling; mostly superseded by step 4 script |
| 3 | `03_create_targeted_active_learning_batch.py` | Create a targeted active-learning CSV |
| 4 | `04_run_llmproxy_active_learning_suggestions.py` | Label an active-learning CSV with the UvA LLM proxy |
| 5 | `05_compare_classifier_models.py` | Compare TF-IDF, human-label embedding, and silver-label embedding classifiers |
| 6 | `06_train_silver_classifier.py` | Train the final combined silver-label classifier and classify the full corpus |

## Shared Helper Files

| File | Purpose |
|---|---|
| `config.py` | Shared paths and non-secret defaults |
| `dataloader.py` | Loads raw and annotated data from Research Drive/WebDAV |
| `rd_io.py`, `rd_utils.py` | Research Drive/WebDAV I/O helpers |
| `requirements.txt` | Python dependencies |
| `miscellaneous/annotation_interface.py` | Optional Streamlit annotation UI; not needed for the final scripted workflow |

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

Batch 1 was generated from the first active-learning sample. Batch 2 was generated from a targeted follow-up sample focused on weaker countries and boundary cases.

To create a targeted follow-up batch:

```bash
python3 03_create_targeted_active_learning_batch.py \
  --country-targets Sweden:540,United_Kingdom:540,Ukraine:540,Netherlands:420,Serbia:360,Hungary:180,Bulgaria:180,Italy:120,France:120
```

To label that batch with the UvA LLM proxy:

```bash
nohup python3 -u 04_run_llmproxy_active_learning_suggestions.py \
  --input /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/active_learning/active_learning_batch_2_for_annotation.csv \
  --output /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/active_learning/active_learning_batch_2_with_llm_suggestions.csv \
  --max-chars 3000 \
  > llm_active_learning_batch_2.log 2>&1 &
```

Monitor with:

```bash
tail -f llm_active_learning_batch_2.log
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

Run:

```bash
python3 05_compare_classifier_models.py
```

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

### 5. Final Full-Corpus Classification

Train on both silver batches and classify every cleaned country file:

```bash
nohup python3 -u 06_train_silver_classifier.py \
  --threshold 0.30 \
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

> We trained a multilingual sentence-embedding classifier using LLM-generated silver labels from two active-learning batches. The classifier used `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` embeddings and a class-balanced logistic regression model. Candidate models were compared against a held-out manually annotated validation set of 452 articles. The final model was trained on the combined silver-label batches and used a decision threshold of 0.30, selected from validation-set threshold sweeps to balance precision and recall for the political-corruption class.
