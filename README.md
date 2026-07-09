# RESPOND Victims of Corruption

This repository contains the analysis workflow for the RESPOND victims-of-corruption paper.

The project is organized in parts. At the moment, the implemented workflow is **Part 1: identifying articles that are primarily about political corruption**. Later analyses, such as victim visibility and corruption-domain classification, should live in separate folders rather than being added to the political-classifier sequence.

Raw news collection code is maintained in the related RESPOND media repository:

```text
https://github.com/annekroon/RESPOND_media/tree/main/data-collection/news-collection/news-api
```

## Repository Map

| Path | Purpose |
|---|---|
| `political_classifier/` | Part 1 workflow: clean/dedupe, silver labels, classifier comparison, final scoring, attention tables |
| `topic_classification/` | Exploratory topic discovery and corruption-type mapping with country-time stratified samples |
| `config.py` | Shared non-secret paths and defaults |
| `config_local.example.py` | Template for ignored local credentials |
| `dataloader.py` | Shared data-loading helpers |
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

## Current Main Commands

From the repo root on `annecuda`:

```bash
cd ~/RESPOND-victims-of-corruption
git pull
```

Compare candidate political-corruption classifiers:

```bash
python3 political_classifier/scripts/compare_models.py \
  --embedding-models intfloat/multilingual-e5-large \
  --batch-size 32 \
  --extra-human-validation /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/active_learning/uk_human_validation_reviewed.csv
```

Train/evaluate and score the cleaned corpus with the final classifier:

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

Inspect political-corruption attention over time:

```text
political_classifier/notebooks/03_analyze_political_corruption_attention.ipynb
```

## Part 2: Topic Classification And Discovery

See:

```text
topic_classification/README.md
```

This workflow starts with reproducible random samples across country and year,
then fits multilingual BERTopic models for a general news overview or a
political-corruption-only topic map. The topic outputs are intended as a coding
frame discovery step before building final corruption-type labels.
