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
| `config.py` | Shared non-secret paths and defaults |
| `config_local.example.py` | Template for ignored local credentials |
| `dataloader.py` | Shared data-loading helpers |
| `extract_cpi_from_pdfs.py` | Helper to parse CPI PDF reports from Research Drive/SURF into country-year scores |
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

The reproducibility archive for expensive-to-recreate derived data is stored on
Research Drive/WebDAV under:

```text
ASCOR-FMG-5580-RESPOND-news-data (Projectfolder)/victims-of-corruption-paper/derived_data/political_classifier/
```

Create/update that archive from `annecuda` with:

```bash
python3 political_classifier/scripts/archive_derived_data_to_webdav.py
```

Restore the archived derived data into the expected local folder with:

```bash
python3 political_classifier/scripts/restore_derived_data_from_webdav.py
```

The archive contains cleaned/deduplicated files, denominator tables,
silver-labelled training data, validation supplements, classifier outputs,
attention outputs, and a `derived_data_manifest.json` with file sizes,
checksums, destination paths, and the git commit used for the archive.

## CPI / Corruption Perceptions Index PDFs

The CPI PDF reports used as contextual country-year corruption-perception data
are stored on Research Drive/SURF under:

```text
ASCOR-FMG-5580-RESPOND-news-data (Projectfolder)/victims-of-corruption-paper/CPI/
```

Parse those PDFs into a tidy country-year CSV with:

```bash
python3 extract_cpi_from_pdfs.py \
  --source webdav \
  --output output/cpi_country_year_scores.csv
```

The output contains `year`, `country`, `cpi_score`, `cpi_rank`, `source_pdf`,
`source_page`, and `extraction_method`. The script also writes an extraction log
next to the output CSV. Use the log to spot PDFs whose table layout needs manual
checking.

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

This workflow starts with reproducible random samples across country and year
among articles classified as political corruption, then fits multilingual
BERTopic models, labels topics with GPT 5.1 through the UvA LLM proxy, and
uses a notebook to inspect topic tables, example articles, and interactive
country/time visualizations. The topic outputs are intended as an inductive
appendix/discovery step before building final substantive variables. They can
help inspire later coding of victim visibility, victim type, corruption domain,
and case scope, but they are not treated as final measurement of those
variables.
