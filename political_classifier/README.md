# Part 1: Political-Corruption Classifier

This folder contains the workflow for identifying which cleaned news articles are primarily about political corruption. It is deliberately separate from later victim-visibility analyses.

## Folder Structure

| Path | Purpose |
|---|---|
| `notebooks/01_clean_dedupe_data.ipynb` | Load raw country files, clean text, deduplicate, and write denominator tables |
| `notebooks/02_inspect_classifier_comparison.ipynb` | Inspect saved classifier comparison outputs and generate manuscript tables |
| `notebooks/03_analyze_political_corruption_attention.ipynb` | Analyze relative attention to political corruption over time |
| `scripts/create_silver_label_batch.py` | Create targeted source files for the silver-labelled training set |
| `scripts/label_silver_batch.py` | Send silver-training-set source files to the UvA LLM proxy |
| `scripts/compare_models.py` | Reproducible classifier comparison and validation |
| `scripts/train_final_classifier.py` | Train/evaluate the final classifier and optionally score all cleaned country files |
| `scripts/upload_manuscript_tables.py` | Upload generated LaTeX tables to Research Drive |
| `scripts/archive_derived_data_to_webdav.py` | Archive expensive-to-recreate derived data and a manifest to Research Drive |
| `scripts/restore_derived_data_from_webdav.py` | Restore archived derived data from Research Drive into the local pipeline folder |
| `scripts/merge_uk_validation_annotations.py` | Save a merged annotation file with the reviewed UK supplement |
| `tools/annotation_interface.py` | Streamlit UI for manual review |
| `archive/` | Older notebook versions kept for provenance |

## Final Classifier Decision

The final political-corruption classifier uses one combined LLM silver-labelled training set. The manually reviewed UK supplement is used only for validation, not for training.

| Metric | Value |
|---|---|
| Label source | Silver-labelled training set |
| Training rows | `3,982` |
| Embedding model | `intfloat/multilingual-e5-large` |
| Classifier | Balanced logistic regression |
| Threshold | `0.40` |
| Validation rows | `502` |
| Political-corruption support | `141` |
| Accuracy | `0.863` |
| Political precision | `0.731` |
| Political recall | `0.809` |
| Political F1 | `0.768` |
| Macro F1 | `0.835` |
| Weighted F1 | `0.865` |
| Predicted positive rate on validation | `0.311` |

## Workflow

Run commands from the repository root on `annecuda`.

The notebooks include a small bootstrap cell that finds the repository root and adds it to `sys.path`. This keeps imports such as `from config import RD_BASE_DIR` and `from dataloader import ...` working even though the notebooks live in `political_classifier/notebooks/`.

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

Create targeted files for the silver-labelled training set:

```bash
python3 political_classifier/scripts/create_silver_label_batch.py \
  --country-targets Sweden:540,United_Kingdom:540,Ukraine:540,Netherlands:420,Serbia:360,Hungary:180,Bulgaria:180,Italy:120,France:120
```

Label each file with the UvA LLM proxy:

```bash
nohup python3 -u political_classifier/scripts/label_silver_batch.py \
  --input /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/active_learning/active_learning_batch_2_for_annotation.csv \
  --output /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/active_learning/active_learning_batch_2_with_llm_suggestions.csv \
  --max-chars 3000 \
  > llm_silver_label_batch_2.log 2>&1 &
```

### 3. UK Validation Supplement

The original balanced human validation set had low UK positive support. A small UK supplement was therefore manually reviewed and is used as validation-only data. If you need to inspect or edit that reviewed file, use Streamlit:

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
/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/classifier_comparison/
```

Then inspect:

```text
political_classifier/notebooks/02_inspect_classifier_comparison.ipynb
```

This notebook generates manuscript-ready LaTeX classifier tables. The final model is reported as trained on one silver-labelled training set rather than as separate data-collection batches.

The inspection notebook intentionally reads only `classifier_comparison/`. If old folders such as `classifier_comparison_uk_calibration/` still exist on disk, they are ignored. You may archive or delete them manually after confirming you no longer need them.

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
  > silver_classifier_final_scoring.log 2>&1 &
```

Monitor:

```bash
tail -f silver_classifier_final_scoring.log
```

Outputs are written to:

```text
/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/silver_classifier/
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

Upload generated attention figures and tables to Research Drive:

```bash
python3 political_classifier/scripts/upload_attention_outputs.py
```

Default Research Drive destinations:

```text
ASCOR-FMG-5580-RESPOND-news-data (Projectfolder)/victims-of-corruption-paper/output/figures/attention/
ASCOR-FMG-5580-RESPOND-news-data (Projectfolder)/victims-of-corruption-paper/output/tables/attention/
```

The attention notebook writes LaTeX tables from the saved CSV outputs into:

```text
/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/attention_tables/latex/
```

The upload script sends both the CSV intermediates and the generated LaTeX files.
The main method/descriptive attention table is:

```text
table_attention_corpus_construction_country_summary.tex
```

It is generated from the saved attention CSVs and reports, by country, the
cleaned/deduplicated corruption-query corpus size, the final classified
political-corruption corpus size, the political-corruption share within the
query corpus, the total-news denominator, and the political-corruption share of
total news.

The data-pipeline figure is also generated as LaTeX/TikZ so that it can be
included directly in the paper:

```text
figure_political_corruption_data_pipeline_tikz.tex
```

It shows the two separate NewsAPI routes: one route retrieves the
corruption-query article corpus, and the other route retrieves weekly
total-news counts used only as the denominator for relative attention.

Recommended manuscript figures:

```latex
% Method/data section; requires \usepackage{tikz}
% and \usetikzlibrary{arrows.meta, positioning}
\input{tables/attention/latex/figure_political_corruption_data_pipeline_tikz}

% Main results section
\includegraphics[width=\textwidth]{figures/attention/political_corruption_relative_attention_total_news_month_small_multiples.png}

% Appendix
\includegraphics[width=\textwidth]{figures/attention/political_corruption_absolute_volume_month_stacked.png}
```

A draft results section and appendix figure text are available in:

```text
docs/results_political_corruption_attention.tex
```

## Manuscript Tables

The comparison notebook writes classifier validation tables to:

```text
/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/manuscript_tables/
```

The reproducible table outputs are generated by
`political_classifier/notebooks/02_inspect_classifier_comparison.ipynb`.
Use the `table_pc_classifier*.tex` files in Overleaf, for example:

```latex
\input{tables/table_pc_classifier_comparison_main}

% Appendix
\input{tables/table_pc_classifier_comparison_appendix}
\input{tables/table_pc_classifier_threshold_sweep_appendix}
\input{tables/table_pc_classifier_country_validation_appendix}
```

Upload them to Research Drive:

```bash
python3 political_classifier/scripts/upload_manuscript_tables.py
```

## Reproducibility Archive

GitHub should contain code, notebooks, and documentation. Large generated data
belong on Research Drive/WebDAV. After rerunning the workflow or updating
important outputs, archive the derived data with:

```bash
python3 political_classifier/scripts/archive_derived_data_to_webdav.py
```

To restore the archived outputs on a fresh or cleaned machine:

```bash
python3 political_classifier/scripts/restore_derived_data_from_webdav.py
```

This reads `derived_data_manifest.json` from Research Drive, downloads selected
files into the expected local `political_corruption_pipeline` structure, and
verifies SHA-256 checksums when they are present.

Default destination:

```text
ASCOR-FMG-5580-RESPOND-news-data (Projectfolder)/victims-of-corruption-paper/derived_data/political_classifier/
```

The archive script uploads these groups:

| Group | Contents |
|---|---|
| `cleaned_deduped` | Cleaned/deduplicated country files, minimal combined file, denominator tables |
| `silver_training_data` | LLM-labelled silver-training-set source files and annotation input files |
| `validation_data` | UK validation review files used as validation-only supplement |
| `classifier_outputs` | Final classifier predictions, summaries, selected threshold, embedding model, and model artifact |
| `classifier_comparison` | Validation comparison CSV outputs |
| `attention_outputs` | Attention CSVs, generated LaTeX attention tables, figures, and classifier manuscript tables |

The script writes and uploads:

```text
derived_data_manifest.json
```

The manifest records the selected files, Research Drive paths, file sizes,
SHA-256 checksums by default, archive time, and git commit. For a faster dry run:

```bash
python3 political_classifier/scripts/archive_derived_data_to_webdav.py --dry-run --no-checksum
```

To archive only selected groups:

```bash
python3 political_classifier/scripts/archive_derived_data_to_webdav.py \
  --groups classifier_outputs attention_outputs
```

To restore only selected groups:

```bash
python3 political_classifier/scripts/restore_derived_data_from_webdav.py \
  --groups classifier_outputs attention_outputs
```

To preview a restore without downloading:

```bash
python3 political_classifier/scripts/restore_derived_data_from_webdav.py --dry-run
```
