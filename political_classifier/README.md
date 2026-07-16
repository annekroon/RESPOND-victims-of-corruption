# Part 1: Political-Corruption Classifier

This folder contains the workflow for identifying which cleaned news articles are primarily about political corruption. It is deliberately separate from later victim-visibility analyses.

## Folder Structure

| Path | Purpose |
|---|---|
| `notebooks/01_clean_dedupe_data.ipynb` | Load raw country files, clean text, deduplicate, and write denominator tables |
| `notebooks/02_inspect_classifier_comparison.ipynb` | Inspect saved classifier comparison outputs and generate manuscript tables |
| `notebooks/03_analyze_political_corruption_attention.ipynb` | Analyze relative attention to political corruption over time |
| `scripts/00_download_source_workbook.py` | Download the reviewed source workbook from Research Drive/WebDAV |
| `scripts/01_clean_dedupe_data.py` | Reproducibly clean and deduplicate raw corruption-query files |
| `scripts/02_create_source_filtered_silver_seed.py` | Numbered wrapper for creating a fresh source-filtered silver-label annotation file |
| `scripts/03_label_silver_batch.py` | Numbered wrapper for UvA LLM proxy silver labelling |
| `scripts/04_compare_models.py` | Numbered wrapper for classifier comparison and validation |
| `scripts/05_train_final_classifier.py` | Numbered wrapper for final classifier training/scoring |
| `scripts/06_build_attention_outputs.py` | Execute the attention notebook to rebuild CSVs, figures, and LaTeX tables |
| `scripts/07_upload_outputs.py` | Upload manuscript tables and attention outputs |
| `scripts/08_archive_derived_data.py` | Archive expensive-to-recreate derived data |
| unnumbered scripts | Implementation helpers/backwards-compatible entry points used by the numbered pipeline |
| `scripts/reset_rebuild_outputs.py` | Maintenance helper for deleting old generated outputs before a clean rebuild |
| `scripts/filter_classified_outputs.py` | Maintenance helper for post-filtering already classified files |
| `scripts/restore_derived_data_from_webdav.py` | Restore archived derived data from Research Drive into the local pipeline folder |
| `scripts/merge_uk_validation_annotations.py` | Save a merged annotation file with the reviewed UK supplement |
| `tools/annotation_interface.py` | Streamlit UI for manual review |
| `archive/` | Older notebook versions kept for provenance |

## Final Classifier Decision

The final political-corruption classifier uses one combined LLM silver-labelled training set. The manually reviewed UK supplement is used only for validation, not for training.
The metrics below describe the last completed classifier run; after rebuilding
the source-filtered silver-labelled set, rerun the comparison notebook and
update these values from the generated tables.

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

For the clean end-to-end rebuild after source review, use:

```text
political_classifier/REBUILD_WORKFLOW.md
```

### Quick Rebuild Checklist After Source Review

Use this when the source inclusion workbook changes and the silver-labelled
training set should be rebuilt from scratch:

1. Download the reviewed source workbook through the WebDAV API:

   ```bash
   python3 political_classifier/scripts/00_download_source_workbook.py --overwrite
   ```

2. Delete old generated outputs:

   ```bash
   python3 political_classifier/scripts/reset_rebuild_outputs.py
   python3 political_classifier/scripts/reset_rebuild_outputs.py \
     --confirm DELETE_OLD_POLITICAL_CLASSIFIER_OUTPUTS
   ```

3. Clean and deduplicate the raw corruption-query corpus:

   ```bash
   python3 political_classifier/scripts/01_clean_dedupe_data.py --overwrite
   ```

4. Create a fresh source-filtered silver annotation input:

   ```bash
   python3 political_classifier/scripts/02_create_source_filtered_silver_seed.py \
     --overwrite \
     --country-targets Bulgaria:500,France:500,Hungary:500,Italy:500,Netherlands:500,Serbia:500,Sweden:500,Ukraine:500,United_Kingdom:500
   ```

5. Label the fresh silver set:

   ```bash
   nohup python3 -u political_classifier/scripts/03_label_silver_batch.py \
     --input /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/active_learning/silver_training_source_filtered_for_annotation.csv \
     --output /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/active_learning/silver_training_source_filtered_with_llm_suggestions.csv \
     --max-chars 3000 \
     --overwrite \
     > llm_silver_training_source_filtered.log 2>&1 &
   ```

6. Rerun classifier comparison on the included-source validation universe:

   ```bash
   python3 political_classifier/scripts/04_compare_models.py \
     --embedding-models intfloat/multilingual-e5-large \
     --batch-size 32 \
     --extra-human-validation /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/active_learning/uk_human_validation_reviewed.csv
   ```

7. Rerun `political_classifier/notebooks/02_inspect_classifier_comparison.ipynb`
   to regenerate classifier validation tables.

8. Score the full source-filtered corpus after accepting classifier performance:

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

9. Rebuild attention tables/figures:

   ```bash
   python3 political_classifier/scripts/06_build_attention_outputs.py
   ```

10. Upload updated tables and figures:

   ```bash
   python3 political_classifier/scripts/07_upload_outputs.py
   ```

11. Archive the updated derived data:

   ```bash
   python3 political_classifier/scripts/08_archive_derived_data.py \
     --groups source_inclusion cleaned_deduped silver_training_data classifier_comparison classifier_outputs attention_outputs
   ```

### 1. Clean And Dedupe

Run the reproducible script:

```bash
python3 political_classifier/scripts/01_clean_dedupe_data.py --overwrite
```

Use `political_classifier/notebooks/01_clean_dedupe_data.ipynb` only for
interactive inspection/debugging.

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

### 2. Create And Label Fresh Source-Filtered Silver Data

The current reproducible route starts the classifier training data from
scratch. The seed file is sampled from the cleaned corruption-query corpus after
applying the source-inclusion workbook. It does not use previous silver labels
or a provisional classifier.

Create the fresh source-filtered annotation input:

```bash
python3 political_classifier/scripts/02_create_source_filtered_silver_seed.py \
  --overwrite \
  --country-targets Bulgaria:500,France:500,Hungary:500,Italy:500,Netherlands:500,Serbia:500,Sweden:500,Ukraine:500,United_Kingdom:500
```

This writes:

```text
/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/active_learning/silver_training_source_filtered_for_annotation.csv
```

Label it with the UvA LLM proxy:

```bash
nohup python3 -u political_classifier/scripts/03_label_silver_batch.py \
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

The final analytical sample uses the source-inclusion workbook:

```text
/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/source_inclusion/political_corruption_all_sources_classified.xlsx
```

Only country/source rows with `conventional_journalism == Yes` are retained.
Rows marked `No`, missing, or anything else are excluded. Place the workbook at
the path above before rerunning classifier validation or attention outputs.

Run the final comparison on the included-source validation universe:

```bash
python3 political_classifier/scripts/04_compare_models.py \
  --embedding-models intfloat/multilingual-e5-large \
  --batch-size 32 \
  --extra-human-validation /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/active_learning/uk_human_validation_reviewed.csv
```

By default this reads:

```text
/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/active_learning/silver_training_source_filtered_with_llm_suggestions.csv
```

For a broader embedding comparison:

```bash
python3 political_classifier/scripts/04_compare_models.py \
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

Pass `--no-source-filter` only for legacy/unfiltered diagnostics.

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
python3 political_classifier/scripts/05_train_final_classifier.py \
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
nohup python3 -u political_classifier/scripts/05_train_final_classifier.py \
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

If the classifier has already been scored and only the source-inclusion scheme
changed, do **not** rerun the expensive embedding scoring. Instead, post-filter
the existing classified outputs:

```bash
python3 political_classifier/scripts/filter_classified_outputs.py
```

This writes filtered classified files to:

```text
/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/silver_classifier/classified_country_files_source_filtered/
```

The attention notebook also applies the same source filter when it loads
classified files, so this post-filtered directory is mainly a reproducible
archive/checkpoint.

### 6. Attention Over Time

After full scoring, run:

```text
political_classifier/notebooks/03_analyze_political_corruption_attention.ipynb
```

Section 3 of the notebook applies the source-inclusion workbook to the
classified political-corruption articles. The final analytical numerator is
therefore the classified political-corruption sample restricted to explicitly
included sources. The total-news denominator remains the separate country-period
NewsAPI count series; the available denominator is not source-specific.

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

It also writes source-filter diagnostics:

```text
political_corruption_source_filter_summary_by_country.csv
political_corruption_source_filter_decision_counts.csv
```

Regenerate the manuscript data-pipeline TikZ figure by rerunning the notebook
cell titled:

```text
# Method figure: final TikZ data-pipeline figure.
```

Upload generated attention figures and tables to Research Drive:

```bash
python3 political_classifier/scripts/07_upload_outputs.py
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

The same notebook also writes outlet/source descriptives for the classified
political-corruption corpus after source filtering:

```text
political_corruption_source_summary.csv
political_corruption_source_country_summary.csv
political_corruption_top_sources_by_country.csv
political_corruption_source_concentration_by_country.csv
latex/table_attention_top_sources_overall.tex
latex/table_attention_top_sources_by_country.tex
latex/table_attention_source_concentration_by_country.tex
```

The data-pipeline figure is also generated as LaTeX/TikZ so that it can be
included directly in the paper:

```text
figure_political_corruption_data_pipeline_tikz.tex
```

It shows the two separate NewsAPI routes: one route retrieves the
corruption-query article corpus, and the other route retrieves weekly
total-news counts used only as the denominator for relative attention.
Regenerate it by rerunning the notebook cell:

```text
# Method figure: final TikZ data-pipeline figure.
```

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
python3 political_classifier/scripts/07_upload_outputs.py --skip-attention-outputs
```

## Reproducibility Archive

GitHub should contain code, notebooks, and documentation. Large generated data
belong on Research Drive/WebDAV. After rerunning the workflow or updating
important outputs, archive the derived data with:

```bash
python3 political_classifier/scripts/08_archive_derived_data.py
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
| `source_inclusion` | Source inclusion/exclusion workbook and source-filter diagnostics |
| `classifier_outputs` | Final classifier predictions, source-filtered classifier files, summaries, selected threshold, embedding model, and model artifact |
| `classifier_comparison` | Validation comparison CSV outputs |
| `attention_outputs` | Attention CSVs, generated LaTeX attention tables, figures, and classifier manuscript tables |

The script writes and uploads:

```text
derived_data_manifest.json
```

The manifest records the selected files, Research Drive paths, file sizes,
SHA-256 checksums by default, archive time, and git commit. For a faster dry run:

```bash
python3 political_classifier/scripts/08_archive_derived_data.py --dry-run --no-checksum
```

To archive only selected groups:

```bash
python3 political_classifier/scripts/08_archive_derived_data.py \
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
