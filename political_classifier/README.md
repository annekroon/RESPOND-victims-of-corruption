# Political-Corruption Classifier

This folder contains Part 1 of the paper workflow: identifying articles that
are primarily about political corruption and measuring their attention relative
to all news coverage. Later victim, frame, location, and accused-actor coding is
kept in `content-classification/`. Exploratory topic modelling is kept in
`topic_classification/`.

## Start Here

The only canonical execution guide is:

```text
political_classifier/REBUILD_WORKFLOW.md
```

It contains the exact numbered commands, long-run `nohup` commands, checkpoints,
output paths, upload command, and immutable archive/restore commands. Keeping
one runbook avoids contradictory command lists and stale hard-coded results.

## Active Layout

| Path | Role |
|---|---|
| `scripts/00_download_source_workbook.py` | Download reviewed source decisions through WebDAV |
| `scripts/01_clean_dedupe_data.py` | Chunked within-country cleaning and exact deduplication |
| `scripts/02_create_source_filtered_corpus.py` | Build the full source-eligible corpus |
| `scripts/03_prepare_classifier_training_sample.py` | Draw fresh silver candidates and exclude human-benchmark overlap |
| `scripts/04_label_silver_batch.py` | GPT-5.1 translation/silver labels with manifests and audit JSONL |
| `scripts/05_compare_models.py` | Threshold calibration and held-out evaluation |
| `scripts/06_train_final_classifier.py` | Refit, validate, and optionally score the full corpus |
| `scripts/07_build_attention_outputs.py` | Rebuild attention CSVs, figures, tables, Figure 1, and LaTeX values |
| `scripts/08_upload_outputs.py` | Upload only a current validated build |
| `scripts/09_archive_derived_data.py` | Archive an immutable, checksummed Research Drive snapshot |
| `scripts/_impl/` | Implementations behind the stable numbered entry points |
| `tools/` | Manual annotation, reset, restore, and one-off migration utilities |
| `notebooks/02_inspect_classifier_comparison.ipynb` | Optional comparison inspection |

Production does not execute notebooks. The remaining comparison notebook is an
optional read-only inspection layer over outputs written by the scripts.

## Methodological Contract

1. Clean and deduplicate each country independently. Production deduplication
   uses exact URI and normalized-text hashes; prefix-based near-deduplication is
   disabled.
2. Remove a country-source pair only when the reviewed workbook explicitly says
   `conventional_journalism == No`; retain and audit all other pairs.
3. Draw a new country-balanced silver-training set from that eligible corpus.
4. Exclude URI and normalized-text overlap with all human benchmark records.
5. Fit the pre-specified E5-large plus class-balanced logistic regression model.
6. Select its threshold on a fixed 40% country/class-stratified calibration
   partition and report its performance once on the remaining 60% held-out
   partition.
7. Score the complete source-filtered country files with that frozen model and
   threshold.
8. Aggregate numerator and total-news denominator from compatible weekly bins.

Human five-fold CV models are diagnostics. They are not the final model and
must not be described as training the production classifier on human labels.

## Canonical Data Locations

Pipeline root on `annecuda`:

```text
/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/
```

Reviewed source workbook:

```text
source_inclusion/political_corruption_all_sources_classified.xlsx
```

Cleaned/deduplicated country files:

```text
{country}_cleaned_deduped.csv.gz
```

Full source-filtered country files:

```text
cleaned_deduped_source_filtered/{country}_cleaned_deduped_source_filtered.csv.gz
```

Canonical final classified country files:

```text
silver_classifier/classified_country_files/{country}_classified.csv.gz
```

The final political-corruption analytical sample is the subset where
`pred_political_corruption == 1` in those files. There is deliberately no second
post-hoc source-filtered classifier directory in the active workflow.

## Current Results Without Hand-Copied Numbers

Do not keep current Ns or performance metrics in this README. After steps 05-07,
the authoritative values are generated from the run artifacts:

```text
classifier_comparison/best_model_results.csv
classifier_comparison/selected_threshold.txt
classifier_comparison/classifier_comparison_run_manifest.json
silver_classifier/classified_country_summary.csv
silver_classifier/classifier_run_manifest.json
attention_tables/political_corruption_corpus_construction_country_summary.csv
attention_tables/latex/political_corruption_manuscript_values.tex
manuscript_tables/manuscript_output_manifest.json
```

The Method section inputs `political_corruption_manuscript_values.tex`, so its
counts, threshold, split sizes, and held-out metrics update with the same run as
the tables and Figure 1.

## Manuscript Artifacts

Step 07 generates:

```text
manuscript_tables/table_pc_classifier_comparison_main.tex
manuscript_tables/table_pc_classifier_comparison_appendix.tex
manuscript_tables/table_pc_classifier_threshold_sweep_appendix.tex
manuscript_tables/table_pc_classifier_country_validation_appendix.tex
attention_tables/latex/table_attention_corpus_construction_country_summary.tex
attention_tables/latex/figure_political_corruption_data_pipeline_tikz.tex
attention_tables/latex/political_corruption_manuscript_values.tex
attention_figures/political_corruption_relative_attention_total_news_month_small_multiples.png
attention_figures/political_corruption_absolute_volume_month_stacked.png
```

The main classifier table contains only the selected silver model's held-out
performance. The full appendix comparison marks human five-fold CV rows as
diagnostics. The threshold appendix table is calibration-only, and the country
appendix table is held-out. A ready-to-input appendix fragment is tracked at
`docs/appendix_political_corruption.tex`.

Step 08 flattens the internal `attention_tables/latex/` directory when
publishing. The canonical latest Research Drive paths are therefore
`output/tables/attention/*.tex`, while the three tracked manuscript fragments
are published to `output/manuscript/`. Start with
`output/00_LATEST_MANUSCRIPT_BUILD.txt`; timestamped historical runs belong only
under the immutable archive.

Generated tables use `\scriptsize` and `adjustbox` with `max width`, so they are
not enlarged to fill the page. The Overleaf preamble needs:

```tex
\usepackage{booktabs}
\usepackage{adjustbox}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning}
```

## Source Workbook And Annotation Data

The source workbook is a reviewed input, not a generated classifier result.
Human political-corruption annotations, the UK validation supplement, and later
content-codebook development samples must be preserved across rebuilds. The
reset helper deletes generated outputs only and supports a dry run:

```bash
python3 political_classifier/tools/reset_rebuild_outputs.py
```

Always read the dry-run list before using its explicit confirmation argument.

## Research Drive Reproducibility Archive

Step 09 writes immutable snapshots below:

```text
ASCOR-FMG-5580-RESPOND-news-data (Projectfolder)/
victims-of-corruption-paper/derived_data/political_classifier/runs/
```

Each snapshot includes a manifest with Git commit, paths, sizes, and SHA-256
checksums. Restore the latest or a named version with
`tools/restore_derived_data_from_webdav.py`; see the canonical runbook for the
commands.
