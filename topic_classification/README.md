# Descriptive Topic Modelling

This folder contains a separate exploratory BERTopic description of articles
already classified as political corruption. It is appendix material and a
sensitizing input for later qualitative codebook development. It is not a
validated corruption-type classifier and is not used for causal inference.

## Current Scientific Decision

The first production specification clustered multilingual article text
directly and then assigned 30 fine-grained clusters to six predefined
higher-order groups. Manual inspection showed that most clusters tracked
publication language, country, recurring personalities, and events more than a
shared substantive topic. That completed run remains in Research Drive as a
reproducible sensitivity analysis, but its six-group prevalence table must not
be interpreted as a distribution of corruption types.

The maintained model has a narrower purpose: produce a small and readable
description of recurring political-corruption coverage. GPT-5.1 first converts
each sampled article into a short English abstraction that preserves the
alleged practice, institutional setting, and response while removing names,
countries, outlets, dates, and case-specific details. A stability-selection
stage compares density-clustering specifications on repeated resamples and
retains the most stable solution satisfying predeclared descriptive adequacy
guardrails. The topic count is therefore an output rather than a presentation
target. No higher-order grouping is imposed.

## Input And Sampling

The upstream input is the completed source-filtered political-corruption
classifier run:

```text
/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/
  political_corruption_pipeline/silver_classifier/
    classifier_run_manifest.json
    classified_country_summary.csv
    selected_threshold.txt
    classified_country_files/*_classified.csv.gz
```

The sampler verifies the classifier threshold, country files, source-filter
policy, row counts, predictions, hashes, and Git provenance. It draws up to 50
articles from every non-empty `country x year` stratum and retains inverse
sampling-probability weights. The smaller sample is sufficient for a compact
descriptive model and keeps the auditable GPT abstraction stage tractable.

## Maintained Compact Workflow

| Step | Script | Purpose |
|---|---|---|
| 01 | `01_create_stratified_topic_sample.py` | Verified country-year sample |
| 02 | `02_create_descriptive_abstractions.py` | Language-neutral English case abstractions |
| 03 | `03_fit_descriptive_bertopic.py` | Stability-selected direct BERTopic solution |
| 04 | `04_label_descriptive_topics_with_llm.py` | Readable labels from neutral abstractions |
| 05 | `05_build_topic_visualizations.py` | Direct-topic country and time diagnostics |
| 06 | `06_build_descriptive_topic_review.py` | Table, diagnostics, and manual-review packet |
| 08 | `08_publish_descriptive_topic_outputs.py` | Publish an approved table, figures, and diagnostics |

The final stability-selected build applies the reviewed publication wording in
`manual_labels/final_stability_v1_topic_labels.csv`. The file records both the
expected GPT-5.1 label and its neutral publication label. A mismatch stops the
build rather than applying a reviewed label to a different topic.

Use the single maintained entry point rather than assembling these calls
manually:

```text
scripts/07_run_compact_descriptive_topic_model.sh
```

## Run The Final Candidate

After pulling the latest repository on `annecuda`:

```bash
cd ~/RESPOND-victims-of-corruption

python3 -m pip install -r topic_classification/requirements-topic.txt

export DATA_ROOT=/home/akroon/data/1t_storage/RESPOND-victims-of-corruption
export TMPDIR=/home/akroon/data/1t_storage/tmp
export HF_HOME=/home/akroon/data/1t_storage/huggingface_cache
export CUDA_VISIBLE_DEVICES=1
export LLMPROXY_MODEL=gpt-5.1

mkdir -p "$DATA_ROOT/topic_classification/logs"

LOG="$DATA_ROOT/topic_classification/logs/stability_selected_topic_model.log"
PIDFILE="$DATA_ROOT/topic_classification/logs/stability_selected_topic_model.pid"

nohup bash topic_classification/scripts/07_run_compact_descriptive_topic_model.sh \
  > "$LOG" 2>&1 &

echo $! > "$PIDFILE"
tail -f "$LOG"
```

The abstraction stage checkpoints every ten articles. The stability selector
compares 90 HDBSCAN candidates and five 80-percent resamples per candidate.
Rerunning the same
command reuses the verified sample, resumes unfinished GPT abstractions, and
reuses valid model and label stages. It does not touch classifier outputs,
human annotations, or the archived multilingual sensitivity model.

For a cheaper end-to-end smoke test before the default run, use a separate
five-per-stratum output tree with a smaller density threshold:

```bash
PER_COUNTRY_YEAR=5 TARGET_TOPICS=auto MIN_TOPIC_SIZE=10 \
  HDBSCAN_MIN_SAMPLES=3 CLUSTERER=hdbscan \
  CLUSTER_SELECTION_METHOD=leaf \
  bash topic_classification/scripts/07_run_compact_descriptive_topic_model.sh
```

These parameters are recorded in the manifests and produce different paths,
so the smoke test cannot overwrite the default 50-per-stratum run.

GPT-5.1 is the currently permitted UvA proxy model. GPT-5.6-terra was tested
but denied for the project key and must not be selected without a fresh access
test.

## Compact Specification

- Target population: source-eligible articles classified as political
  corruption at the final classifier threshold.
- Sample: up to 50 articles per non-empty country-year stratum.
- Abstraction: GPT-5.1, temperature 0, 18-45 English words.
- Removed from abstractions: names, countries, cities, outlets, dates,
  nationalities, quotations, exact amounts, and other case identifiers.
- Retained: alleged corrupt practice, institutional or sectoral setting, and
  the investigation, trial, sanction, reform, or response when central.
- Boundary handling: abstractions marked `boundary_or_unclear` are retained in
  the audit data but excluded from BERTopic.
- Embeddings: `intfloat/multilingual-e5-large`, using the already verified and
  cached model; all clustering inputs are English.
- BERTopic selection grid: UMAP neighborhood sizes 15, 30, and 50; HDBSCAN
  minimum cluster sizes 30, 40, 60, 80, and 120; `min_samples` values 2, 5,
  and 10; and both leaf and excess-of-mass extraction.
- Stability: each candidate is compared with five fits on 80% resamples using
  the adjusted Rand index among mutually assigned articles.
- Adequacy guardrails: 4--12 topics, at most 45% outliers, no topic exceeding
  40% of assigned articles, and country--topic NMI no higher than .25 in the
  balanced sample. Across refits, at least 35% of articles must receive a
  non-outlier assignment in both solutions. These bounds define an
  interpretable appendix solution but do not request an exact topic count.
- Selection: among adequate candidates, maximize mean resample adjusted Rand
  index, followed by the mutually assigned share, weighted cluster persistence,
  and relative validity; use baseline coverage and the more compact solution
  only as final tie-breakers.
- Topic representations: class-based TF-IDF, seed 42. If no candidate meets
  every guardrail, the workflow stops and no manuscript solution is produced.
- Reporting: direct BERTopic topics only; no forced higher-order taxonomy.
- Shares: inverse country-year weighted and calculated among non-outliers.

## Outputs To Inspect

```text
/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/
  topic_classification/
    political_corruption_descriptive_country_year_sample_50.csv.gz
    political_corruption_descriptive_country_year_sample_50_run_manifest.json
    political_corruption_descriptive_country_year_sample_50_english_abstracts.csv.gz
    political_corruption_descriptive_country_year_sample_50_english_abstracts_run_manifest.json
    bertopic_political_corruption_descriptive_hdbscan_stability_v1_sample_50/
      hdbscan_stability_candidates.csv
      hdbscan_stability_selection.json
      topic_info.csv
      document_topics.csv.gz
      topic_model/
      topic_labels_llm.csv
      topic_labels_llm_audit.jsonl
      visualizations/
        figure_topic_prevalence.pdf
        figure_topic_country_heatmap.pdf
        figure_topic_trends.pdf
        figure_topic_model_selection.pdf
        *.png
        *.html
      descriptive_outputs/
        descriptive_topic_summary.csv
        descriptive_topic_country_shares.csv
        descriptive_topic_year_shares.csv
        descriptive_topic_manual_review.csv
        descriptive_topic_diagnostics.json
        latex/
          descriptive_topic_manuscript_values.tex
          table_topic_descriptive_summary.tex
      descriptive_topic_output_manifest.json
      00_LATEST_DESCRIPTIVE_TOPIC_BUILD.txt
```

Open `00_LATEST_DESCRIPTIVE_TOPIC_BUILD.txt` first. Then manually review
`descriptive_topic_manual_review.csv`, which contains the exact neutral
abstractions used to label each topic.

## Acceptance Checks

The workflow deliberately does not publish manuscript files automatically.
Before promotion, inspect:

1. Whether examples within each topic describe a recognizable common pattern.
2. Whether the short label and summary fit those examples without adding an
   unstated taxonomy.
3. The largest weighted topic share. A single residual topic should not absorb
   most of the corpus.
4. The number of topics for which one publication country supplies at least
   80% of the sampled inlier abstractions. This leakage check is deliberately
   unweighted because the sample is balanced by country-year.
5. Unweighted normalized mutual information between country and topic. This is
   a leakage diagnostic, not an inferential test.
6. Outlier and `boundary_or_unclear` shares.

No numerical criterion proves substantive validity. The grid prevents the
workflow from selecting the fragmented 17-topic leaf result or the degenerate
two-topic excess-of-mass result solely because one diagnostic looks favorable.
Promotion still requires manual coherence review of the retained examples.

The manuscript appendix uses the descriptive topic table, country-composition
heatmap, and annual topic-composition figure. The prevalence and technical
model-selection figures remain in the published Research Drive archive. Static
figures are written as vector PDFs with embedded TrueType fonts and as 600-dpi
PNGs; the color choices are designed to remain distinguishable under common
forms of color-vision deficiency.

For an exact sensitivity refit of a candidate listed in
`hdbscan_stability_candidates.csv`, pass its UMAP neighborhood size as well as
its HDBSCAN settings. For example, the `u50`, minimum-size 80, `min_samples=5`
candidate is reproduced with:

```bash
PER_COUNTRY_YEAR=50 CLUSTERER=hdbscan UMAP_NEIGHBORS=50 \
  CLUSTER_SELECTION_METHOD=leaf TARGET_TOPICS=none \
  MIN_TOPIC_SIZE=80 HDBSCAN_MIN_SAMPLES=5 \
  bash topic_classification/scripts/07_run_compact_descriptive_topic_model.sh
```

The UMAP setting is encoded in the sensitivity output-directory name, so it
cannot silently reuse a fit made with a different neighborhood size.

`max_sample_country_share` is an unweighted leakage diagnostic for the balanced
sample. The country percentages printed under `top_countries` are inverse
country-year weighted estimates of corpus composition; they answer a different
question and therefore need not be equal.

## Publish An Approved Build

First run a dry validation of every table, figure, and provenance link:

```bash
TOPIC_DIR="$DATA_ROOT/topic_classification/bertopic_political_corruption_descriptive_hdbscan_stability_v1_sample_50"

python3 topic_classification/scripts/08_publish_descriptive_topic_outputs.py \
  --bertopic-dir "$TOPIC_DIR"
```

After the manual-review packet has been approved, add `--upload`. This publishes
the LaTeX table under `output/tables/topic_models`, the PDF and PNG figures
under `output/figures/topic_models`, and diagnostics and manifests under
`output/topic_models` on Research Drive.

## Interpretation Rules

- Call the procedure **LLM-assisted descriptive BERTopic modelling**.
- Do not describe the topics as validated corruption types.
- Do not use topic IDs as stable constructs across runs.
- Do not report weighted topic shares before manual review.
- Preserve prompts, raw responses, sampled examples, model names, hashes, and
  run manifests.
- Treat the earlier 30-topic/six-group run as an archived sensitivity analysis,
  not the manuscript result.

## Archived Sensitivity Workflow

The previous runner and its notebook/table helpers are retained under
`topic_classification/legacy/` only to document the archived multilingual
direct-text model:

```text
legacy/07_rerun_final_source_filtered_topic_solution.sh
```

It should not be rerun for the current descriptive analysis and its
`table_topic_higher_order_summary.tex` should not be inserted into the
manuscript as a distribution of corruption types.
