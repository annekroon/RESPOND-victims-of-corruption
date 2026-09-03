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

The maintained pilot has a narrower purpose: produce a small and readable
description of recurring political-corruption coverage. GPT-5.1 first converts
each sampled article into a short English abstraction that preserves the
alleged practice, institutional setting, and response while removing names,
countries, outlets, dates, and case-specific details. BERTopic then estimates
a compact number of direct descriptive topics from those abstractions. The
topic count is learned from the data rather than fixed for presentation. No
higher-order grouping is imposed.

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
| 03 | `03_fit_descriptive_bertopic.py` | Data-driven direct BERTopic solution |
| 04 | `04_label_descriptive_topics_with_llm.py` | Readable labels from neutral abstractions |
| 05 | `05_build_topic_visualizations.py` | Direct-topic country and time diagnostics |
| 06 | `06_build_descriptive_topic_review.py` | Table, diagnostics, and manual-review packet |

Use the single maintained entry point rather than assembling these calls
manually:

```text
scripts/07_run_compact_descriptive_topic_model.sh
```

## Run The Pilot

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

LOG="$DATA_ROOT/topic_classification/logs/compact_descriptive_topic_model.log"
PIDFILE="$DATA_ROOT/topic_classification/logs/compact_descriptive_topic_model.pid"

nohup bash topic_classification/scripts/07_run_compact_descriptive_topic_model.sh \
  > "$LOG" 2>&1 &

echo $! > "$PIDFILE"
tail -f "$LOG"
```

The abstraction stage checkpoints every ten articles. Rerunning the same
command reuses the verified sample, resumes unfinished GPT abstractions, and
reuses valid model and label stages. It does not touch classifier outputs,
human annotations, or the archived multilingual sensitivity model.

For a cheaper end-to-end smoke test before the default run, use a separate
five-per-stratum output tree with a smaller density threshold:

```bash
PER_COUNTRY_YEAR=5 TARGET_TOPICS=auto MIN_TOPIC_SIZE=10 \
  HDBSCAN_MIN_SAMPLES=3 CLUSTERER=hdbscan \
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
- BERTopic: HDBSCAN leaf clustering with minimum topic size 40 and
  `min_samples = 10`, followed by BERTopic's automatic topic reduction. The
  topic count is not fixed. Topic representations use class-based TF-IDF;
  seed 42.
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
    bertopic_political_corruption_descriptive_hdbscan_auto_sample_50/
      topic_info.csv
      document_topics.csv.gz
      topic_model/
      topic_labels_llm.csv
      topic_labels_llm_audit.jsonl
      visualizations/
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

No single threshold proves validity. Promotion requires substantive manual
coherence plus clearly lower country dependence than the archived direct-text
specification.

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
