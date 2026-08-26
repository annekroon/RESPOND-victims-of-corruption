# Topic Classification And Discovery

This folder contains the exploratory workflow for mapping human-interpretable
topics among articles already classified as primarily discussing political
corruption.

The core idea is to avoid fitting one topic model to the full news corpus at
once. The corpus is large and uneven across countries and time, so the first
step is a reproducible stratified sample across `country x year` among
political-corruption articles. The second step fits a multilingual BERTopic
model with enough fine-grained topics to preserve country and time variation.
The third step asks
GPT 5.1, through the same UvA LLM proxy used in Part 1, to name and interpret
the topics. The final step exports interactive visualizations for cross-country
and over-time topic patterns.

## Folder Structure

| Path | Purpose |
|---|---|
| `scripts/01_create_stratified_topic_sample.py` | Create balanced random samples across country and year |
| `scripts/02_fit_multilingual_bertopic.py` | Fit BERTopic on a sample and export document-topic assignments |
| `scripts/03_label_topics_with_llm.py` | Ask GPT 5.1 to create human-readable topic labels and coding rules |
| `scripts/04_group_topics_with_llm.py` | Ask GPT 5.1 to assign fine-grained topics to higher-order topics |
| `scripts/05_build_topic_visualizations.py` | Build interactive Plotly country/time topic graphs |
| `scripts/06_publish_topic_archive_to_webdav.py` | Package final topic outputs, code, manifests, and manuscript tables for Research Drive/WebDAV archiving |
| `scripts/_impl/` | Internal helper code used by the numbered scripts |
| `notebooks/01_inspect_topic_results.ipynb` | Read final outputs and inspect topic tables, examples, and country/time graphs |
| `requirements-topic.txt` | Optional extra dependencies for BERTopic |

## Recommended Design

Main sample/model:

- Input: final classified country files created by
  `political_classifier/scripts/06_train_final_classifier.py --score-corpus`.
  Those files should be based on the cleaned, deduplicated, source-filtered
  corruption-query corpus from Part 1.
- Filter: `pred_political_corruption == 1`.
- Sampling: balanced across `country x year`, with `analysis_weight` saved so
  visualizations can recover weighted article counts and shares.
- Topic model: multilingual BERTopic using `intfloat/multilingual-e5-large`.
- Final raw-topic specification: `political_corruption_country_year_sample_200.csv.gz`
  with `--min-topic-size 10 --nr-topics auto`, producing 67 non-outlier
  lower-level BERTopic topics in
  `bertopic_political_corruption_200_min10`.
- Interpretability: fit a granular raw topic model first, then use GPT 5.1 to
  label the discovered clusters inductively from representative documents and
  topic keywords. GPT 5.1 then groups the fine-grained topics into a smaller
  set of higher-order topics for manuscript interpretation.
- Visualizations: interactive country-topic heatmap, stacked topic shares over
  time, and faceted country-over-time topic trends.

BERTopic is useful for discovering structure, but it is not the main
classification workflow. In this project, topic modelling is a separate,
inductive appendix step used to inspect recurring themes and to inspire later
substantive coding decisions, especially around victim visibility and
corruption type. Topic clusters are not the same as validated corruption-type
labels.

## Install Optional Topic Dependencies

From the repository root:

```bash
python3 -m pip install -r topic_classification/requirements-topic.txt
```

On `annecuda`, keep Hugging Face and temp files on the large disk:

```bash
TMPDIR=/home/akroon/data/1t_storage/tmp \
HF_HOME=/home/akroon/data/1t_storage/huggingface_cache \
python3 -m pip install -r topic_classification/requirements-topic.txt
```

## 1. Create A Political-Corruption Sample

This reads the classified country files created by
`political_classifier/scripts/06_train_final_classifier.py --score-corpus` and
keeps only rows with `pred_political_corruption == 1`.

For the current manuscript rerun, the classified inputs should be the
source-filtered classifier outputs:

```text
/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/
  political_corruption_pipeline/silver_classifier/classified_country_files/
```

These files are produced directly by the final classifier from the canonical
cleaned/deduplicated/source-filtered country files.

```bash
python3 topic_classification/scripts/01_create_stratified_topic_sample.py \
  --source classified \
  --classified-dir /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/silver_classifier/classified_country_files \
  --political-only \
  --per-country-year 200 \
  --output-name political_corruption_source_filtered_country_year_sample_200.csv.gz
```

If a country-year has fewer than the requested number of rows, all available
rows in that stratum are retained. The output includes `analysis_weight`,
`stratum_total_rows`, and `stratum_sample_rows`, which are used by the
visualization script.

For a reproducibility rerun from the archived Research Drive/WebDAV classifier
outputs, use the `classified-webdav` source. This reads the archived
`*_classified.csv.gz` files from:

```text
ASCOR-FMG-5580-RESPOND-news-data (Projectfolder)/victims-of-corruption-paper/derived_data/political_classifier/classifier_outputs/classified_country_files/
```

The command uses the same WebDAV credentials as the rest of the repository
(`config_local.py`, or `RD_USER`/`RD_PASS` environment variables):

```bash
python3 topic_classification/scripts/01_create_stratified_topic_sample.py \
  --source classified-webdav \
  --political-only \
  --per-country-year 200 \
  --output-name political_corruption_country_year_sample_200.csv.gz
```

If the cleaned/deduplicated/source-filtered country files need to be sampled
directly before political-corruption classification, use
`--source source-filtered` or `--source source-filtered-webdav`. For the topic
model used in the appendix, use the classified source with `--political-only`.

## 2. Fit Multilingual BERTopic

Small test run:

```bash
CUDA_VISIBLE_DEVICES=1 \
python3 topic_classification/scripts/02_fit_multilingual_bertopic.py \
  --sample /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/topic_classification/political_corruption_country_year_sample.csv.gz \
  --output-dir /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/topic_classification/bertopic_political_corruption_test \
  --max-docs 2000 \
  --min-topic-size 25 \
  --nr-topics auto
```

Final manuscript run on the source-filtered sample:

```bash
TMPDIR=/home/akroon/data/1t_storage/tmp \
HF_HOME=/home/akroon/data/1t_storage/huggingface_cache \
CUDA_VISIBLE_DEVICES=1 \
nohup python3 -u topic_classification/scripts/02_fit_multilingual_bertopic.py \
  --sample /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/topic_classification/political_corruption_source_filtered_country_year_sample_200.csv.gz \
  --output-dir /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/topic_classification/bertopic_political_corruption_source_filtered_200_min10 \
  --min-topic-size 10 \
  --nr-topics auto \
  > bertopic_political_corruption_source_filtered_200_min10.log 2>&1 &
```

Main outputs:

```text
topic_info.csv
document_topics.csv.gz
topic_model/
run_manifest.json
```

`run_manifest.json` records the command-line arguments, git commit, Python
version, package versions, input/output checksums, random seed, embedding model,
and relevant environment variables. Treat this file as part of the model output.

### Final Topic-Number Choice

The current final manuscript appendix should use
`bertopic_political_corruption_source_filtered_200_min10`, estimated on the
classified political-corruption articles after the final source filter has been
applied. The specification remains the same as the previous final run:
stratified sampling up to 200 articles per country-year stratum and
`min_topic_size = 10` with `nr_topics = auto`. This is the preferred
specification because it matches the substantive goal of the topic model:
discover fine-grained, country- and time-varying patterns first, then use GPT
5.1 to aggregate those discovered topics into a smaller set of interpretable
higher-order topics.

Alternative specifications were inspected but rejected for the appendix:

| Output directory | Non-outlier topics | Assessment |
|---|---:|---|
| `bertopic_political_corruption` | 7 | Too coarse; collapses distinct scandals, countries, and institutions into broad generic topics. |
| `bertopic_political_corruption_granular` | 8 | Misleading directory name; also too coarse for the intended lower-level topic inventory. |
| `bertopic_political_corruption_200_topics30` | 19 | More interpretable than the 7/8-topic runs, but still merges several event-, country-, and institution-specific patterns. |
| `bertopic_political_corruption_200_min10` | 67 | Previous final run before the final source-filtered classifier outputs were adopted. |
| `bertopic_political_corruption_source_filtered_200_min10` | rerun-dependent | Current final choice; same specification as the 67-topic run, but estimated on source-filtered classified outputs. |

The many-topic BERTopic solution should therefore be understood as the
lower-level inventory, not the final substantive typology. The
manuscript-facing typology is the GPT-assisted higher-order topic grouping. The
lower-level topics remain useful because they show which events, countries, and
issue areas compose each higher-order topic.

### One-Command Source-Filtered Final Rerun

After the source-filtered classified outputs exist, the full final topic-model
solution can be rerun with:

```bash
topic_classification/scripts/07_rerun_final_source_filtered_topic_solution.sh
```

This creates the source-filtered topic sample, fits the final BERTopic
specification, labels the fine-grained topics with GPT 5.1, groups them into
higher-order topics, executes the inspection notebook, and regenerates the CSV
and LaTeX tables under:

```text
/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/
  topic_classification/bertopic_political_corruption_source_filtered_200_min10/inspection_tables/
```

To also upload the regenerated LaTeX tables to Research Drive:

```bash
topic_classification/scripts/07_rerun_final_source_filtered_topic_solution.sh --upload-tables
```

## 3. Label Topics With GPT 5.1

This uses the same `LLMPROXY_BASE_URL`, `LLMPROXY_API_KEY`, and
`LLMPROXY_MODEL` settings as the political-corruption classifier. By default,
`LLMPROXY_MODEL` is `gpt-5.1`.

```bash
python3 topic_classification/scripts/03_label_topics_with_llm.py \
  --bertopic-dir /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/topic_classification/bertopic_political_corruption \
  --model gpt-5.1
```

Output:

```text
topic_labels_llm.csv
topic_labels_llm_audit.jsonl
topic_labels_run_manifest.json
```

Each row includes an inductive topic label, short chart label, topic summary,
inclusion rule, exclusion rule, country/event-specific flag, cross-country
comparability rating, label rationale, and confidence score. GPT 5.1 is used
only to interpret and name the discovered BERTopic clusters; it is not asked to
apply a predefined corruption-type taxonomy.

The audit file contains one JSON record per labelled topic with the prompt,
selected examples, raw LLM response, parsed response, model name, temperature,
and prompt version. This is the archival source for reviewing what the LLM saw.

## 4. Group Fine-Grained Topics Into Higher-Order Topics

For a many-topic solution, keep the fine-grained BERTopic clusters, then ask GPT
5.1 to assign those discovered topics to higher-order topics. These
higher-order topics describe how corruption is organized in the news coverage rather than
claiming to measure objective corruption types.

Current higher-order topics:

| Higher-order topic | Meaning |
|---|---|
| `individualized_elite_scandal` | Named politicians, leaders, trials, accusations, scandals, or personal misconduct |
| `systemic_institutional_corruption` | Broader institutional dysfunction, governance crisis, state capture, anti-corruption politics, or abuse of power |
| `transnational_investigative_corruption` | Cross-border probes, international investigations, foreign-linked cases, offshore money, sanctions, or investigative journalism |
| `boundary_or_nonpolitical_cases` | Corruption/scandal language that is less clearly centered on political corruption by public officials |
| `local_sectoral_corruption` | Municipal, public-service, school, housing, charity, sport, public company, or other sector-specific cases |
| `electoral_party_finance_scandal` | Campaign finance, party funding, vote manipulation, electoral control, or election-centered corruption allegations |

```bash
python3 topic_classification/scripts/04_group_topics_with_llm.py \
  --bertopic-dir /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/topic_classification/bertopic_political_corruption_200_min10 \
  --model gpt-5.1 \
  --target-groups 12
```

`--target-groups` is kept for command compatibility, but the higher-order topic set is fixed
by the script. Each fine-grained topic is assigned to exactly one higher-order topic, with an
LLM rationale.

Outputs:

```text
topic_groups_llm.csv
topic_group_summaries_llm.csv
topic_groups_llm_audit.json
topic_groups_run_manifest.json
```

The group audit file stores the full higher-order-topic prompt, raw LLM response,
parsed assignments, fixed higher-order topic definitions, model name, temperature, and prompt
version.

## 5. Inspect Final Results In A Notebook

Preferred inspection workflow:

```text
topic_classification/notebooks/01_inspect_topic_results.ipynb
```

The notebook reads the final `topic_info.csv`, `document_topics.csv.gz`, and
`topic_labels_llm.csv` files, and reads `topic_groups_llm.csv` when available.
It shows topic/group-size tables, GPT summaries, example articles, a
country-topic heatmap, topic shares over time, and faceted country trends. It
also exports the summary tables under `inspection_tables/`.

By default, the notebook plots `topic_group_short_label`, the higher-order
topic assigned by GPT-5.1. Set `ANALYSIS_LABEL_COLUMN = "topic_short_label"`
in the first code cell to inspect the fine-grained BERTopic topics directly.
It also includes lift diagnostics and country-specificity checks to evaluate
whether the discovered topics capture variation across countries and over time.

The final export cell writes journal/appendix-ready tables under:

```text
inspection_tables/latex/
```

Recommended manuscript tables:

| File | Use |
|---|---|
| `table_topic_higher_order_summary.tex` | Main appendix table summarizing the higher-order topics, weighted shares, main contributing countries, and substantive interpretation |
| `table_all_topics_llm_higher_order_topics.tex` | Continued appendix inventory listing every fine-grained BERTopic topic with its LLM-assigned higher-order topic, weighted articles / share, largest contributing country name, and a broader short interpretive summary |

The LaTeX topic inventory is intentionally compact and uses `longtable` so that
it continues across pages as one table instead of floating away from the topic
appendix. `Weighted articles / share` means the estimated number of
political-corruption articles represented by the topic after applying
country-year sampling weights, followed by the topic's weighted share of the
sampled political-corruption corpus. The appendix table reports only the name
of the largest contributing country, without within-topic country percentages,
so the interpretive summary column remains readable. Full GPT-5.1 topic descriptions,
inclusion rules, exclusion rules, assignment rationales,
country/event-specificity assessments, and cross-country comparability notes
remain in the CSV exports under
`inspection_tables/`, especially `topic_group_assignment_explanation.csv`.
Use those CSV files as the full audit/codebook version and the LaTeX files as
the journal appendix version.

The manuscript preamble needs `booktabs`, `tabularx`, and `longtable` for the
generated appendix tables.

## 6. Optional: Export Standalone HTML Visualizations

```bash
python3 topic_classification/scripts/05_build_topic_visualizations.py \
  --bertopic-dir /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/topic_classification/bertopic_political_corruption \
  --top-n 12 \
  --time-unit year
```

Outputs are written under:

```text
visualizations/
country_topic_heatmap.html
topic_shares_over_time.html
country_topic_trends.html
topic_weighted_totals.csv
```

## Interpretation Notes

- Use multilingual embeddings (`intfloat/multilingual-e5-large` by default) so
  documents from different languages can share a semantic space.
- Inspect topic composition by country and time. A topic that is mostly one
  country-language may be a real country-specific issue, a language artifact, or
  both.
- Do not force the raw BERTopic model to be too small. A good default is
  `--min-topic-size 25 --nr-topics auto`; if topics remain too broad, try a
  larger sample and `--min-topic-size 10`.
- If the topic labels collapse into vague categories such as `elite corruption`
  or `corruption investigations`, refit a more granular BERTopic model before
  interpreting.
- The over-time and cross-country plots are weighted back to the political
  corruption article counts within each sampled country-year stratum.
- Treat topic IDs as unstable. Save the model, sample, random seed, and
  exported topic info together.
- Use topic outputs to build a small hand-coded validation set for corruption
  type. The final corruption-type labels should not rely only on unsupervised
  topic IDs.

## Reproducibility And Archiving

The workflow is reproducible in two different senses:

1. **Published-result reproducibility:** preserve the exact files used in the
   paper so readers can verify the topic tables, labels, and plots.
2. **Full rerun reproducibility:** rerun sampling, embeddings, BERTopic, and LLM
   interpretation from source data. This is harder because package versions,
   Hugging Face model files, GPU libraries, and hosted LLM behavior can change.

For a journal submission, archive the published-result artifacts as the source
of truth:

```text
political_corruption_country_year_sample*.csv.gz
political_corruption_country_year_sample*_strata.csv
political_corruption_country_year_sample*_run_manifest.json
topic_info.csv
document_topics.csv.gz
topic_model/
run_manifest.json
topic_labels_llm.csv
topic_labels_llm_audit.jsonl
topic_labels_run_manifest.json
topic_groups_llm.csv
topic_group_summaries_llm.csv
topic_groups_llm_audit.json
topic_groups_run_manifest.json
inspection_tables/
visualizations/
topic_classification/README.md
topic_classification/scripts/
topic_classification/notebooks/
topic_classification/requirements-topic.txt
```

Final manuscript reproducibility checklist:

- The final BERTopic directory is
  `/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/topic_classification/bertopic_political_corruption_200_min10`.
- The final sample is
  `political_corruption_country_year_sample_200.csv.gz`.
- The final raw-topic model has 67 non-outlier lower-level BERTopic topics.
  Check with:

```bash
python3 - <<'PY'
from pathlib import Path
import pandas as pd

topic_dir = Path("/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/topic_classification/bertopic_political_corruption_200_min10")
topic_info = pd.read_csv(topic_dir / "topic_info.csv")
print(topic_info[topic_info["Topic"].ne(-1)].shape[0])
PY
```

- The final notebook table export should produce:

```text
inspection_tables/latex/table_topic_higher_order_summary.tex
inspection_tables/latex/table_all_topics_llm_higher_order_topics.tex
```

- The many-topic appendix table should be around 97 lines after export. If it
  is around 36 lines, it is the stale 8-topic export from the wrong directory.
- Archive both the compact LaTeX tables and the full CSV/JSON audit files. The
  LaTeX tables are for the journal appendix; the CSV/JSON files are the
  complete reproducibility record.
- Treat `topic_groups_llm.csv` and `topic_groups_llm_audit.json` as published
  outputs. Re-running GPT 5.1 years later may not produce byte-identical
  assignments, even with `temperature=0`.

For a full rerun from Research Drive rather than local `annecuda` paths, the
recommended starting point is:

```bash
python3 topic_classification/scripts/01_create_stratified_topic_sample.py \
  --source classified-webdav \
  --classified-rd-dir "ASCOR-FMG-5580-RESPOND-news-data (Projectfolder)/victims-of-corruption-paper/derived_data/political_classifier/classifier_outputs/classified_country_files" \
  --political-only \
  --per-country-year 200 \
  --output-name political_corruption_source_filtered_country_year_sample_200.csv.gz
```

Then fit BERTopic from that saved sample and continue with the LLM labelling,
grouping, notebook inspection, and table/archive upload steps below. This keeps
the input-data dependency explicit and avoids relying on a local mounted folder
still existing in the same place.

The manifests record package versions from the environment where the scripts
were run. For even stronger rerun reproducibility, also save a frozen
environment file from `annecuda`:

```bash
python3 -m pip freeze > topic_classification/requirements-topic-freeze-$(date +%Y%m%d).txt
```

If the exact embedding-model revision is needed, download or cache the Hugging
Face model files used by `intfloat/multilingual-e5-large` and archive the cache
or record the model commit hash. The current script records the embedding model
name, but not a pinned Hugging Face revision.

LLM labels and higher-order-topic groupings should be treated as archived outputs,
not guaranteed-regenerable outputs. Even with `temperature=0`, hosted LLMs can
change over time. The audit files preserve the exact prompt and response for
the published labels.

## Publish Topic Archive To Research Drive

Use the packaging script after the sample, BERTopic model, LLM labels/groups,
notebook tables, and visualizations have been generated.

For the current manuscript workflow, the usual order is:

1. Pull the latest repository code.
2. Rerun `topic_classification/notebooks/01_inspect_topic_results.ipynb` using
   the final many-topic BERTopic directory,
   `/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/topic_classification/bertopic_political_corruption_source_filtered_200_min10`.
3. Confirm that `inspection_tables/latex/` contains
   `table_topic_higher_order_summary.tex` and
   `table_all_topics_llm_higher_order_topics.tex`.
4. Confirm that the all-topics table is the many-topic export, not a stale
   8-topic export.
5. Upload the LaTeX tables to the paper output folder with
   `--upload-latex-tables --tables-only`.
6. Optionally upload the full reproducibility archive with `--upload`.

Create a local archive only:

```bash
python3 topic_classification/scripts/06_publish_topic_archive_to_webdav.py \
  --bertopic-dir /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/topic_classification/bertopic_political_corruption_source_filtered_200_min10 \
  --sample /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/topic_classification/political_corruption_source_filtered_country_year_sample_200.csv.gz
```

The browser folder is still visible at:

```text
https://uva.data.surf.nl/apps/files/?dir=/ASCOR-FMG-5580-RESPOND-news-data%20%28Projectfolder%29/victims-of-corruption-paper
```

For command-line upload, the script uses the same Research Drive/WebDAV settings
as the rest of the repository: `BASE_URL`, `USER`, and `APP_PASSWORD` from
`config_local.py`, or `RD_BASE_URL`, `RD_USER`, and `RD_PASS` from the
environment.

Upload the full topic archive:

```bash
python3 topic_classification/scripts/06_publish_topic_archive_to_webdav.py \
  --bertopic-dir /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/topic_classification/bertopic_political_corruption_source_filtered_200_min10 \
  --sample /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/topic_classification/political_corruption_source_filtered_country_year_sample_200.csv.gz \
  --upload
```

Default archive destination:

```text
ASCOR-FMG-5580-RESPOND-news-data (Projectfolder)/victims-of-corruption-paper/derived_data/topic_classification/
```

If the archive is too large because it includes `topic_model/`, rerun with
`--no-model` for a smaller upload, but keep a separate archived copy of
`topic_model/` for maximum reproducibility.

To push the manuscript-ready LaTeX tables to the paper output folder, first
make sure the notebook has exported tables from the many-topic model. The
all-topics table should be substantially longer than the stale 8-topic export
(the 8-topic table is only about 36 lines):

```bash
wc -l /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/topic_classification/bertopic_political_corruption_source_filtered_200_min10/inspection_tables/latex/table_all_topics_llm_higher_order_topics.tex
```

For the source-filtered many-topic model, this should be clearly longer than
the stale 8-topic export after the notebook export cell has been rerun. Then
upload the tables to a `topic models` subfolder under the standard
`output/tables` folder:

```bash
python3 topic_classification/scripts/06_publish_topic_archive_to_webdav.py \
  --bertopic-dir /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/topic_classification/bertopic_political_corruption_source_filtered_200_min10 \
  --upload-latex-tables \
  --tables-only \
  --tables-folder-name "topic models"
```

This uploads:

```text
ASCOR-FMG-5580-RESPOND-news-data (Projectfolder)/victims-of-corruption-paper/output/tables/topic models/table_topic_higher_order_summary.tex
ASCOR-FMG-5580-RESPOND-news-data (Projectfolder)/victims-of-corruption-paper/output/tables/topic models/table_all_topics_llm_higher_order_topics.tex
```

After downloading the tables from Research Drive/SURF, verify that
`table_all_topics_llm_higher_order_topics.tex` is the many-topic export:

```bash
wc -l "table_all_topics_llm_higher_order_topics.tex"
```

If it is still around 36 lines, the uploaded file is stale and was generated
from the old 8-topic directory rather than from
`bertopic_political_corruption_source_filtered_200_min10`.
