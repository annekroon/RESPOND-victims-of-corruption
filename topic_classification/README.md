# Topic Classification And Discovery

This folder contains the exploratory workflow for mapping human-interpretable
topics among articles already classified as primarily discussing political
corruption.

The core idea is to avoid fitting one topic model to the full news corpus at
once. The corpus is large and uneven across countries and time, so the first
step is a reproducible stratified sample across `country x year` among
political-corruption articles. The second step fits a multilingual BERTopic
model with a deliberately small target number of topics. The third step asks
GPT 5.1, through the same UvA LLM proxy used in Part 1, to name and interpret
the topics. The final step exports interactive visualizations for cross-country
and over-time topic patterns.

## Folder Structure

| Path | Purpose |
|---|---|
| `scripts/create_stratified_topic_sample.py` | Create balanced random samples across country and year |
| `scripts/fit_multilingual_bertopic.py` | Fit BERTopic on a sample and export document-topic assignments |
| `scripts/label_topics_with_llm.py` | Ask GPT 5.1 to create human-readable topic labels and coding rules |
| `scripts/group_topics_with_llm.py` | Ask GPT 5.1 to group fine-grained topics into higher-order inductive groups |
| `scripts/build_topic_visualizations.py` | Build interactive Plotly country/time topic graphs |
| `notebooks/01_inspect_topic_results.ipynb` | Read final outputs and inspect topic tables, examples, and country/time graphs |
| `requirements-topic.txt` | Optional extra dependencies for BERTopic |

## Recommended Design

Main sample/model:

- Input: country files from `political_classifier/scripts/train_final_classifier.py --score-corpus`.
- Filter: `pred_political_corruption == 1`.
- Sampling: balanced across `country x year`, with `analysis_weight` saved so
  visualizations can recover weighted article counts and shares.
- Topic model: multilingual BERTopic using `intfloat/multilingual-e5-large`.
- Interpretability: fit a somewhat granular raw topic model first, then use GPT
  5.1 to label the discovered clusters inductively from representative
  documents and topic keywords.
- Visualizations: interactive country-topic heatmap, stacked topic shares over
  time, and faceted country-over-time topic trends.

BERTopic is useful for discovering structure, but corruption type should
probably become a supervised or human-in-the-loop coding task after the first
topic map is inspected. Topic clusters are not the same as valid corruption-type
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
TRANSFORMERS_CACHE=/home/akroon/data/1t_storage/huggingface_cache \
python3 -m pip install -r topic_classification/requirements-topic.txt
```

## 1. Create A Political-Corruption Sample

This reads the classified country files created by
`political_classifier/scripts/train_final_classifier.py --score-corpus` and
keeps only rows with `pred_political_corruption == 1`.

```bash
python3 topic_classification/scripts/create_stratified_topic_sample.py \
  --source classified \
  --political-only \
  --per-country-year 100 \
  --output-name political_corruption_country_year_sample.csv.gz
```

If a country-year has fewer than the requested number of rows, all available
rows in that stratum are retained. The output includes `analysis_weight`,
`stratum_total_rows`, and `stratum_sample_rows`, which are used by the
visualization script.

## 2. Fit Multilingual BERTopic

Small test run:

```bash
CUDA_VISIBLE_DEVICES=1 \
python3 topic_classification/scripts/fit_multilingual_bertopic.py \
  --sample /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/topic_classification/political_corruption_country_year_sample.csv.gz \
  --output-dir /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/topic_classification/bertopic_political_corruption_test \
  --max-docs 2000 \
  --min-topic-size 25 \
  --nr-topics auto
```

Full run:

```bash
TMPDIR=/home/akroon/data/1t_storage/tmp \
HF_HOME=/home/akroon/data/1t_storage/huggingface_cache \
TRANSFORMERS_CACHE=/home/akroon/data/1t_storage/huggingface_cache \
CUDA_VISIBLE_DEVICES=1 \
nohup python3 -u topic_classification/scripts/fit_multilingual_bertopic.py \
  --sample /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/topic_classification/political_corruption_country_year_sample.csv.gz \
  --output-dir /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/topic_classification/bertopic_political_corruption \
  --min-topic-size 25 \
  --nr-topics auto \
  > bertopic_political_corruption.log 2>&1 &
```

Main outputs:

```text
topic_info.csv
document_topics.csv.gz
topic_model/
```

## 3. Label Topics With GPT 5.1

This uses the same `LLMPROXY_BASE_URL`, `LLMPROXY_API_KEY`, and
`LLMPROXY_MODEL` settings as the political-corruption classifier. By default,
`LLMPROXY_MODEL` is `gpt-5.1`.

```bash
python3 topic_classification/scripts/label_topics_with_llm.py \
  --bertopic-dir /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/topic_classification/bertopic_political_corruption \
  --model gpt-5.1
```

Output:

```text
topic_labels_llm.csv
```

Each row includes an inductive topic label, short chart label, topic summary,
inclusion rule, exclusion rule, country/event-specific flag, cross-country
comparability rating, label rationale, and confidence score. GPT 5.1 is used
only to interpret and name the discovered BERTopic clusters; it is not asked to
apply a predefined corruption-type taxonomy.

## 4. Group Fine-Grained Topics Inductively

For a many-topic solution, keep the fine-grained BERTopic clusters, then ask GPT
5.1 to group those discovered topics into higher-order groups. This is still
inductive: GPT sees the topic labels/summaries and proposes groups from them,
without a predefined taxonomy.

```bash
python3 topic_classification/scripts/group_topics_with_llm.py \
  --bertopic-dir /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/topic_classification/bertopic_political_corruption_200_min10 \
  --model gpt-5.1 \
  --target-groups 12
```

Outputs:

```text
topic_groups_llm.csv
topic_group_summaries_llm.csv
```

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

By default, the notebook plots `topic_group_short_label`, the GPT-labelled
higher-order inductive group. Set `ANALYSIS_LABEL_COLUMN = "topic_short_label"`
in the first code cell to inspect the fine-grained BERTopic topics directly.
It also includes lift diagnostics and country-specificity checks to evaluate
whether the discovered topics capture variation across countries and over time.

## 6. Optional: Export Standalone HTML Visualizations

```bash
python3 topic_classification/scripts/build_topic_visualizations.py \
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
