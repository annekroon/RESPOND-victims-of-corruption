# Article-Level Content Classification

This folder contains the zero-shot GPT 5.1 workflow for coding article-level
variables among articles already classified as primarily discussing political
corruption. The expected corpus is the final political-corruption article set
from `political_classifier/scripts/train_final_classifier.py`, currently about
`474,328` articles.

The workflow is intentionally separate from the political-corruption classifier:
that first classifier identifies the analysis corpus; these scripts measure
substantive variables inside that corpus.

## Variables

| Script | Main output |
|---|---|
| `scripts/classify_victim_visibility.py` | `victim_visibility`, `victim_visible` |
| `scripts/classify_corruption_frame.py` | `corruption_frame` |
| `scripts/classify_abroad_case.py` | `case_location`, `abroad_case` |
| `scripts/classify_accused_actor.py` | `accused_actor_visibility`, `accused_actor_visible` |
| `scripts/create_validation_sample.py` | country-year stratified validation sample for human/GPT comparison |
| `scripts/translate_validation_sample.py` | GPT translation of validation-sample articles into English for human coding |
| `tools/annotation_flask_app.py` | Browser-based Flask app for manual coding with original and translated text |
| `scripts/merge_content_labels.py` | one merged silver-labelled article-level dataset |

Each classifier sends article text to the UvA LLM proxy with deterministic
settings where supported (`temperature=0`) and requires structured JSON output
containing a category, evidence, a short explanation, and a confidence score.
Prompts are stored in `scripts/content_prompts.py` with explicit prompt-version
strings.

## Inputs

By default the scripts read local classified country files:

```text
/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/
  political_corruption_pipeline/silver_classifier/classified_country_files/
```

Only rows with `pred_political_corruption == 1` are classified unless
`--keep-non-political` is passed. The scripts can also read archived classified
files directly from Research Drive/WebDAV:

```bash
--source classified-webdav
```

or a single CSV/CSV.GZ file:

```bash
--source csv --input /path/to/articles.csv.gz
```

## Pilot Runs

Always run small pilots before launching the full corpus. From the repository
root on `annecuda`:

```bash
python3 content-classification/scripts/classify_victim_visibility.py \
  --limit 25 \
  --output-dir /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/content_classification/pilot
```

Repeat for the other concepts:

```bash
python3 content-classification/scripts/classify_corruption_frame.py \
  --limit 25 \
  --output-dir /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/content_classification/pilot

python3 content-classification/scripts/classify_abroad_case.py \
  --limit 25 \
  --output-dir /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/content_classification/pilot

python3 content-classification/scripts/classify_accused_actor.py \
  --limit 25 \
  --output-dir /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/content_classification/pilot
```

Inspect outputs with:

```text
content-classification/notebooks/01_inspect_content_classification.ipynb
```

Set `CONTENT_DIR` in the first notebook cell to the pilot or full output folder.

## Codebook Development Sample

Before final validation, use a small country-year stratified random sample to
read cases, refine the codebook, check category boundaries, and test the manual
annotation interface. This is a development sample, not a held-out validation
set. Any article read while changing the codebook or prompts should be excluded
from the final validation logic.

Create an `N=100` stratified random sample from the political-corruption corpus:

```bash
CODEBOOK_DIR=/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/content_classification/codebook_development

python3 content-classification/scripts/create_validation_sample.py \
  --total-sample 100 \
  --sample-purpose codebook_development \
  --output-name content_codebook_dev_sample_100.csv.gz \
  --output-dir "$CODEBOOK_DIR"
```

This writes:

```text
content_codebook_dev_sample_100.csv.gz
content_codebook_dev_sample_100_strata.csv
```

The file includes `sample_purpose = codebook_development`, country-year stratum
diagnostics, `validation_weight`, article text, and blank `human_*` columns.
Use it for codebook development, category clarification, coder training, and
interface testing only.

Translate the same `N=100` development sample to English:

```bash
nohup python3 -u content-classification/scripts/translate_validation_sample.py \
  --input "$CODEBOOK_DIR/content_codebook_dev_sample_100.csv.gz" \
  --output "$CODEBOOK_DIR/content_codebook_dev_sample_100_english.csv.gz" \
  --model gpt-5.1 \
  --max-chars 20000 \
  > content_codebook_dev_translation.log 2>&1 &
```

Monitor translation progress with:

```bash
tail -f content_codebook_dev_translation.log
```

To test the annotation interface on this development sample:

```bash
CONTENT_ANNOTATION_INPUT="$CODEBOOK_DIR/content_codebook_dev_sample_100_english.csv.gz" \
CONTENT_ANNOTATION_OUTPUT="$CODEBOOK_DIR/content_codebook_dev_sample_100_human_notes.csv.gz" \
CONTENT_ANNOTATION_PASSWORD="choose-a-password" \
CONTENT_ANNOTATION_CODER_ID="anne_codebook_dev" \
flask --app content-classification/tools/annotation_flask_app.py run \
  --host 127.0.0.1 \
  --port 8502
```

## Validation Sample

For validation, create a smaller country-year stratified random sample from the
political-corruption corpus after the codebook and GPT prompts are frozen. Do
not use the codebook-development sample as final validation evidence. A
500-article sample gives roughly 6-7 articles per non-empty country-year stratum
if the period has 72 strata (9 countries by 8 years):

```bash
python3 content-classification/scripts/create_validation_sample.py \
  --total-sample 500 \
  --output-dir /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/content_classification/validation
```

This writes:

```text
content_validation_sample_500.csv.gz
content_validation_sample_500_strata.csv
```

The sample includes the article text, stratum totals, sample counts,
`validation_weight`, and blank human-coding columns for all four concepts.
For a balanced country validation design, use 100 articles per country:

```bash
python3 content-classification/scripts/create_validation_sample.py \
  --per-country 100 \
  --output-dir /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/content_classification/validation
```

This writes:

```text
content_validation_sample_100_per_country.csv.gz
content_validation_sample_100_per_country_strata.csv
```

Translate the 900 sampled articles to English for human coding:

```bash
VALIDATION_DIR=/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/content_classification/validation

nohup python3 -u content-classification/scripts/translate_validation_sample.py \
  --input "$VALIDATION_DIR/content_validation_sample_100_per_country.csv.gz" \
  --output "$VALIDATION_DIR/content_validation_sample_100_per_country_english.csv.gz" \
  > content_validation_translation.log 2>&1 &
```

The translation output preserves all original columns and adds
`translated_text_en`, `translation_notes`, `translation_confidence`,
`translation_model`, and `translation_error`. The script is resumable.

### Manual Annotation Interface

After translation, launch the Flask annotation app. It shows the English
translation by default, lets coders switch to the original article or a
side-by-side view, and keeps the codebook definitions visible while coding.
Coders log in with a coder ID; if `CONTENT_ANNOTATION_OUTPUT_TEMPLATE` is set,
each coder automatically writes to a separate output file.

For a local-only session on `annecuda`:

```bash
VALIDATION_DIR=/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/content_classification/validation

CONTENT_ANNOTATION_INPUT="$VALIDATION_DIR/content_validation_sample_100_per_country_english.csv.gz" \
CONTENT_ANNOTATION_OUTPUT="$VALIDATION_DIR/content_validation_sample_100_per_country_human_coded.csv.gz" \
CONTENT_ANNOTATION_PASSWORD="choose-a-password" \
CONTENT_ANNOTATION_CODER_ID="anne" \
flask --app content-classification/tools/annotation_flask_app.py run \
  --host 127.0.0.1 \
  --port 8502
```

If working through an SSH tunnel:

```bash
ssh -L 8502:127.0.0.1:8502 akroon@annecuda
```

Then open:

```text
http://localhost:8502
```

For external coders, run the app on a reachable host or behind a reverse proxy
with `--host 0.0.0.0`, a strong `CONTENT_ANNOTATION_PASSWORD`, and a non-default
`CONTENT_ANNOTATION_SECRET_KEY`. Use `CONTENT_ANNOTATION_OUTPUT_TEMPLATE` so all
coders can use the same app URL while their annotations are saved separately:

```bash
CONTENT_ANNOTATION_INPUT="$VALIDATION_DIR/content_validation_sample_100_per_country_english.csv.gz" \
CONTENT_ANNOTATION_OUTPUT_TEMPLATE="$VALIDATION_DIR/content_validation_sample_100_per_country_{coder_id}.csv.gz" \
CONTENT_ANNOTATION_PASSWORD="strong-password-here" \
CONTENT_ANNOTATION_SECRET_KEY="another-long-random-secret" \
flask --app content-classification/tools/annotation_flask_app.py run \
  --host 0.0.0.0 \
  --port 8502
```

Coders then open the server URL in their own browser, enter their coder ID and
the shared password, and annotate independently. The app saves the human codes
in `human_*` columns and is safe to restart: if a coder-specific output file
already exists, that coder resumes from the reviewed file.

After manual coding, the same sample can be sent through the GPT classifiers to
compare GPT labels against human labels:

```bash
VALIDATION_DIR=/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/content_classification/validation
SAMPLE="$VALIDATION_DIR/content_validation_sample_500.csv.gz"

python3 content-classification/scripts/classify_victim_visibility.py \
  --source csv \
  --input "$SAMPLE" \
  --output-dir "$VALIDATION_DIR/gpt_labels"

python3 content-classification/scripts/classify_corruption_frame.py \
  --source csv \
  --input "$SAMPLE" \
  --output-dir "$VALIDATION_DIR/gpt_labels"

python3 content-classification/scripts/classify_abroad_case.py \
  --source csv \
  --input "$SAMPLE" \
  --output-dir "$VALIDATION_DIR/gpt_labels"

python3 content-classification/scripts/classify_accused_actor.py \
  --source csv \
  --input "$SAMPLE" \
  --output-dir "$VALIDATION_DIR/gpt_labels"
```

This design keeps the expensive full-corpus GPT labelling separate from the
validation exercise. If the 500-case validation shows weak agreement on a
concept, revise the prompt version before running that concept on all 474,328
political-corruption articles.

## Full Runs

The full corpus is large, so run each concept separately with `nohup`. The
scripts are resumable: if an output file already exists, completed `article_id`
rows are skipped. Use `--retry-errors` to reprocess rows with non-empty
`llm_error`.

```bash
CONTENT_DIR=/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/content_classification

nohup python3 -u content-classification/scripts/classify_victim_visibility.py \
  --output-dir "$CONTENT_DIR" \
  --max-chars 6000 \
  > content_victim_visibility.log 2>&1 &

nohup python3 -u content-classification/scripts/classify_corruption_frame.py \
  --output-dir "$CONTENT_DIR" \
  --max-chars 6000 \
  > content_corruption_frame.log 2>&1 &

nohup python3 -u content-classification/scripts/classify_abroad_case.py \
  --output-dir "$CONTENT_DIR" \
  --max-chars 6000 \
  > content_abroad_case.log 2>&1 &

nohup python3 -u content-classification/scripts/classify_accused_actor.py \
  --output-dir "$CONTENT_DIR" \
  --max-chars 6000 \
  > content_accused_actor.log 2>&1 &
```

Monitor progress with:

```bash
tail -f content_victim_visibility.log
```

## Merge Silver Labels

After all four concept files are present:

```bash
python3 content-classification/scripts/merge_content_labels.py \
  --input-dir /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/content_classification \
  --cpi output/cpi_country_year_scores.csv
```

This writes:

```text
/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/
  content_classification/content_silver_labels_merged.csv.gz
```

Derived variables include:

| Variable | Definition |
|---|---|
| `victim_visible_binary` | 1 for concrete or institutional/societal victim; 0 for no victim |
| `concrete_victim_visible` | 1 for concrete victim; 0 for no victim or institutional/societal victim |
| `institutional_societal_victim_visible` | 1 for institutional/societal victim; 0 otherwise |
| `abroad_case_binary` | 1 for abroad; 0 for domestic |
| `accused_actor_visible_binary` | 1 when any accused actor is visible; 0 when none is visible |
| `frame_individualized` | 1 for individualized frame; 0 for systemic or other/mixed |
| `frame_systemic` | 1 for systemic frame; 0 for individualized or other/mixed |
| `perceived_corruption_lag1` | `100 - CPI` from the previous country-year |

Rows coded `unclear` retain missing values in the derived binary variables so
they can be excluded from the relevant regression models.

## Reproducibility Notes

- The prompts live in `scripts/content_prompts.py`; do not edit them mid-run
  unless intentionally starting a new prompt version.
- Each output row stores `classifier_name` and `prompt_version`.
- Each classifier writes an audit JSONL next to the output file containing the
  prompt and raw model response for every successful article.
- The default model is read from `LLMPROXY_MODEL` in `config.py`, currently
  `gpt-5.1`.
- Credentials must be supplied through environment variables or ignored
  `config_local.py`: `LLMPROXY_API_KEY`, and if using WebDAV, Research Drive
  credentials.
