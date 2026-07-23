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
| `scripts/classify_all_content_categories.py` | convenience wrapper that runs all four GPT coders on one sample file |
| `scripts/create_validation_sample.py` | country-year stratified validation sample for human/GPT comparison |
| `scripts/translate_validation_sample.py` | GPT translation of validation-sample articles into English for human coding |
| `tools/annotation_flask_app.py` | Browser-based Flask app for manual coding with original and translated text |
| `scripts/merge_content_labels.py` | one merged silver-labelled article-level dataset |

Each classifier sends article text to the UvA LLM proxy with deterministic
settings where supported (`temperature=0`) and requires structured JSON output
containing a category, evidence, a short explanation, and a confidence score.
Prompts are stored in `scripts/content_prompts.py` with explicit prompt-version
strings.

The current codebook prompt versions are:

| Variable | Prompt version |
|---|---|
| `victim_visibility` | `victim_visibility_zero_shot_v6` |
| `corruption_frame` | `corruption_frame_zero_shot_v3` |
| `case_location` / `abroad_case` | `abroad_case_zero_shot_v3` |
| `accused_actor_visibility` | `accused_actor_zero_shot_v6` |

These versions implement the stricter rule that the model must first isolate
the corruption allegation/case, use only information stated in the article, and
avoid coding victims or actors that are linked only to unrelated harms or
unrelated misconduct. Earlier GPT outputs generated with older prompt versions
should be treated as pilot outputs and regenerated before comparison with human
coding.

For `victim_visibility`, version 6 uses a victim-specific prompt rather than
sending the other three variables' instructions to that classifier. GPT first
extracts the harm cause and status, whether a deprived entity is explicit,
whether coercive pressure was communicated, the victim entity, and the
corruption-to-harm evidence. It then codes concrete and
institutional/societal victim presence independently. The final label is
derived mechanically as `no_victim`,
`concrete_victim`, `institutional_societal_victim`,
`both_concrete_and_institutional`, or `unclear`. This preserves cases in which
both victim types are visible and prevents a positive label when the article
describes only possible, intended, future, unrelated, or inferred harm.

Version 6 also distinguishes harm caused by corruption from harm caused by the
investigation, prosecution, scandal, resignation, or institutional response. A
communicated coercive or extortionate demand counts as realized adverse
treatment even when its threatened consequence does not occur. The model may
not infer a public, party, or private funding source or deprived entity when the
article does not identify one.

The agreement evaluator now writes an additional `*_human_gpt_confusion.csv`
table and expands `*_human_gpt_disagreements.csv` into an adjudication file. It
retains translated/original article text, GPT evidence, reasoning, and
confidence where available, classifies victim errors as a visibility-gate or
victim-type disagreement, and adds blank adjudication fields for researcher
review. Use `--variables victim_visibility` to evaluate a newly generated
victim-only pilot without rerunning or copying the other three classifier
outputs.

For `accused_actor_visibility`, the finalized organization rule is deliberately
narrow and uses a mandatory two-test method. Count only actors whom the article
explicitly represents as committing, attempting, assisting, enabling, financing,
directing, or concealing the corruption. An organization counts only when the
text attributes corrupt participation to the organization itself, for example by
stating that it paid bribes, manipulated a tender, financed a scheme,
facilitated money laundering, concealed misconduct, or was investigated for
corruption. The required organizational allegation may appear in the same
sentence as the individual allegation; it must simply be independently
expressed. An organization does not count merely because its employee, leader,
owner, subsidiary, member, or associate is accused, or because it benefited from
or was connected to corruption. Human annotations made before this rule was
finalized, especially `both_individual_and_organizational` labels, should be
reviewed before being treated as gold-standard validation labels.

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

For a translated codebook-development sample, run all four content coders with:

```bash
CODEBOOK_DIR=/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/content_codebook_validation

python3 content-classification/scripts/classify_all_content_categories.py \
  --input "$CODEBOOK_DIR/Bulgaria_content_codebook_validation_n12_english.csv" \
  --output-dir "$CODEBOOK_DIR/gpt51_test_labels" \
  --model gpt-5.1 \
  --save-every 2
```

This writes one GPT-labelled file per content variable and keeps JSONL audit
logs with prompts and raw model responses in the same output folder.

To rerun all translated country-level codebook samples with the frozen prompts
using `gpt-5.1`, write the outputs to a new folder so old pilot labels remain
auditable:

```bash
nohup bash -c '
set -e
CODEBOOK_DIR=/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/content_codebook_validation
LLM_OUT_DIR="$CODEBOOK_DIR/gpt51_test_labels_v4_codebook"
mkdir -p "$LLM_OUT_DIR"
shopt -s nullglob
FILES=("$CODEBOOK_DIR"/*_content_codebook_validation_n12_english.csv)
if [ ${#FILES[@]} -eq 0 ]; then
  echo "No translated country sample files found in $CODEBOOK_DIR"
  exit 1
fi
for FILE in "${FILES[@]}"; do
  COUNTRY=$(basename "$FILE" _content_codebook_validation_n12_english.csv)
  echo "Running GPT-5.1 content coding for $COUNTRY"
  python3 content-classification/scripts/classify_all_content_categories.py \
    --input "$FILE" \
    --output-dir "$LLM_OUT_DIR/$COUNTRY" \
    --model gpt-5.1 \
    --save-every 2
done
echo "Done."
' > content_gpt51_codebook_validation_v4_codebook.log 2>&1 &
```

Monitor with:

```bash
tail -f content_gpt51_codebook_validation_v4_codebook.log
```

Compare the regenerated LLM labels against the human-coded countries currently
available:

```bash
CODEBOOK_DIR=/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/content_codebook_validation

python3 content-classification/scripts/evaluate_codebook_gpt_against_human.py \
  --codebook-dir "$CODEBOOK_DIR" \
  --gpt-dir "$CODEBOOK_DIR/gpt51_test_labels_v4_codebook" \
  --countries Bulgaria France Hungary Serbia \
  --coder-id anne \
  --output-prefix first4_gpt51_v4_codebook
```

After more countries are human-coded, add them to `--countries` and rerun the
same evaluation command.

You can also run individual coders, for example:

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
CONTENT_ANNOTATION_OUTPUT_TEMPLATE="$CODEBOOK_DIR/content_codebook_dev_sample_100_{coder_id}.csv.gz" \
CONTENT_ANNOTATION_PASSWORD="choose-a-password" \
CONTENT_ANNOTATION_HOST=127.0.0.1 \
CONTENT_ANNOTATION_PORT=8502 \
python3 content-classification/tools/annotation_flask_app.py
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
Coders log in with their first name; the app stores both that first name and a
unique `human_code_session_id` on every saved row. If
`CONTENT_ANNOTATION_OUTPUT_TEMPLATE` is set, each coder automatically writes to
a separate output file.

For a local-only session on `annecuda`:

```bash
VALIDATION_DIR=/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/content_classification/validation

CONTENT_ANNOTATION_INPUT="$VALIDATION_DIR/content_validation_sample_100_per_country_english.csv.gz" \
CONTENT_ANNOTATION_OUTPUT="$VALIDATION_DIR/content_validation_sample_100_per_country_human_coded.csv.gz" \
CONTENT_ANNOTATION_PASSWORD="choose-a-password" \
CONTENT_ANNOTATION_HOST=127.0.0.1 \
CONTENT_ANNOTATION_PORT=8502 \
python3 content-classification/tools/annotation_flask_app.py
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
CONTENT_ANNOTATION_HOST=0.0.0.0 \
CONTENT_ANNOTATION_PORT=8502 \
python3 content-classification/tools/annotation_flask_app.py
```

Coders then open the server URL in their own browser, enter their first name and
the shared password, and annotate independently. Use distinct first names or add
an initial when two coders share a name, because the first name is also used to
construct the coder-specific output filename. The app saves the human codes in
`human_*` columns and is safe to restart: if a coder-specific output file
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
