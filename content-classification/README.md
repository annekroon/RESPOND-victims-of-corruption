# Article-Level Content Classification

This folder measures four substantive variables within the final corpus of
articles classified as political corruption:

- `victim_visibility`
- `corruption_frame`
- `case_location`
- `accused_actor_visibility`

The political-corruption classifier defines the analysis population. The
content scripts do not redraw that boundary. They verify and then use the final
source-screened classifier output from:

```text
/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/
  political_corruption_pipeline/silver_classifier/
```

The completed upstream run uses threshold `0.60`, the source policy
`exclude_explicit_no_retain_all_other_decisions`, and contains `326,093`
political-corruption articles across nine countries. Scripts read these values
from manifests and fail if files, hashes, counts, or the threshold disagree.

## Current Status

| Stage | Status |
|---|---|
| Final political-corruption corpus | Complete |
| Codebook-development set, 108 articles (12 per country) | Complete; preserve as development data |
| Prompts | Current versions live in `scripts/content_prompts.py` |
| Separate held-out content validation | Not yet complete |
| Full-corpus content coding | Run only after held-out validation |

GPT-5.1 is the currently accessible, documented model. An access test for
`gpt-5.6-terra` returned HTTP 403, so it has not been evaluated and must not be
named as the production model. A future model change requires a successful
access check and a like-for-like validation before production coding.

## Maintained Files

| File | Purpose |
|---|---|
| `scripts/00_verify_final_corpus.py` | Validate upstream manifests, threshold, source policy, files, and counts |
| `scripts/create_validation_sample.py` | Draw reproducible country-year samples and exclude development articles |
| `scripts/translate_validation_sample.py` | Translate a sample for human coding and preserve provenance |
| `tools/annotation_streamlit_app.py` | Recommended content-coding interface with country queues, progress, and review |
| `tools/annotation_flask_app.py` | Compatibility interface for existing Flask-based coding sessions |
| `scripts/content_prompts.py` | Versioned codebook prompts and output schemas |
| `scripts/classify_all_content_categories.py` | Run all four coders on one validation sample |
| `scripts/evaluate_codebook_gpt_against_human.py` | Agreement, kappa, F1, confusion, and disagreement outputs |
| `scripts/05_run_final_content_classification.sh` | Resume all four production coders in sequence and merge them |
| `scripts/merge_content_labels.py` | Strict one-to-one production merge |

The notebook is optional inspection material, not a production entry point.

## 00 Verify The Final Corpus

Run this after any political-classifier rebuild and before sampling or content
coding:

```bash
cd ~/RESPOND-victims-of-corruption

python3 content-classification/scripts/00_verify_final_corpus.py
```

The command streams through all nine classified country files. Success should
report threshold `0.60` and political-corruption N `326,093`. Use
`--manifest-only` only for a quick status check; production work should use the
full verification.

## 01 Preserve Development Data

The 108 articles already read while developing the codebook are development
data, not held-out validation. Keep the original samples, English translations,
human annotations, GPT outputs, and disagreement analyses under:

```text
/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/
  political_corruption_pipeline/content_codebook_validation/
```

Do not overwrite or relabel these as final validation. The final sampler
requires at least 108 distinct exclusions and records their file hashes.

The substantive definitions are frozen in `scripts/content_prompts.py`. The
most important narrow rules are:

- Victim harm must be explicitly connected to the corruption, not inferred
  from the offense, investigation, scandal, or surrounding controversy.
- Concrete victims take priority when concrete and institutional harm are both
  explicit.
- An organization is an accused actor only when the article independently
  attributes corrupt participation to that organization.
- `case_location` compares the main corruption case with the supplied
  publication country.

## 02 Draw The Held-Out Validation Sample

The canonical design is 500 articles stratified across country-year cells.
Every article inspected during codebook development must be excluded. The
following uses the nine coder files as the exclusion list:

```bash
cd ~/RESPOND-victims-of-corruption

PIPE=/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline
DEV="$PIPE/content_codebook_validation"
VAL=/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/content_classification/validation_final
mkdir -p "$VAL"

DEV_FILES=("$DEV"/*_content_codebook_validation_n12_english_anne.csv)
printf 'Development files: %s\n' "${#DEV_FILES[@]}"

python3 content-classification/scripts/create_validation_sample.py \
  --sample-purpose final_validation \
  --total-sample 500 \
  --random-state 20260908 \
  --output-dir "$VAL" \
  --output-name content_validation_final_n500.csv.gz \
  --exclude-sample "${DEV_FILES[@]}"
```

Stop if the shell does not report nine development files or if the sampler
does not report at least 108 distinct exclusions. The outputs are:

```text
content_validation_final_n500.csv.gz
content_validation_final_n500_strata.csv
content_validation_final_n500.csv.gz.sample_manifest.json
```

The sample manifest records the random seed, exclusions, classifier threshold,
source policy, upstream hashes, and output hashes. Existing samples are never
overwritten unless `--overwrite` is explicitly supplied.

## 03 Translate And Annotate

Translate the held-out sample:

```bash
nohup python3 -u content-classification/scripts/translate_validation_sample.py \
  --input "$VAL/content_validation_final_n500.csv.gz" \
  --output "$VAL/content_validation_final_n500_english.csv.gz" \
  --model gpt-5.1 \
  --max-chars 20000 \
  --save-every 10 \
  > "$VAL/translation.log" 2>&1 &

tail -f "$VAL/translation.log"
```

The completed translation receives its own sample manifest linked to the
original draw. Partial or error-containing translations do not.

Start the Streamlit annotation app with coder-specific, resumable output:

```bash
CONTENT_ANNOTATION_INPUT="$VAL/content_validation_final_n500_english.csv.gz" \
CONTENT_ANNOTATION_OUTPUT_TEMPLATE="$VAL/content_validation_final_n500_english_{coder_id}.csv.gz" \
CONTENT_ANNOTATION_CODER_ID="anne" \
CONTENT_ANNOTATION_CODER_FIRST_NAME="Anne" \
CONTENT_ANNOTATION_PASSWORD="choose-a-strong-password" \
streamlit run content-classification/tools/annotation_streamlit_app.py \
  --server.address 127.0.0.1 \
  --server.port 8502
```

For an SSH tunnel:

```bash
ssh -L 8502:127.0.0.1:8502 akroon@annecuda
```

Open `http://localhost:8502`. The sidebar selects a country and an uncoded,
all, or completed queue. The main view presents the English translation,
original text, four required coding questions, and one prominent Save and
continue action. Codes are saved after every article; restarting with the same
coder ID resumes the existing file. The completion screen still allows coders
to return to earlier items and revise saved codes. A download button provides a
manual backup at any point.

The Flask app remains available for an existing session whose output was
created with it. Both interfaces use the same CSV columns and annotation
manifest contract; do not run both against the same coder file simultaneously.

## 04 Code And Evaluate The Held-Out Sample

Run all four frozen GPT-5.1 prompts on the translated sample:

```bash
GPT_VAL="$VAL/gpt51_labels"

nohup python3 -u content-classification/scripts/classify_all_content_categories.py \
  --input "$VAL/content_validation_final_n500_english.csv.gz" \
  --output-dir "$GPT_VAL" \
  --model gpt-5.1 \
  --max-chars 6000 \
  --save-every 10 \
  > "$VAL/gpt51_validation.log" 2>&1 &

tail -f "$VAL/gpt51_validation.log"
```

After human coding is complete, evaluate it. Replace `anne` only if the coder
ID differs:

```bash
python3 content-classification/scripts/evaluate_codebook_gpt_against_human.py \
  --human-file "$VAL/content_validation_final_n500_english_anne.csv.gz" \
  --gpt-dir "$GPT_VAL" \
  --coder-id anne \
  --require-final-validation \
  --output-prefix final_n500_gpt51
```

Outputs include overall and country-specific agreement, Cohen's kappa, macro
F1, design-weighted agreement, confusion counts, and a disagreement packet.
The design-weighted statistics use the saved country-year sampling weights.

This sample is a final test only while its cases do not change the prompts. If
its disagreements are used to tune definitions or prompts, treat it as new
development data, exclude it, and draw a fresh final validation sample.

The evaluator retains backward compatibility with the old country-by-country
development files through `--countries` and `--codebook-dir`.

## 05 Run Full Production Coding

Start this only after the model and all prompt versions have passed held-out
validation and have been frozen. The maintained runner verifies the complete
upstream corpus, resumes failed or interrupted rows, runs the four variables
sequentially, and performs the strict merge:

```bash
cd ~/RESPOND-victims-of-corruption

DATA=/home/akroon/data/1t_storage/RESPOND-victims-of-corruption
CONTENT_OUTPUT_DIR="$DATA/content_classification/final_gpt51"
LOG="$DATA/content_classification/final_gpt51_run.log"
mkdir -p "$CONTENT_OUTPUT_DIR"

nohup env \
  CONTENT_MODEL=gpt-5.1 \
  CONTENT_OUTPUT_DIR="$CONTENT_OUTPUT_DIR" \
  bash content-classification/scripts/05_run_final_content_classification.sh \
  > "$LOG" 2>&1 &

echo $! > "$CONTENT_OUTPUT_DIR/run.pid"
tail -f "$LOG"
```

Each variable writes a CSV.GZ checkpoint, append-only JSONL audit, immutable run
manifest, and completion marker. A completion marker is written only when all
expected rows are present, labels are nonblank, no errors remain, and the input
matches the final classifier run.

The final merged measurement file is:

```text
/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/
  content_classification/final_gpt51/
  political_corruption_content_categories_final.csv.gz
```

Its adjacent manifest records the exact four inputs, content model, article-ID
fingerprint, upstream classifier hash, threshold, and source-filter policy.
The canonical merge rejects validation samples, country subsets, partial files,
failed rows, duplicate IDs, invalid labels, different models, and outputs from
different upstream corpus builds.

CPI and other covariates are intentionally not added to this canonical
measurement artifact. Add them in a separate analysis-data construction step.
The merge retains diagnostic `--allow-partial`, `--allow-errors`, and `--cpi`
options, but those outputs are not canonical production data.

## Reproducibility Contract

- Never edit `scripts/content_prompts.py` while a run is in progress.
- Start changed prompts, changed models, or changed maximum text lengths in a
  new output directory.
- Use `--retry-errors` to resume failed rows; use `--overwrite` only when
  intentionally discarding an entire prior run.
- Keep the JSONL audits. They retain prompts, raw responses, parsed responses,
  timestamps, model names, and errors.
- Keep all `.run.json`, `.sample_manifest.json`, `.complete.json`, and merged
  manifest files with the generated data.
- Do not present the 108 development articles as held-out validation.
- Do not launch a newly available model on the full corpus before a successful
  access check and frozen-sample comparison.
