"""Create a second targeted active-learning batch from cleaned country files.

The script trains a classifier on LLM-generated silver labels, scores candidate
articles from the cleaned corpus, and samples a new batch focused on weaker
countries and articles near the selected decision threshold.

Example:
    python3 02_create_targeted_active_learning_batch.py

Then label the resulting batch with:
    nohup python3 -u 03_run_llmproxy_batch_labels.py \
      --input /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/active_learning/active_learning_batch_2_for_annotation.csv \
      --output /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/active_learning/active_learning_batch_2_with_llm_suggestions.csv \
      > llm_active_learning_batch_2.log 2>&1 &
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


DEFAULT_PIPELINE_DIR = Path(
    "/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/"
    "political_corruption_pipeline"
)
DEFAULT_ACTIVE_LEARNING_DIR = DEFAULT_PIPELINE_DIR / "active_learning"
DEFAULT_SILVER_LABEL_PATH = DEFAULT_ACTIVE_LEARNING_DIR / "active_learning_batch_with_llm_suggestions.csv"
DEFAULT_OUTPUT_PATH = DEFAULT_ACTIVE_LEARNING_DIR / "active_learning_batch_2_for_annotation.csv"

DEFAULT_COUNTRY_TARGETS = {
    "Sweden": 180,
    "United_Kingdom": 180,
    "Ukraine": 180,
    "Netherlands": 140,
    "Serbia": 120,
    "Hungary": 60,
    "Bulgaria": 60,
    "Italy": 40,
    "France": 40,
}

OUTPUT_COLUMNS = [
    "al_bucket",
    "uri",
    "country",
    "date_parsed",
    "year",
    "source_uri",
    "prob_political_corruption",
    "model_pred",
    "human_final_label",
    "human_notes",
    "article_text",
]


def fix_mojibake(text):
    if not isinstance(text, str):
        return ""

    candidates = [text]

    try:
        candidates.append(text.encode("latin1").decode("utf-8"))
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass

    try:
        candidates.append(text.encode("cp1252").decode("utf-8"))
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass

    def badness(s):
        markers = ["Ð", "Ñ", "Ã", "Â", "Ä", "Å", "�"]
        return sum(s.count(m) for m in markers)

    return min(candidates, key=badness)


def normalize_text(text):
    text = fix_mojibake(text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_country_targets(text: str | None) -> dict[str, int]:
    if not text:
        return DEFAULT_COUNTRY_TARGETS.copy()

    targets = {}
    for item in text.split(","):
        country, count = item.split(":", 1)
        targets[country.strip()] = int(count.strip())
    return targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a targeted active-learning batch.")
    parser.add_argument("--pipeline-dir", type=Path, default=DEFAULT_PIPELINE_DIR)
    parser.add_argument("--silver-labels", type=Path, default=DEFAULT_SILVER_LABEL_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--embedding-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--pool-per-country", type=int, default=50000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--country-targets",
        default=None,
        help="Comma-separated targets, e.g. Sweden:180,United_Kingdom:180.",
    )
    parser.add_argument(
        "--exclude",
        type=Path,
        action="append",
        default=None,
        help="CSV whose uri values should be excluded. Can be passed multiple times.",
    )
    return parser.parse_args()


def load_seen_uris(paths, silver_label_path):
    import pandas as pd

    seen = set()
    paths_to_check = [silver_label_path]
    if paths:
        paths_to_check.extend(paths)

    for path in paths_to_check:
        if not path.exists():
            continue
        data = pd.read_csv(path, usecols=lambda col: col == "uri")
        if "uri" in data.columns:
            seen.update(data["uri"].dropna().astype(str))

    return seen


def prepare_silver_training_data(path: Path):
    import pandas as pd

    label_map = {
        "Yes": 1,
        "No": 0,
        "Mentioned but not central": 0,
    }

    data = pd.read_csv(path)
    data = data[data["llm_label_suggestion"].isin(label_map)].copy()
    data["y"] = data["llm_label_suggestion"].map(label_map).astype(int)

    if "translated_text" in data.columns:
        text_source = data["translated_text"]
    else:
        text_source = data["article_text"]

    data["model_text"] = text_source.fillna("").astype(str).map(normalize_text)
    data = data[data["model_text"].str.strip().ne("")].copy()
    return data


def cleaned_country_path(pipeline_dir: Path, country: str) -> Path:
    return pipeline_dir / f"{country}_cleaned_deduped.csv.gz"


def load_country_pool(path: Path, country: str, seen_uris: set[str], pool_size: int, random_state: int):
    import pandas as pd

    columns = [
        "uri",
        "country",
        "date_parsed",
        "year",
        "source_uri",
        "article_text",
        "word_count",
    ]
    data = pd.read_csv(path, usecols=lambda col: col in columns)
    data = data[data["article_text"].fillna("").astype(str).str.strip().ne("")].copy()

    if "uri" in data.columns and seen_uris:
        data = data[~data["uri"].astype(str).isin(seen_uris)].copy()

    if len(data) > pool_size:
        data = data.sample(n=pool_size, random_state=random_state).copy()

    data["country"] = country
    return data.reset_index(drop=True)


def score_pool(data, embedder, clf, threshold: float):
    import numpy as np
    from tqdm.auto import tqdm

    texts = data["article_text"].fillna("").astype(str).map(normalize_text).tolist()

    embeddings = embedder.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    probs = clf.predict_proba(embeddings)[:, 1]

    scored = data.copy()
    scored["prob_political_corruption"] = probs
    scored["model_pred"] = (scored["prob_political_corruption"] >= threshold).astype(int)

    # Keep tqdm imported here so failures happen before long loops in environments
    # where dependencies are missing.
    _ = tqdm
    _ = np
    return scored


def take_unique(pool, selected_uris: set[str], n: int):
    if n <= 0 or pool.empty:
        return pool.head(0).copy()
    if "uri" in pool.columns:
        pool = pool[~pool["uri"].astype(str).isin(selected_uris)].copy()
    sample = pool.head(n).copy()
    if "uri" in sample.columns:
        selected_uris.update(sample["uri"].dropna().astype(str))
    return sample


def sample_country_batch(scored, target_n: int, threshold: float):
    import pandas as pd

    selected_uris: set[str] = set()

    n_boundary = int(target_n * 0.50)
    n_positive = int(target_n * 0.30)
    n_negative = int(target_n * 0.10)
    n_random = target_n - n_boundary - n_positive - n_negative

    scored = scored.copy()
    scored["distance_to_threshold"] = (scored["prob_political_corruption"] - threshold).abs()

    boundary = scored.sort_values("distance_to_threshold").copy()
    boundary["al_bucket"] = "threshold_boundary"

    likely_positive = scored[scored["prob_political_corruption"] >= threshold].sort_values(
        "prob_political_corruption",
        ascending=False,
    )
    likely_positive = likely_positive.copy()
    likely_positive["al_bucket"] = "likely_positive"

    likely_negative = scored[scored["prob_political_corruption"] < threshold].sort_values(
        "prob_political_corruption",
        ascending=True,
    )
    likely_negative = likely_negative.copy()
    likely_negative["al_bucket"] = "likely_negative"

    random_pool = scored.sample(frac=1, random_state=42).copy()
    random_pool["al_bucket"] = "random_check"

    parts = [
        take_unique(boundary, selected_uris, n_boundary),
        take_unique(likely_positive, selected_uris, n_positive),
        take_unique(likely_negative, selected_uris, n_negative),
        take_unique(random_pool, selected_uris, n_random),
    ]

    batch = pd.concat(parts, ignore_index=True)
    if len(batch) < target_n:
        topup = scored.sort_values("distance_to_threshold").copy()
        topup["al_bucket"] = "topup_boundary"
        batch = pd.concat(
            [batch, take_unique(topup, selected_uris, target_n - len(batch))],
            ignore_index=True,
        )

    return batch


def main() -> None:
    args = parse_args()

    import pandas as pd
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    from tqdm.auto import tqdm

    country_targets = parse_country_targets(args.country_targets)

    train_df = prepare_silver_training_data(args.silver_labels)
    print(f"Silver training rows: {len(train_df):,}", flush=True)
    print(train_df["y"].value_counts().rename(index={0: "No", 1: "Political corruption"}), flush=True)

    embedder = SentenceTransformer(args.embedding_model)
    x_train = embedder.encode(
        train_df["model_text"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.random_state)
    clf.fit(x_train, train_df["y"])

    seen_uris = load_seen_uris(args.exclude, args.silver_labels)
    print(f"Excluding already labelled/scored URIs: {len(seen_uris):,}", flush=True)

    country_batches = []
    for country, target_n in tqdm(country_targets.items(), desc="Scoring countries"):
        path = cleaned_country_path(args.pipeline_dir, country)
        if not path.exists():
            print(f"Missing cleaned file for {country}: {path}", flush=True)
            continue

        print(f"\n{country}: loading candidate pool from {path}", flush=True)
        pool = load_country_pool(
            path=path,
            country=country,
            seen_uris=seen_uris,
            pool_size=args.pool_per_country,
            random_state=args.random_state,
        )
        print(f"{country}: scoring {len(pool):,} candidates", flush=True)

        scored = score_pool(pool, embedder, clf, args.threshold)
        country_batch = sample_country_batch(scored, target_n=target_n, threshold=args.threshold)
        print(
            f"{country}: selected {len(country_batch):,} rows "
            f"(positive rate at threshold {args.threshold}: {scored['model_pred'].mean():.2%})",
            flush=True,
        )
        country_batches.append(country_batch)

    if not country_batches:
        raise RuntimeError("No country batches were created.")

    batch = pd.concat(country_batches, ignore_index=True)
    batch["human_final_label"] = ""
    batch["human_notes"] = ""

    output_cols = [column for column in OUTPUT_COLUMNS if column in batch.columns]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    batch[output_cols].to_csv(args.output, index=False)

    print(f"\nSaved targeted active-learning batch: {args.output}", flush=True)
    print(f"Rows: {len(batch):,}", flush=True)
    print("\nBy country:", flush=True)
    print(batch["country"].value_counts(), flush=True)
    print("\nBy bucket:", flush=True)
    print(batch["al_bucket"].value_counts(), flush=True)


if __name__ == "__main__":
    main()
