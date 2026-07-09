"""Fit a multilingual BERTopic model on a stratified topic sample."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-large"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit BERTopic on a multilingual sample.")
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--text-column", default="article_text")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--min-topic-size",
        type=int,
        default=25,
        help="Larger values produce fewer, broader raw clusters before domain aggregation.",
    )
    parser.add_argument(
        "--nr-topics",
        default="auto",
        help="Target number of raw topics after reduction. Use 'auto' to preserve discovered granularity.",
    )
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def normalize_text(text: object) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def format_texts_for_embedding(texts: list[str], embedding_model: str) -> list[str]:
    if "multilingual-e5" in embedding_model.lower():
        return [f"passage: {text}" for text in texts]
    return texts


def parse_nr_topics(value: str):
    if value == "auto":
        return value
    try:
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--nr-topics must be 'auto' or an integer.") from exc


def main() -> None:
    args = parse_args()

    try:
        import pandas as pd
        from bertopic import BERTopic
        from hdbscan import HDBSCAN
        from sentence_transformers import SentenceTransformer
        from sklearn.feature_extraction.text import CountVectorizer
        from umap import UMAP
    except ImportError as exc:
        raise SystemExit(
            "Missing topic-modeling dependencies. Install them with:\n"
            "python3 -m pip install -r topic_classification/requirements-topic.txt"
        ) from exc

    if not args.sample.exists():
        raise FileNotFoundError(args.sample)

    data = pd.read_csv(args.sample)
    if args.text_column not in data.columns:
        raise ValueError(f"Text column not found: {args.text_column}")

    data[args.text_column] = data[args.text_column].map(normalize_text)
    data = data[data[args.text_column].str.strip().ne("")].copy()
    if args.max_docs is not None:
        data = data.sample(n=min(args.max_docs, len(data)), random_state=args.random_state).copy()

    docs = data[args.text_column].tolist()
    print(f"Documents for BERTopic: {len(docs):,}", flush=True)

    embedder = SentenceTransformer(args.embedding_model)
    embeddings = embedder.encode(
        format_texts_for_embedding(docs, args.embedding_model),
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=args.random_state,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=args.min_topic_size,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    vectorizer_model = CountVectorizer(
        lowercase=True,
        min_df=5,
        max_df=0.80,
        ngram_range=(1, 2),
    )

    topic_model = BERTopic(
        language="multilingual",
        embedding_model=embedder,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        min_topic_size=args.min_topic_size,
        nr_topics=parse_nr_topics(str(args.nr_topics)),
        calculate_probabilities=False,
        verbose=True,
    )

    topics, _ = topic_model.fit_transform(docs, embeddings)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(args.output_dir / "topic_info.csv", index=False)

    data["topic"] = topics
    data.to_csv(args.output_dir / "document_topics.csv.gz", index=False, compression="gzip")

    topic_model.save(
        args.output_dir / "topic_model",
        serialization="safetensors",
        save_ctfidf=True,
        save_embedding_model=False,
    )

    print(f"Saved topic info: {args.output_dir / 'topic_info.csv'}", flush=True)
    print(f"Saved document topics: {args.output_dir / 'document_topics.csv.gz'}", flush=True)
    print(f"Saved BERTopic model: {args.output_dir / 'topic_model'}", flush=True)


if __name__ == "__main__":
    main()
