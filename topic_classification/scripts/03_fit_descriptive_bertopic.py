"""Fit compact BERTopic topics to language-neutral English abstractions."""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import unicodedata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from political_classifier.reproducibility import embedding_revision
from topic_classification.provenance import validate_verified_topic_sample
from topic_classification.scripts._impl.reproducibility import write_run_manifest


DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-large"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit compact BERTopic topics to verified English abstractions."
    )
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--text-column", default="article_text")
    parser.add_argument(
        "--status-column",
        default=None,
        help="Optional abstraction-status column used to filter model inputs.",
    )
    parser.add_argument(
        "--include-status",
        nargs="+",
        default=None,
        help="Status values retained when --status-column is supplied.",
    )
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--min-topic-size",
        type=int,
        default=25,
        help="Larger values produce fewer, broader raw clusters.",
    )
    parser.add_argument(
        "--nr-topics",
        default="auto",
        help=(
            "Topic reduction: integer for a target, 'auto' for automatic "
            "reduction, or 'none' to preserve the original clusters."
        ),
    )
    parser.add_argument(
        "--clusterer",
        choices=["kmeans", "hdbscan"],
        default="hdbscan",
        help=(
            "Clustering algorithm. K-means produces exactly --nr-topics broad "
            "descriptive topics; HDBSCAN estimates a density-based count."
        ),
    )
    parser.add_argument(
        "--cluster-selection-method",
        choices=["leaf", "eom"],
        default="leaf",
        help="HDBSCAN-only cluster selection method.",
    )
    parser.add_argument(
        "--hdbscan-min-samples",
        type=int,
        default=10,
        help=(
            "HDBSCAN core-density requirement. Values below min topic size can "
            "recover stable smaller structures without fixing the topic count."
        ),
    )
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--allow-unverified-sample",
        action="store_true",
        help="Permit an exploratory sample without final-classifier provenance.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove known outputs from an earlier run in this output directory.",
    )
    return parser.parse_args()


def normalize_text(text: object) -> str:
    if not isinstance(text, str):
        return ""
    try:
        from ftfy import fix_text

        text = fix_text(text)
    except ImportError:
        pass
    text = unicodedata.normalize("NFKC", text).replace("\u00a0", " ")
    text = "".join(
        char
        for char in text
        if char in "\n\t" or unicodedata.category(char) != "Cc"
    )
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def format_texts_for_embedding(texts: list[str], embedding_model: str) -> list[str]:
    if "multilingual-e5" in embedding_model.lower():
        return [f"passage: {text}" for text in texts]
    return texts


def parse_nr_topics(value: str):
    normalized = value.strip().lower()
    if normalized == "auto":
        return normalized
    if normalized in {"none", "null"}:
        return None
    try:
        return int(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--nr-topics must be 'auto', 'none', or an integer."
        ) from exc


GENERATED_OUTPUTS = [
    "topic_info.csv",
    "document_topics.csv.gz",
    "topic_model",
    "run_manifest.json",
    "topic_labels_llm.csv",
    "topic_labels_llm_audit.jsonl",
    "topic_labels_run_manifest.json",
    "topic_groups_llm.csv",
    "topic_groups_llm_audit.json",
    "topic_group_summaries_llm.csv",
    "topic_groups_run_manifest.json",
    "inspection_tables",
    "inspection_notebooks",
    "visualizations",
    "topic_model_build_summary.json",
    "topic_model_output_manifest.json",
    "00_LATEST_TOPIC_MODEL_BUILD.txt",
    "descriptive_outputs",
    "descriptive_topic_output_manifest.json",
    "00_LATEST_DESCRIPTIVE_TOPIC_BUILD.txt",
]


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    existing = [output_dir / name for name in GENERATED_OUTPUTS if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "BERTopic outputs already exist. Pass --overwrite for a deliberate clean "
            "rebuild: "
            + ", ".join(str(path) for path in existing)
        )
    if overwrite:
        for path in existing:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()

    if not args.sample.exists():
        raise FileNotFoundError(args.sample)
    sample_provenance = None
    if args.allow_unverified_sample:
        print("WARNING: fitting an unverified exploratory topic sample.", flush=True)
    else:
        sample_provenance = validate_verified_topic_sample(args.sample)
        upstream = sample_provenance["upstream_classifier"]
        print(
            "Verified topic sample from final classifier: "
            f"threshold={float(upstream['threshold']):.2f}, "
            f"political-corruption N={int(upstream['political_corruption_articles']):,}",
            flush=True,
        )
    prepare_output_dir(args.output_dir, args.overwrite)

    try:
        import numpy as np
        import pandas as pd
        from bertopic import BERTopic
        from hdbscan import HDBSCAN
        from sentence_transformers import SentenceTransformer
        from sklearn.cluster import KMeans
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.metrics import silhouette_score
        from umap import UMAP
    except ImportError as exc:
        raise SystemExit(
            "Missing topic-modeling dependencies. Install them with:\n"
            "python3 -m pip install -r topic_classification/requirements-topic.txt"
        ) from exc

    data = pd.read_csv(args.sample)
    if args.text_column not in data.columns:
        raise ValueError(f"Text column not found: {args.text_column}")

    input_rows = len(data)
    if args.status_column is not None:
        if args.status_column not in data.columns:
            raise ValueError(f"Status column not found: {args.status_column}")
        if not args.include_status:
            raise ValueError("--include-status is required with --status-column.")
        data = data[data[args.status_column].isin(args.include_status)].copy()
        print(
            f"Status filter retained {len(data):,} / {input_rows:,} rows: "
            f"{args.status_column} in {args.include_status}",
            flush=True,
        )

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
    requested_topics = parse_nr_topics(str(args.nr_topics))
    if args.clusterer == "kmeans":
        if requested_topics in {"auto", None}:
            raise ValueError("--clusterer kmeans requires an integer --nr-topics.")
        if not 2 <= requested_topics < len(docs):
            raise ValueError(
                "For K-means, --nr-topics must be at least 2 and smaller than "
                "the number of modeled documents."
            )
        cluster_model = KMeans(
            n_clusters=requested_topics,
            random_state=args.random_state,
            n_init=20,
        )
        bertopic_nr_topics = None
    else:
        cluster_model = HDBSCAN(
            min_cluster_size=args.min_topic_size,
            min_samples=args.hdbscan_min_samples,
            metric="euclidean",
            cluster_selection_method=args.cluster_selection_method,
            prediction_data=True,
        )
        bertopic_nr_topics = requested_topics
    vectorizer_model = CountVectorizer(
        lowercase=True,
        stop_words="english",
        # BERTopic fits this vectorizer to one concatenated document per topic,
        # not to every article. min_df must therefore remain valid even for a
        # deliberately compact solution with only a few topic documents.
        min_df=1,
        max_df=1.0,
        ngram_range=(1, 2),
    )

    topic_model = BERTopic(
        language="english",
        embedding_model=embedder,
        umap_model=umap_model,
        hdbscan_model=cluster_model,
        vectorizer_model=vectorizer_model,
        min_topic_size=args.min_topic_size,
        nr_topics=bertopic_nr_topics,
        calculate_probabilities=False,
        verbose=True,
    )

    topics, _ = topic_model.fit_transform(docs, embeddings)
    topic_array = np.asarray(topics)
    inlier_mask = topic_array != -1
    observed_topic_ids = sorted(set(int(topic) for topic in topic_array[inlier_mask]))
    silhouette = None
    if 1 < len(observed_topic_ids) < int(inlier_mask.sum()):
        silhouette = float(
            silhouette_score(
                embeddings[inlier_mask],
                topic_array[inlier_mask],
                metric="cosine",
            )
        )

    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(args.output_dir / "topic_info.csv", index=False)

    data["topic"] = topics
    data.to_csv(args.output_dir / "document_topics.csv.gz", index=False, compression="gzip")

    topic_model.save(
        args.output_dir / "topic_model",
        serialization="safetensors",
        save_ctfidf=True,
        save_embedding_model=args.embedding_model,
    )

    upstream_classifier = (
        sample_provenance["upstream_classifier"]
        if sample_provenance is not None
        else {"verified_final_classifier": False}
    )
    inlier_documents = int(sum(topic != -1 for topic in topics))
    outlier_documents = int(len(topics) - inlier_documents)
    non_outlier_topics = int(topic_info["Topic"].ne(-1).sum())
    manifest_inputs = {"sample": args.sample}
    if sample_provenance is not None:
        manifest_inputs["sample_manifest"] = sample_provenance["manifest_path"]

    write_run_manifest(
        args.output_dir,
        script_name=Path(__file__).name,
        args=args,
        inputs=manifest_inputs,
        outputs={
            "topic_info": args.output_dir / "topic_info.csv",
            "document_topics": args.output_dir / "document_topics.csv.gz",
        },
        extra={
            "documents_for_model": int(len(docs)),
            "input_rows_before_status_filter": int(input_rows),
            "status_column": args.status_column,
            "included_statuses": args.include_status,
            "embedding_model": args.embedding_model,
            "embedding_model_revision": embedding_revision(embedder),
            "random_state": args.random_state,
            "min_topic_size": args.min_topic_size,
            "nr_topics": args.nr_topics,
            "clusterer": args.clusterer,
            "cluster_selection_method": args.cluster_selection_method,
            "hdbscan_min_samples": args.hdbscan_min_samples,
            "silhouette_cosine_original_embeddings": silhouette,
            "bertopic_model_dir": str(args.output_dir / "topic_model"),
            "non_outlier_topics": non_outlier_topics,
            "inlier_documents": inlier_documents,
            "outlier_documents": outlier_documents,
            "outlier_share": outlier_documents / len(docs),
            "topic_sample_manifest_sha256": (
                sample_provenance["manifest_sha256"]
                if sample_provenance is not None
                else None
            ),
            "upstream_classifier": upstream_classifier,
        },
    )

    print(f"Saved topic info: {args.output_dir / 'topic_info.csv'}", flush=True)
    print(f"Saved document topics: {args.output_dir / 'document_topics.csv.gz'}", flush=True)
    print(f"Saved BERTopic model: {args.output_dir / 'topic_model'}", flush=True)


if __name__ == "__main__":
    main()
