"""Fit compact BERTopic topics to language-neutral English abstractions."""

from __future__ import annotations

import argparse
import json
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
        "--umap-neighbors",
        type=int,
        default=15,
        help=(
            "UMAP neighborhood size for a single kmeans or hdbscan fit. "
            "The hdbscan-stability selector instead uses "
            "--search-umap-neighbors."
        ),
    )
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
        choices=["kmeans", "hdbscan", "hdbscan-stability"],
        default="hdbscan",
        help=(
            "Clustering algorithm. K-means produces exactly --nr-topics broad "
            "descriptive topics; HDBSCAN estimates a density-based count; "
            "hdbscan-stability selects a density specification by resampling "
            "without fixing the topic count."
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
    parser.add_argument(
        "--search-umap-neighbors",
        nargs="+",
        type=int,
        default=[15, 30, 50],
        help="UMAP neighborhood sizes compared by hdbscan-stability.",
    )
    parser.add_argument(
        "--search-min-topic-sizes",
        nargs="+",
        type=int,
        default=[30, 40, 60, 80, 120],
        help="Minimum HDBSCAN cluster sizes compared by hdbscan-stability.",
    )
    parser.add_argument(
        "--search-min-samples",
        nargs="+",
        type=int,
        default=[2, 5, 10],
        help="HDBSCAN density requirements compared by hdbscan-stability.",
    )
    parser.add_argument(
        "--stability-repeats",
        type=int,
        default=5,
        help="Number of 80-percent resamples used to assess partition stability.",
    )
    parser.add_argument("--stability-subsample", type=float, default=0.80)
    parser.add_argument("--selection-min-topics", type=int, default=4)
    parser.add_argument("--selection-max-topics", type=int, default=12)
    parser.add_argument("--selection-max-outlier-share", type=float, default=0.45)
    parser.add_argument(
        "--selection-max-largest-topic-share", type=float, default=0.40
    )
    parser.add_argument("--selection-max-country-nmi", type=float, default=0.25)
    parser.add_argument(
        "--selection-min-resample-common-share", type=float, default=0.35
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


def candidate_meets_guardrails(
    *,
    topic_count: int,
    outlier_share: float,
    largest_topic_share: float,
    country_nmi: float,
    minimum_topics: int,
    maximum_topics: int,
    maximum_outlier_share: float,
    maximum_largest_topic_share: float,
    maximum_country_nmi: float,
    mean_resample_common_share: float = 1.0,
    minimum_resample_common_share: float = 0.0,
) -> bool:
    """Apply predeclared descriptive-adequacy limits without selecting a count."""
    import math

    return (
        minimum_topics <= topic_count <= maximum_topics
        and outlier_share <= maximum_outlier_share
        and largest_topic_share <= maximum_largest_topic_share
        and math.isfinite(country_nmi)
        and country_nmi <= maximum_country_nmi
        and mean_resample_common_share >= minimum_resample_common_share
    )


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
    "hdbscan_stability_candidates.csv",
    "hdbscan_stability_selection.json",
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
        from sklearn.metrics import (
            adjusted_rand_score,
            normalized_mutual_info_score,
            silhouette_score,
        )
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

    requested_topics = parse_nr_topics(str(args.nr_topics))
    selected_spec = None
    candidate_diagnostics_path = None
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
        selected_umap_neighbors = args.umap_neighbors
    elif args.clusterer == "hdbscan":
        cluster_model = HDBSCAN(
            min_cluster_size=args.min_topic_size,
            min_samples=args.hdbscan_min_samples,
            metric="euclidean",
            cluster_selection_method=args.cluster_selection_method,
            prediction_data=True,
        )
        bertopic_nr_topics = requested_topics
        selected_umap_neighbors = args.umap_neighbors
    else:
        if requested_topics is not None:
            raise ValueError(
                "hdbscan-stability requires --nr-topics none because the topic "
                "count must be selected by density and stability rather than "
                "post-hoc reduction."
            )
        if not 0.5 <= args.stability_subsample < 1.0:
            raise ValueError("--stability-subsample must be in [0.5, 1.0).")
        if args.stability_repeats < 2:
            raise ValueError("--stability-repeats must be at least 2.")
        if args.selection_min_topics < 2:
            raise ValueError("--selection-min-topics must be at least 2.")
        if args.selection_max_topics < args.selection_min_topics:
            raise ValueError(
                "--selection-max-topics must be at least --selection-min-topics."
            )

        countries = data["country"].fillna("missing").astype(str).to_numpy()
        rng = np.random.default_rng(args.random_state)
        subsample_n = max(2, int(round(len(docs) * args.stability_subsample)))
        resamples = [
            np.sort(rng.choice(len(docs), size=subsample_n, replace=False))
            for _ in range(args.stability_repeats)
        ]
        candidate_rows = []

        for n_neighbors in sorted(set(args.search_umap_neighbors)):
            if not 2 <= n_neighbors < len(docs):
                raise ValueError(
                    "Each --search-umap-neighbors value must be at least 2 and "
                    "smaller than the number of documents."
                )
            selector_umap = UMAP(
                n_neighbors=n_neighbors,
                n_components=5,
                min_dist=0.0,
                metric="cosine",
                random_state=args.random_state,
            )
            # Match BERTopic's final training path: UMAP uses fit_transform on
            # the complete training set before HDBSCAN is fitted.
            reduced = selector_umap.fit_transform(embeddings)
            resampled_geometries = []
            for indices in resamples:
                resample_umap = UMAP(
                    n_neighbors=n_neighbors,
                    n_components=5,
                    min_dist=0.0,
                    metric="cosine",
                    random_state=args.random_state,
                )
                resampled_reduced = resample_umap.fit_transform(
                    embeddings[indices]
                )
                resampled_geometries.append((indices, resampled_reduced))

            for min_topic_size in sorted(set(args.search_min_topic_sizes)):
                if not 2 <= min_topic_size < len(docs):
                    raise ValueError(
                        "Each --search-min-topic-sizes value must be at least 2 "
                        "and smaller than the number of documents."
                    )
                for min_samples in sorted(set(args.search_min_samples)):
                    if min_samples < 1:
                        raise ValueError(
                            "Each --search-min-samples value must be positive."
                        )
                    for selection_method in ["leaf", "eom"]:
                        candidate = HDBSCAN(
                            min_cluster_size=min_topic_size,
                            min_samples=min_samples,
                            metric="euclidean",
                            cluster_selection_method=selection_method,
                            prediction_data=True,
                            gen_min_span_tree=True,
                        )
                        labels = candidate.fit_predict(reduced)
                        inlier_mask = labels != -1
                        topic_ids, topic_counts = np.unique(
                            labels[inlier_mask], return_counts=True
                        )
                        topic_count = int(len(topic_ids))
                        outlier_share = float(1.0 - inlier_mask.mean())
                        largest_topic_share = (
                            float(topic_counts.max() / topic_counts.sum())
                            if topic_count
                            else 1.0
                        )
                        if 1 < topic_count < int(inlier_mask.sum()):
                            country_nmi = float(
                                normalized_mutual_info_score(
                                    countries[inlier_mask], labels[inlier_mask]
                                )
                            )
                        else:
                            country_nmi = float("nan")

                        persistence = np.asarray(
                            getattr(candidate, "cluster_persistence_", []),
                            dtype=float,
                        )
                        if len(persistence) == len(topic_counts) and len(persistence):
                            weighted_persistence = float(
                                np.average(persistence, weights=topic_counts)
                            )
                        else:
                            weighted_persistence = float("nan")
                        try:
                            relative_validity = float(candidate.relative_validity_)
                        except (AttributeError, ValueError):
                            relative_validity = float("nan")

                        bootstrap_ari = []
                        bootstrap_common_share = []
                        for indices, resampled_fit_reduced in resampled_geometries:
                            try:
                                resampled_labels = HDBSCAN(
                                    min_cluster_size=min_topic_size,
                                    min_samples=min_samples,
                                    metric="euclidean",
                                    cluster_selection_method=selection_method,
                                ).fit_predict(resampled_fit_reduced)
                            except ValueError:
                                bootstrap_ari.append(0.0)
                                bootstrap_common_share.append(0.0)
                                continue
                            baseline_labels = labels[indices]
                            common = (baseline_labels != -1) & (
                                resampled_labels != -1
                            )
                            bootstrap_common_share.append(float(common.mean()))
                            if (
                                int(common.sum()) >= 20
                                and len(np.unique(baseline_labels[common])) >= 2
                                and len(np.unique(resampled_labels[common])) >= 2
                            ):
                                bootstrap_ari.append(
                                    float(
                                        adjusted_rand_score(
                                            baseline_labels[common],
                                            resampled_labels[common],
                                        )
                                    )
                                )
                            else:
                                bootstrap_ari.append(0.0)

                        adequate = candidate_meets_guardrails(
                            topic_count=topic_count,
                            outlier_share=outlier_share,
                            largest_topic_share=largest_topic_share,
                            country_nmi=country_nmi,
                            minimum_topics=args.selection_min_topics,
                            maximum_topics=args.selection_max_topics,
                            maximum_outlier_share=args.selection_max_outlier_share,
                            maximum_largest_topic_share=(
                                args.selection_max_largest_topic_share
                            ),
                            maximum_country_nmi=args.selection_max_country_nmi,
                            mean_resample_common_share=float(
                                np.mean(bootstrap_common_share)
                            ),
                            minimum_resample_common_share=(
                                args.selection_min_resample_common_share
                            ),
                        )
                        candidate_rows.append(
                            {
                                "umap_n_neighbors": int(n_neighbors),
                                "min_topic_size": int(min_topic_size),
                                "min_samples": int(min_samples),
                                "cluster_selection_method": selection_method,
                                "topics": topic_count,
                                "inliers": int(inlier_mask.sum()),
                                "outliers": int((~inlier_mask).sum()),
                                "outlier_share": outlier_share,
                                "largest_inlier_topic_share": largest_topic_share,
                                "country_topic_nmi": country_nmi,
                                "weighted_cluster_persistence": weighted_persistence,
                                "relative_validity": relative_validity,
                                "mean_resample_ari": float(np.mean(bootstrap_ari)),
                                "min_resample_ari": float(np.min(bootstrap_ari)),
                                "mean_resample_common_share": float(
                                    np.mean(bootstrap_common_share)
                                ),
                                "adequate": bool(adequate),
                            }
                        )

        candidates = pd.DataFrame(candidate_rows)
        candidate_diagnostics_path = (
            args.output_dir / "hdbscan_stability_candidates.csv"
        )
        candidates.sort_values(
            ["adequate", "mean_resample_ari", "weighted_cluster_persistence"],
            ascending=[False, False, False],
        ).to_csv(candidate_diagnostics_path, index=False)
        adequate = candidates[candidates["adequate"]].copy()
        if adequate.empty:
            raise RuntimeError(
                "No HDBSCAN candidate met the pre-specified adequacy guardrails. "
                f"Inspect {candidate_diagnostics_path}; do not publish a forced "
                "topic solution."
            )
        selected = adequate.sort_values(
            [
                "mean_resample_ari",
                "mean_resample_common_share",
                "weighted_cluster_persistence",
                "relative_validity",
                "outlier_share",
                "topics",
            ],
            ascending=[False, False, False, False, True, True],
            na_position="last",
        ).iloc[0]
        integer_keys = {
            "umap_n_neighbors",
            "min_topic_size",
            "min_samples",
            "topics",
            "inliers",
            "outliers",
        }
        float_keys = {
            "outlier_share",
            "largest_inlier_topic_share",
            "country_topic_nmi",
            "weighted_cluster_persistence",
            "relative_validity",
            "mean_resample_ari",
            "min_resample_ari",
            "mean_resample_common_share",
        }
        selected_spec = {}
        for key in selected.index:
            if key == "adequate":
                continue
            value = selected[key]
            if pd.isna(value):
                selected_spec[key] = None
            elif key in integer_keys:
                selected_spec[key] = int(value)
            elif key in float_keys:
                selected_spec[key] = float(value)
            else:
                selected_spec[key] = str(value)
        candidates["selected"] = (
            candidates["umap_n_neighbors"].eq(selected["umap_n_neighbors"])
            & candidates["min_topic_size"].eq(selected["min_topic_size"])
            & candidates["min_samples"].eq(selected["min_samples"])
            & candidates["cluster_selection_method"].eq(
                selected["cluster_selection_method"]
            )
        )
        candidates.sort_values(
            ["selected", "adequate", "mean_resample_ari", "weighted_cluster_persistence"],
            ascending=[False, False, False, False],
        ).to_csv(candidate_diagnostics_path, index=False)
        selection_path = args.output_dir / "hdbscan_stability_selection.json"
        selection_path.write_text(
            json.dumps(
                {
                    "selection_rule": (
                        "Among candidates satisfying the declared topic-count, "
                        "coverage, dominance, and country-NMI guardrails: maximize "
                        "mean resample adjusted Rand index, then the mean share "
                        "mutually assigned across refits, weighted cluster persistence, "
                        "and relative validity; break remaining ties by baseline "
                        "coverage and the more compact solution."
                    ),
                    "guardrails": {
                        "minimum_topics": args.selection_min_topics,
                        "maximum_topics": args.selection_max_topics,
                        "maximum_outlier_share": args.selection_max_outlier_share,
                        "maximum_largest_inlier_topic_share": args.selection_max_largest_topic_share,
                        "maximum_country_topic_nmi": args.selection_max_country_nmi,
                        "minimum_mean_resample_common_share": (
                            args.selection_min_resample_common_share
                        ),
                    },
                    "selected": selected_spec,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print("Selected stability-based HDBSCAN specification:", flush=True)
        print(pd.Series(selected_spec).to_string(), flush=True)

        selected_umap_neighbors = int(selected["umap_n_neighbors"])
        cluster_model = HDBSCAN(
            min_cluster_size=int(selected["min_topic_size"]),
            min_samples=int(selected["min_samples"]),
            metric="euclidean",
            cluster_selection_method=str(selected["cluster_selection_method"]),
            prediction_data=True,
            gen_min_span_tree=True,
        )
        bertopic_nr_topics = None

    if not 2 <= selected_umap_neighbors < len(docs):
        raise ValueError(
            "The selected UMAP neighborhood size must be at least 2 and smaller "
            "than the number of modeled documents."
        )

    umap_model = UMAP(
        n_neighbors=selected_umap_neighbors,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=args.random_state,
    )
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
    if selected_spec is not None:
        actual_outlier_share = float(1.0 - inlier_mask.mean())
        _, actual_counts = np.unique(topic_array[inlier_mask], return_counts=True)
        actual_largest_share = float(actual_counts.max() / actual_counts.sum())
        actual_country_nmi = float(
            normalized_mutual_info_score(
                data["country"].fillna("missing").astype(str).to_numpy()[inlier_mask],
                topic_array[inlier_mask],
            )
        )
        if not candidate_meets_guardrails(
            topic_count=len(observed_topic_ids),
            outlier_share=actual_outlier_share,
            largest_topic_share=actual_largest_share,
            country_nmi=actual_country_nmi,
            minimum_topics=args.selection_min_topics,
            maximum_topics=args.selection_max_topics,
            maximum_outlier_share=args.selection_max_outlier_share,
            maximum_largest_topic_share=args.selection_max_largest_topic_share,
            maximum_country_nmi=args.selection_max_country_nmi,
            mean_resample_common_share=float(
                selected_spec["mean_resample_common_share"]
            ),
            minimum_resample_common_share=args.selection_min_resample_common_share,
        ):
            raise RuntimeError(
                "The final fit of the selected specification did not satisfy the "
                "declared adequacy guardrails. Preserve the diagnostics and "
                "inspect the environment before publishing."
            )
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

    manifest_outputs = {
        "topic_info": args.output_dir / "topic_info.csv",
        "document_topics": args.output_dir / "document_topics.csv.gz",
    }
    if candidate_diagnostics_path is not None:
        manifest_outputs["hdbscan_stability_candidates"] = candidate_diagnostics_path
        manifest_outputs["hdbscan_stability_selection"] = (
            args.output_dir / "hdbscan_stability_selection.json"
        )

    write_run_manifest(
        args.output_dir,
        script_name=Path(__file__).name,
        args=args,
        inputs=manifest_inputs,
        outputs=manifest_outputs,
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
            "selected_umap_neighbors": selected_umap_neighbors,
            "stability_selected_specification": selected_spec,
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
