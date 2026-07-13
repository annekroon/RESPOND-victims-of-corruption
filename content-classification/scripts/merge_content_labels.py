"""Merge article-level content labels into one silver-labelled dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_OUTPUT_DIR = Path(
    "/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/"
    "content_classification"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge zero-shot content labels and derive analysis variables."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--cpi", type=Path, default=Path("output/cpi_country_year_scores.csv"))
    parser.add_argument("--skip-cpi", action="store_true")
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Keep only articles present in all four classifier outputs.",
    )
    return parser.parse_args()


def read_csv(path: Path):
    import pandas as pd

    return pd.read_csv(path, compression="gzip" if path.name.endswith(".gz") else "infer")


def write_csv(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False, compression="gzip" if path.name.endswith(".gz") else None)


def standardize_country(country: object) -> str:
    return str(country).replace("_", " ").strip()


def normalize_yes_no(value: object):
    value = str(value).strip().lower()
    if value in {"yes", "1", "true"}:
        return 1
    if value in {"no", "0", "false"}:
        return 0
    return None


def derive_variables(data):
    data = data.copy()

    data["victim_visible_binary"] = data["victim_visibility"].map(
        {
            "concrete_victim": 1,
            "institutional_societal_victim": 1,
            "no_victim": 0,
        }
    )
    data["concrete_victim_visible"] = data["victim_visibility"].map(
        {
            "concrete_victim": 1,
            "institutional_societal_victim": 0,
            "no_victim": 0,
        }
    )
    data["institutional_societal_victim_visible"] = data["victim_visibility"].map(
        {
            "institutional_societal_victim": 1,
            "concrete_victim": 0,
            "no_victim": 0,
        }
    )
    data["abroad_case_binary"] = data["abroad_case"].map(normalize_yes_no)
    data["accused_actor_visible_binary"] = data["accused_actor_visible"].map(normalize_yes_no)
    data["frame_individualized"] = data["corruption_frame"].map(
        {"individualized": 1, "systemic": 0, "other_or_mixed": 0}
    )
    data["frame_systemic"] = data["corruption_frame"].map(
        {"systemic": 1, "individualized": 0, "other_or_mixed": 0}
    )
    data["frame_other_or_mixed"] = data["corruption_frame"].map(
        {"other_or_mixed": 1, "individualized": 0, "systemic": 0}
    )
    return data


def base_metadata(frames: dict, base_cols: list[str]):
    import pandas as pd

    parts = []
    for frame in frames.values():
        columns = [column for column in base_cols if column in frame.columns]
        if columns:
            parts.append(frame[columns])
    if not parts:
        raise ValueError("No metadata columns found in classifier outputs.")
    return pd.concat(parts, ignore_index=True).drop_duplicates("article_id", keep="first")


def add_cpi_context(data, cpi_path: Path):
    import pandas as pd

    if not cpi_path.exists():
        raise FileNotFoundError(cpi_path)

    cpi = pd.read_csv(cpi_path)
    cpi["country_join"] = cpi["country"].map(standardize_country)
    cpi["year"] = pd.to_numeric(cpi["year"], errors="coerce").astype("Int64")
    cpi["perceived_corruption"] = 100 - pd.to_numeric(cpi["cpi_score"], errors="coerce")
    cpi["year"] = cpi["year"] + 1
    cpi = cpi.rename(
        columns={
            "cpi_score": "cpi_score_lag1",
            "cpi_rank": "cpi_rank_lag1",
            "perceived_corruption": "perceived_corruption_lag1",
        }
    )
    cpi = cpi[
        [
            "country_join",
            "year",
            "cpi_score_lag1",
            "cpi_rank_lag1",
            "perceived_corruption_lag1",
        ]
    ]

    data = data.copy()
    data["country_join"] = data["country"].map(standardize_country)
    data["year"] = pd.to_numeric(data["year"], errors="coerce").astype("Int64")
    data = data.merge(cpi, on=["country_join", "year"], how="left")
    data = data.drop(columns=["country_join"])
    return data


def main() -> None:
    args = parse_args()

    victim_path = args.input_dir / "victim_visibility_labels.csv.gz"
    frame_path = args.input_dir / "corruption_frame_labels.csv.gz"
    abroad_path = args.input_dir / "abroad_case_labels.csv.gz"
    accused_path = args.input_dir / "accused_actor_labels.csv.gz"
    output_path = args.output or (args.input_dir / "content_silver_labels_merged.csv.gz")

    frames = {
        "victim": read_csv(victim_path),
        "frame": read_csv(frame_path),
        "abroad": read_csv(abroad_path),
        "accused": read_csv(accused_path),
    }

    base_cols = [
        "article_id",
        "uri",
        "country",
        "year",
        "month",
        "week",
        "source_uri",
        "source.uri",
        "word_count",
        "prob_political_corruption",
        "pred_political_corruption",
    ]
    data = base_metadata(frames, base_cols)

    how = "inner" if args.require_all else "outer"
    for name, frame in frames.items():
        frame = frame.rename(
            columns={
                "llm_error": f"{name}_llm_error",
                "prompt_version": f"{name}_prompt_version",
            }
        )
        label_cols = [
            column
            for column in frame.columns
            if column not in set(base_cols + ["classifier_name"])
        ]
        keep_cols = ["article_id", *label_cols]
        data = data.merge(frame[keep_cols], on="article_id", how=how, suffixes=("", f"_{name}"))

    data = derive_variables(data)
    if not args.skip_cpi:
        data = add_cpi_context(data, args.cpi)

    write_csv(data, output_path)
    print(f"Saved merged content labels: {output_path} ({len(data):,} rows)", flush=True)


if __name__ == "__main__":
    main()
