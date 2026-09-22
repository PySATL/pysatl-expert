"""Select fixed HRF features from one frozen training dataset."""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from pysatl_expert.models.feature_selection_contract import (
    build_selection_document,
    validate_selection_document,
)
from pysatl_expert.models.feature_vector import FeatureVector
from pysatl_expert.models.hierarchical_model import HierarchicalExpertModel
from scripts.training_data import load_frozen_training_partition


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("pysatl_expert/config/feature_selection.json"),
    )
    parser.add_argument(
        "--families",
        type=Path,
        default=Path("pysatl_expert/config/domain_distribution_families.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/hrf/model/selected_features.json"),
    )
    parser.add_argument("--workers", type=_positive_int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _resolve_path(path: str | Path, project_root: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else project_root / path


def _validate_frozen_dataset(dataset: object) -> None:
    if not isinstance(dataset, dict):
        raise ValueError("dataset configuration is required")
    if dataset.get("kind") != "frozen_csv":
        raise ValueError("Feature selection requires a frozen CSV dataset")
    if not isinstance(dataset.get("path"), str) or not dataset["path"].strip():
        raise ValueError("dataset.path must be a non-empty path")
    test_size = dataset.get("outer_test_size")
    if isinstance(test_size, bool) or not isinstance(test_size, (float, int)):
        raise ValueError("dataset.outer_test_size must be between 0 and 1")
    if not 0 < test_size < 1:
        raise ValueError("dataset.outer_test_size must be between 0 and 1")
    outer_seed = dataset.get("outer_seed")
    if not isinstance(outer_seed, int) or isinstance(outer_seed, bool):
        raise ValueError("dataset.outer_seed must be an integer")


def validate_run_config(
    config: object,
    family_map: dict[str, list[str]],
) -> dict[str, Any]:
    """Validate the tracked research decision before loading large datasets."""
    if not isinstance(config, dict):
        raise ValueError("Feature selection config must be a JSON object")
    stage1_count = config.get("stage1_feature_count")
    stage2_counts = config.get("stage2_feature_counts")
    expected_stage2 = {
        family for family, members in family_map.items() if len(members) > 1
    }
    if not isinstance(stage1_count, int) or stage1_count < 1:
        raise ValueError("stage1_feature_count must be positive")
    if not isinstance(stage2_counts, dict) or set(stage2_counts) != expected_stage2:
        raise ValueError("stage2_feature_counts must cover every multi-distribution family")
    if any(not isinstance(count, int) or count < 1 for count in stage2_counts.values()):
        raise ValueError("Every Stage 2 feature count must be positive")
    if config.get("stage1_missing_policy") != "allow_nan_and_report":
        raise ValueError("Stage 1 missing policy must be allow_nan_and_report")
    if config.get("stage2_missing_policy") != "complete_within_family":
        raise ValueError("Stage 2 missing policy must be complete_within_family")

    dataset = config.get("dataset")
    forest = config.get("forest")
    _validate_frozen_dataset(dataset)
    if not isinstance(forest, dict):
        raise ValueError("forest configuration is required")
    if forest.get("random_state") != 42 or forest.get("selection_max_depth") != 15:
        raise ValueError("Forest selection must use random_state=42 and max_depth=15")
    if not isinstance(forest.get("estimators"), int) or forest["estimators"] < 1:
        raise ValueError("forest.estimators must be positive")
    return config


def build_missingness_audit(
    training: pd.DataFrame,
    targets: pd.Series,
    family_map: dict[str, list[str]],
    stage1_features: list[str],
    stage2_features: dict[str, list[str]],
) -> dict[str, Any]:
    """Record the exact missingness policy applied during feature selection."""
    columns = FeatureVector.TRAINING_FEATURE_NAMES
    stage1_rates = training[stage1_features].isna().mean()
    stage2_audit: dict[str, Any] = {}
    for family, members in family_map.items():
        if family not in stage2_features:
            continue
        family_rows = training[targets.isin(members)]
        missing = family_rows[columns].columns[
            family_rows[columns].isna().any()
        ].tolist()
        selected_rates = family_rows[stage2_features[family]].isna().mean()
        stage2_audit[family] = {
            "eligible_feature_count": len(columns) - len(missing),
            "excluded_missing_features": missing,
            "selected_missing_rates": selected_rates.to_dict(),
        }
    return {
        "stage1": {
            "eligible_feature_count": len(columns),
            "selected_missing_rates": stage1_rates.to_dict(),
        },
        "stage2": stage2_audit,
    }


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    project_root = Path(__file__).parents[1]
    config_path = _resolve_path(args.config, project_root)
    families_path = _resolve_path(args.families, project_root)
    output_path = _resolve_path(args.output, project_root)
    if output_path.exists() and not args.overwrite:
        parser.error(f"output already exists: {output_path}; pass --overwrite to replace it")

    with families_path.open(encoding="utf-8") as source:
        family_map = json.load(source)
    with config_path.open(encoding="utf-8") as source:
        config = validate_run_config(json.load(source), family_map)

    dataset_config = dict(config["dataset"])
    dataset_path = _resolve_path(dataset_config["path"], project_root)
    logger.info(
        "Approved selection: Stage 1=%d; Stage 2=%s; dataset=%s",
        config["stage1_feature_count"],
        config["stage2_feature_counts"],
        dataset_path,
    )
    training, outer_test_rows = load_frozen_training_partition(dataset_path)
    logger.info(
        "Feature selection uses %d mixture rows; %d outer-test rows remain untouched",
        len(training),
        outer_test_rows,
    )

    targets = training.pop("Target")
    selector = HierarchicalExpertModel(family_map)
    stage1_features, stage2_features = selector.select_features(
        training,
        targets,
        n_stage1=config["stage1_feature_count"],
        n_stage2=config["stage2_feature_counts"],
        complete_stage2=True,
        n_estimators=config["forest"]["estimators"],
        n_jobs=args.workers,
    )
    document = build_selection_document(
        family_map=family_map,
        feature_schema=FeatureVector.FEATURE_NAMES,
        stage1_features=stage1_features,
        stage2_features=stage2_features,
        test_size=dataset_config["outer_test_size"],
        random_state=dataset_config["outer_seed"],
        stage1_count=config["stage1_feature_count"],
        stage2_count=config["stage2_feature_counts"],
        estimators=config["forest"]["estimators"],
        dataset_config=dataset_config,
        stage1_missing_policy=config["stage1_missing_policy"],
        stage2_missing_policy=config["stage2_missing_policy"],
    )
    document["selection_audit"] = build_missingness_audit(
        training, targets, family_map, stage1_features, stage2_features
    )
    document["split"].update(
        {"training_rows": len(training), "outer_test_rows": outer_test_rows}
    )
    validate_selection_document(document, family_map, FeatureVector.FEATURE_NAMES)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8") as output:
        json.dump(document, output, indent=2, ensure_ascii=False)
    temporary_path.replace(output_path)
    logger.info("Selected feature contract saved to %s", output_path)


if __name__ == "__main__":
    main()
