"""Training script for 2-stage Hierarchical Expert Model."""

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from pysatl_expert.models.feature_selection_contract import validate_selection_document
from pysatl_expert.models.feature_vector import FeatureVector
from pysatl_expert.models.hierarchical_model import HierarchicalExpertModel
from pysatl_expert.models.model_manifest import (
    manifest_path_for,
    write_model_manifest,
)
from scripts.training_data import load_frozen_training_dataset


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the training CLI with explicit, non-destructive artifact paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--families",
        type=Path,
        default=Path("pysatl_expert/config/domain_distribution_families.json"),
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("artifacts/hrf/model/selected_features.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/hrf/model/rf_expert_model.joblib"),
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("artifacts/hrf/model/model_metrics.json"),
    )
    parser.add_argument("--estimators", type=_positive_int, default=200)
    parser.add_argument("--workers", type=_positive_int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def prepare_training_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Validate and project a raw dataset onto the active training schema.

    Completed datasets may still contain columns that were explicitly excluded after
    generation.  Those known historical columns are discarded; unknown columns and
    missing active features remain hard schema errors.
    """
    if "Target" not in df.columns:
        raise ValueError("Dataset has no Target column")

    actual_features = [column for column in df.columns if column != "Target"]
    expected_features = FeatureVector.FEATURE_NAMES
    missing_features = sorted(set(expected_features) - set(actual_features))
    if missing_features:
        raise ValueError(
            "Dataset is missing active features: " + ", ".join(missing_features)
        )
    extra_features = set(actual_features) - set(expected_features)
    unexpected_features = sorted(
        extra_features - FeatureVector.EXCLUDED_TRAINING_FEATURES
    )
    if unexpected_features:
        raise ValueError(
            "Dataset contains unexpected features: " + ", ".join(unexpected_features)
        )

    features = df[expected_features].replace([np.inf, -np.inf], np.nan)
    return features, df["Target"]


def load_training_dataset(
    path: Path,
    samples_per_class: int | None = None,
    expected_targets: set[str] | None = None,
    chunksize: int = 50_000,
) -> pd.DataFrame:
    """Load only active float32 features, optionally taking a balanced class prefix."""
    selected_columns = [*FeatureVector.FEATURE_NAMES, "Target"]
    numeric_dtypes = {
        feature_name: np.float32 for feature_name in FeatureVector.FEATURE_NAMES
    }
    read_options = {
        "usecols": selected_columns,
        "dtype": numeric_dtypes,
    }
    if samples_per_class is None:
        return pd.read_csv(path, **read_options)

    selected_parts = []
    selected_counts: dict[str, int] = {}
    for chunk in pd.read_csv(path, chunksize=chunksize, **read_options):
        for target, target_rows in chunk.groupby("Target", sort=False):
            target_name = str(target)
            remaining = samples_per_class - selected_counts.get(target_name, 0)
            if remaining <= 0:
                continue
            selected = target_rows.head(remaining)
            selected_parts.append(selected)
            selected_counts[target_name] = selected_counts.get(target_name, 0) + len(selected)
        if expected_targets and all(
            selected_counts.get(target, 0) >= samples_per_class
            for target in expected_targets
        ):
            break

    required_targets = expected_targets or set(selected_counts)
    incomplete = {
        target: selected_counts.get(target, 0)
        for target in required_targets
        if selected_counts.get(target, 0) < samples_per_class
    }
    if incomplete:
        raise ValueError(f"Dataset has insufficient smoke rows: {incomplete}")
    if not selected_parts:
        raise ValueError("Dataset contains no training rows")
    return pd.concat(selected_parts, ignore_index=True)[selected_columns]


def save_training_artifacts(
    *,
    model_path: Path,
    metrics_path: Path,
    model: HierarchicalExpertModel,
    metrics: dict[str, object],
    feature_selection: dict[str, object],
) -> dict[str, object]:
    """Atomically save model and metrics, then complete the bundle manifest."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_model_path = model_path.with_suffix(f"{model_path.suffix}.tmp")
    temporary_metrics_path = metrics_path.with_suffix(f"{metrics_path.suffix}.tmp")
    joblib.dump(model, temporary_model_path)
    with temporary_metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2, ensure_ascii=False)
        metrics_file.write("\n")
    temporary_model_path.replace(model_path)
    temporary_metrics_path.replace(metrics_path)
    return write_model_manifest(model_path, model, feature_selection)


def _classification_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, object]:
    """Return JSON-serializable evaluation metrics for one classifier."""
    labels = sorted(str(label) for label in y_true.unique())
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
    }


def evaluate_hierarchical_stages(
    model: HierarchicalExpertModel,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, object]:
    """Evaluate family routing and conditional classifiers independently."""
    if model.stage1_model is None or model.stage1_features is None:
        raise RuntimeError("Hierarchical model must be fitted before evaluation")

    true_families = y_test.map(model.dist_to_family)
    predicted_families = model.stage1_model.predict(X_test[model.stage1_features])
    stage2_metrics: dict[str, dict[str, object]] = {}

    for family_name, members in model.family_map.items():
        family_mask = y_test.isin(members)
        family_target = y_test.loc[family_mask]
        if family_name in model.stage2_models:
            family_model = model.stage2_models[family_name]
            family_features = model.stage2_features[family_name]
            predictions = family_model.predict(X_test.loc[family_mask, family_features])
        else:
            predictions = np.repeat(members[0], len(family_target))
        stage2_metrics[family_name] = _classification_metrics(family_target, predictions)

    return {
        "stage1_family": _classification_metrics(true_families, predicted_families),
        "stage2_by_family": stage2_metrics,
    }


def _resolve_path(path: str | Path, project_root: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else project_root / path


def main(argv: list[str] | None = None):
    """Train HierarchicalExpertModel and evaluate accuracy on test split."""
    parser = build_parser()
    args = parser.parse_args(argv)
    project_root = Path(__file__).parents[1]
    json_path = _resolve_path(args.families, project_root)
    features_path = _resolve_path(args.features, project_root)
    model_path = _resolve_path(args.output, project_root)
    metrics_path = _resolve_path(args.metrics_output, project_root)
    manifest_path = manifest_path_for(model_path)

    if model_path.exists() and not args.overwrite:
        parser.error(f"output already exists: {model_path}; pass --overwrite to replace it")
    if metrics_path.exists() and not args.overwrite:
        parser.error(f"output already exists: {metrics_path}; pass --overwrite to replace it")
    if manifest_path.exists() and not args.overwrite:
        parser.error(f"output already exists: {manifest_path}; pass --overwrite to replace it")

    logger.info(f"Loading family mapping from: {json_path}")
    logger.info(f"Loading selected features from: {features_path}")

    with json_path.open(encoding="utf-8") as f:
        family_map = json.load(f)
    with features_path.open(encoding="utf-8") as f:
        feature_selection = json.load(f)
    validate_selection_document(
        feature_selection,
        family_map,
        FeatureVector.FEATURE_NAMES,
    )
    dataset_config = feature_selection["dataset"]
    dataset_path = _resolve_path(dataset_config["path"], project_root)
    training, outer_test = load_frozen_training_dataset(dataset_path)
    X_train = training[FeatureVector.FEATURE_NAMES]
    y_train = training["Target"]
    X_test = outer_test[FeatureVector.FEATURE_NAMES]
    y_test = outer_test["Target"]

    logger.info(f"Dataset split: {len(X_train)} training samples, {len(X_test)} test samples.")

    metrics: dict[str, object] = {}
    metrics["run_config"] = {
        "family_map": family_map,
        "training_data": dataset_config,
        "estimators": args.estimators,
        "workers": args.workers,
        "selected_features": str(features_path),
        "stage1_feature_count": len(
            feature_selection["selection"]["stage1_features"]
        ),
        "stage2_feature_count": {
            family: len(features)
            for family, features in feature_selection["selection"][
                "stage2_features"
            ].items()
        },
        "test_size": feature_selection["split"]["test_size"],
        "random_state": feature_selection["split"]["random_state"],
    }
    metrics["feature_policy"] = {
        "excluded_training_features": sorted(FeatureVector.EXCLUDED_TRAINING_FEATURES),
        "excluded_numerically_unstable_features": sorted(
            FeatureVector.NUMERICALLY_UNSTABLE_TRAINING_FEATURES
        ),
        "omitted_location_features": ["max", "min"],
        "training_feature_count": len(FeatureVector.TRAINING_FEATURE_NAMES),
        "full_schema_feature_count": len(FeatureVector.FEATURE_NAMES),
    }

    model = HierarchicalExpertModel(family_map)
    model.fit_selected(
        X_train,
        y_train,
        stage1_features=feature_selection["selection"]["stage1_features"],
        stage2_features=feature_selection["selection"]["stage2_features"],
        n_estimators=args.estimators,
        n_jobs=args.workers,
    )

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    hierarchical_metrics = _classification_metrics(y_test, y_pred)
    hierarchical_metrics["stage1_features"] = model.stage1_features
    hierarchical_metrics["stage2_features"] = model.stage2_features
    hierarchical_metrics.update(evaluate_hierarchical_stages(model, X_test, y_test))
    metrics["hierarchical_random_forest"] = hierarchical_metrics

    logger.info(f"Overall Test Accuracy: {acc * 100:.2f}%")
    logger.info(
        "Stage 1 Family Accuracy: "
        f"{hierarchical_metrics['stage1_family']['accuracy'] * 100:.2f}%"
    )
    logger.info(
        "Classification Report:\n"
        + classification_report(y_test, y_pred, digits=4, zero_division=0)
    )

    save_training_artifacts(
        model_path=model_path,
        metrics_path=metrics_path,
        model=model,
        metrics=metrics,
        feature_selection=feature_selection,
    )
    logger.info(f"Successfully saved trained model to '{model_path}'")
    logger.info(f"Evaluation metrics saved to '{metrics_path}'")
    logger.info(f"Model manifest saved to '{manifest_path}'")


if __name__ == "__main__":
    main()
