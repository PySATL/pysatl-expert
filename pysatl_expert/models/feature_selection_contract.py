"""Versioned contract between feature selection and final model training."""

from typing import Any, TypeGuard

from pysatl_expert.models.feature_vector import FeatureVector


SELECTION_SCHEMA_VERSION = 1


def _is_integer(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_positive_integer(value: object, label: str) -> None:
    if not _is_integer(value) or value < 1:
        raise ValueError(f"{label} must be a positive integer")


def _validate_fraction(value: object, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not 0 < value < 1:
        raise ValueError(f"{label} must be between 0 and 1")


def build_selection_document(
    *,
    family_map: dict[str, list[str]],
    feature_schema: list[str],
    stage1_features: list[str],
    stage2_features: dict[str, list[str]],
    test_size: float,
    random_state: int,
    stage1_count: int,
    stage2_count: dict[str, int],
    estimators: int,
    dataset_config: dict[str, Any],
    stage1_missing_policy: str,
    stage2_missing_policy: str,
) -> dict[str, Any]:
    """Build the versioned handoff contract consumed by final training."""
    return {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "family_map": family_map,
        "feature_schema": feature_schema,
        "dataset": dataset_config,
        "split": {"test_size": test_size, "random_state": random_state},
        "selector": {
            "stage1_feature_count": stage1_count,
            "stage2_feature_counts": stage2_count,
            "estimators": estimators,
            "stage1_missing_policy": stage1_missing_policy,
            "stage2_missing_policy": stage2_missing_policy,
        },
        "selection": {
            "stage1_features": stage1_features,
            "stage2_features": stage2_features,
        },
    }


def _validate_feature_list(
    value: object,
    *,
    label: str,
    known_features: set[str],
) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(feature, str) for feature in value)
    ):
        raise ValueError(f"{label} must be a non-empty list of feature names")
    if len(value) != len(set(value)):
        raise ValueError(f"{label} contains duplicate features")
    unknown = set(value).difference(known_features)
    if unknown:
        raise ValueError(f"{label} contains unknown feature(s): " + ", ".join(sorted(unknown)))
    excluded = set(value).intersection(FeatureVector.EXCLUDED_TRAINING_FEATURES)
    if excluded:
        raise ValueError(f"{label} contains excluded feature(s): " + ", ".join(sorted(excluded)))
    return value


def _validate_dataset_config(document: dict[str, Any]) -> None:
    dataset = document.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("Feature selection dataset configuration is missing")
    required = {"kind", "path", "outer_test_size", "outer_seed"}
    missing = required.difference(dataset)
    if missing:
        raise ValueError(
            "Feature selection dataset configuration is missing: " + ", ".join(sorted(missing))
        )
    if dataset["kind"] != "frozen_csv":
        raise ValueError("Unsupported feature selection dataset kind")
    path = dataset["path"]
    if not isinstance(path, str) or not path.strip():
        raise ValueError("path must be a non-empty dataset path")
    _validate_fraction(dataset["outer_test_size"], "outer_test_size")
    if not _is_integer(dataset["outer_seed"]):
        raise ValueError("outer_seed must be an integer")


def _validate_selector_config(
    document: dict[str, Any], expected_family_map: dict[str, list[str]]
) -> None:
    selector = document.get("selector")
    if not isinstance(selector, dict):
        raise ValueError("Feature selector configuration is missing")
    stage2_counts = selector.get("stage2_feature_counts")
    expected_stage2 = {
        family for family, members in expected_family_map.items() if len(members) > 1
    }
    if not isinstance(stage2_counts, dict) or set(stage2_counts) != expected_stage2:
        raise ValueError("Stage 2 feature budgets do not match training families")
    _validate_positive_integer(selector.get("stage1_feature_count"), "stage1_feature_count")
    for count in stage2_counts.values():
        _validate_positive_integer(count, "stage2_feature_counts")
    _validate_positive_integer(selector.get("estimators"), "estimators")
    if selector.get("stage1_missing_policy") != "allow_nan_and_report":
        raise ValueError("Unsupported Stage 1 missing-value policy")
    if selector.get("stage2_missing_policy") != "complete_within_family":
        raise ValueError("Unsupported Stage 2 missing-value policy")


def _validate_split_config(document: dict[str, Any]) -> None:
    split = document.get("split")
    if not isinstance(split, dict):
        raise ValueError("Feature selection split configuration is missing")
    test_size = split.get("test_size")
    random_state = split.get("random_state")
    _validate_fraction(test_size, "Feature selection test_size")
    if not _is_integer(random_state):
        raise ValueError("Feature selection random_state must be an integer")


def _validate_selections(
    document: dict[str, Any],
    expected_family_map: dict[str, list[str]],
    expected_feature_schema: list[str],
) -> None:
    selection = document.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("Feature selection payload is missing")
    known_features = set(expected_feature_schema)
    stage1_features = _validate_feature_list(
        selection.get("stage1_features"),
        label="Stage 1 selection",
        known_features=known_features,
    )
    selector = document["selector"]
    if len(stage1_features) != selector["stage1_feature_count"]:
        raise ValueError("Stage 1 selection does not match its feature count")
    stage2 = selection.get("stage2_features")
    if not isinstance(stage2, dict):
        raise ValueError("Stage 2 selection must be an object")
    expected_stage2 = {
        family for family, members in expected_family_map.items() if len(members) > 1
    }
    if set(stage2) != expected_stage2:
        raise ValueError("Stage 2 selection families do not match multi-distribution families")
    for family, features in stage2.items():
        validated_features = _validate_feature_list(
            features,
            label=f"Stage 2 selection for {family}",
            known_features=known_features,
        )
        if len(validated_features) != selector["stage2_feature_counts"][family]:
            raise ValueError(f"Stage 2 selection for {family} does not match its feature count")


def validate_selection_document(
    document: object,
    expected_family_map: dict[str, list[str]],
    expected_feature_schema: list[str],
) -> None:
    """Reject stale or malformed feature-selection artifacts at the boundary."""
    if not isinstance(document, dict):
        raise ValueError("Feature selection document must be a JSON object")
    if document.get("schema_version") != SELECTION_SCHEMA_VERSION:
        raise ValueError("Unsupported feature selection schema version")
    if document.get("family_map") != expected_family_map:
        raise ValueError("Feature selection family map does not match training families")
    if document.get("feature_schema") != expected_feature_schema:
        raise ValueError("Feature selection schema does not match the active feature schema")

    _validate_dataset_config(document)
    _validate_selector_config(document, expected_family_map)
    _validate_split_config(document)
    _validate_selections(document, expected_family_map, expected_feature_schema)

    dataset = document["dataset"]
    split = document["split"]
    if split["test_size"] != dataset["outer_test_size"]:
        raise ValueError("split.test_size must match dataset.outer_test_size")
    if split["random_state"] != dataset["outer_seed"]:
        raise ValueError("split.random_state must match dataset.outer_seed")
