import json
from pathlib import Path

import pandas as pd
import pytest

from pysatl_expert.models.feature_vector import FeatureVector
from scripts.select_features import (
    build_missingness_audit,
    build_parser,
    build_selection_document,
    validate_run_config,
    validate_selection_document,
)


FAMILY_MAP = {"first": ["A", "B"], "second": ["C", "D"]}
FEATURE_SCHEMA = ["shape", "aux", "sample_size"]


def _document() -> dict[str, object]:
    return build_selection_document(
        family_map=FAMILY_MAP,
        feature_schema=FEATURE_SCHEMA,
        stage1_features=["shape", "sample_size"],
        stage2_features={"first": ["aux"], "second": ["shape"]},
        test_size=0.2,
        random_state=42,
        stage1_count=2,
        stage2_count={"first": 1, "second": 1},
        estimators=20,
        dataset_config={
            "kind": "frozen_csv",
            "path": "training.csv",
            "outer_test_size": 0.2,
            "outer_seed": 42,
        },
        stage1_missing_policy="allow_nan_and_report",
        stage2_missing_policy="complete_within_family",
    )


def test_feature_selection_defaults_are_non_destructive():
    args = build_parser().parse_args([])

    assert args.config == Path("pysatl_expert/config/feature_selection.json")
    assert args.output == Path("artifacts/hrf/model/selected_features.json")
    assert args.overwrite is False


def test_selection_document_records_exact_features_and_reproducible_split():
    document = _document()

    validate_selection_document(document, FAMILY_MAP, FEATURE_SCHEMA)

    assert document["selection"] == {
        "stage1_features": ["shape", "sample_size"],
        "stage2_features": {"first": ["aux"], "second": ["shape"]},
    }
    assert document["split"] == {"test_size": 0.2, "random_state": 42}


def test_selection_document_rejects_unknown_or_missing_features():
    document = _document()
    document["selection"]["stage1_features"] = ["unknown"]

    with pytest.raises(ValueError, match="unknown feature"):
        validate_selection_document(document, FAMILY_MAP, FEATURE_SCHEMA)


def test_selection_document_rejects_changed_family_map():
    document = json.loads(json.dumps(_document()))

    with pytest.raises(ValueError, match="family map"):
        validate_selection_document(
            document,
            {"combined": ["A", "B", "C", "D"]},
            FEATURE_SCHEMA,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("stage1_feature_count", 0),
        ("stage2_feature_counts", {"first": 0, "second": 1}),
        ("estimators", 0),
    ],
)
def test_selection_document_rejects_invalid_selector_numbers(field, value):
    document = _document()
    document["selector"][field] = value

    with pytest.raises(ValueError, match=field):
        validate_selection_document(document, FAMILY_MAP, FEATURE_SCHEMA)


def test_selection_document_rejects_feature_count_mismatch():
    document = _document()
    document["selection"]["stage1_features"] = ["shape"]

    with pytest.raises(ValueError, match="Stage 1 selection.*feature count"):
        validate_selection_document(document, FAMILY_MAP, FEATURE_SCHEMA)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("path", ""),
        ("path", None),
        ("outer_test_size", 0.0),
        ("outer_seed", []),
    ],
)
def test_selection_document_rejects_invalid_dataset_values(field, value):
    document = _document()
    document["dataset"][field] = value

    with pytest.raises(ValueError, match=field):
        validate_selection_document(document, FAMILY_MAP, FEATURE_SCHEMA)


def test_selection_document_rejects_inconsistent_outer_split():
    document = _document()
    document["split"]["random_state"] = 99

    with pytest.raises(ValueError, match="outer_seed"):
        validate_selection_document(document, FAMILY_MAP, FEATURE_SCHEMA)


def test_selection_document_rejects_duplicate_selected_features():
    document = _document()
    document["selection"]["stage2_features"]["first"] = ["aux", "aux"]

    with pytest.raises(ValueError, match="duplicate features"):
        validate_selection_document(document, FAMILY_MAP, FEATURE_SCHEMA)


def test_tracked_selection_config_satisfies_generic_schema():
    project_root = Path(__file__).parents[2]
    family_map = json.loads(
        (project_root / "pysatl_expert/config/domain_distribution_families.json").read_text()
    )
    config = json.loads((project_root / "pysatl_expert/config/feature_selection.json").read_text())

    validated = validate_run_config(config, family_map)

    assert validated["stage1_feature_count"] > 0
    assert set(validated["stage2_feature_counts"]) == set(family_map)
    assert all(count > 0 for count in validated["stage2_feature_counts"].values())
    assert validated["dataset"]["kind"] == "frozen_csv"
    assert validated["dataset"]["path"]
    assert 0 < validated["dataset"]["outer_test_size"] < 1
    assert validated["dataset"]["outer_seed"] is not None


def test_missingness_audit_selects_tuple_schema_columns_without_fitting():
    training = pd.DataFrame(1.0, index=range(4), columns=FeatureVector.FEATURE_NAMES)
    training.loc[0, "normal__ks"] = float("nan")
    targets = pd.Series(["A", "B", "C", "D"])

    audit = build_missingness_audit(
        training,
        targets,
        FAMILY_MAP,
        ["sample_size"],
        {"first": ["sample_size"], "second": ["normal__ks"]},
    )

    assert audit["stage1"]["eligible_feature_count"] == len(FeatureVector.TRAINING_FEATURE_NAMES)
    assert audit["stage2"]["first"]["excluded_missing_features"] == ["normal__ks"]
    assert audit["stage2"]["second"]["excluded_missing_features"] == []
