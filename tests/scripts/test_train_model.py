import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pysatl_expert.models.feature_vector import FeatureVector
from pysatl_expert.models.hierarchical_model import HierarchicalExpertModel
from scripts.train_model import (
    _resolve_path,
    build_parser,
    evaluate_hierarchical_stages,
    load_training_dataset,
    prepare_training_data,
    save_training_artifacts,
)


def test_resolve_path_accepts_json_string(tmp_path: Path):
    assert _resolve_path("dataset/training.csv", tmp_path) == (
        tmp_path / "dataset/training.csv"
    )


def test_training_defaults_do_not_overwrite_model():
    args = build_parser().parse_args([])

    assert args.features == Path("artifacts/hrf/model/selected_features.json")
    assert args.output == Path("artifacts/hrf/model/rf_expert_model.joblib")
    assert args.metrics_output == Path("artifacts/hrf/model/model_metrics.json")
    assert args.estimators == 200
    assert args.workers == 2
    assert args.overwrite is False


def test_training_artifacts_include_verified_model_manifest(tmp_path):
    family_map = {
        "SymmetricUnbounded": ["Normal", "Student"],
        "PositiveUnbounded": ["Exponential", "Gamma", "LogNormal", "Weibull"],
        "Bounded": ["Beta", "Uniform"],
    }
    model = HierarchicalExpertModel(family_map)
    model.feature_names = FeatureVector.FEATURE_NAMES
    model.stage1_features = ["normal__ks", "skew"]
    model.stage2_features = {
        "SymmetricUnbounded": ["normal__ks"],
        "PositiveUnbounded": ["gamma__ks"],
        "Bounded": ["beta__ks"],
    }
    selection = {
        "schema_version": 1,
        "family_map": family_map,
        "feature_schema": FeatureVector.FEATURE_NAMES,
        "selector": {
            "stage1_feature_count": 2,
            "stage2_feature_counts": {
                "SymmetricUnbounded": 1,
                "PositiveUnbounded": 1,
                "Bounded": 1,
            },
            "stage1_missing_policy": "allow_nan_and_report",
            "stage2_missing_policy": "complete_within_family",
        },
        "selection": {
            "stage1_features": model.stage1_features,
            "stage2_features": model.stage2_features,
        },
    }
    model_path = tmp_path / "expert.joblib"
    metrics_path = tmp_path / "metrics.json"

    manifest = save_training_artifacts(
        model_path=model_path,
        metrics_path=metrics_path,
        model=model,
        metrics={"accuracy": 0.9},
        feature_selection=selection,
    )

    assert model_path.is_file()
    assert metrics_path.is_file()
    assert (tmp_path / "expert.manifest.json").is_file()
    assert manifest["model"]["filename"] == "expert.joblib"
    assert json.loads(metrics_path.read_text()) == {"accuracy": 0.9}


def test_domain_family_map_covers_every_training_target_once():
    path = (
        Path(__file__).parents[2]
        / "pysatl_expert"
        / "config"
        / "domain_distribution_families.json"
    )
    family_map = json.loads(path.read_text(encoding="utf-8"))
    members = [member for family in family_map.values() for member in family]

    assert set(family_map) == {"SymmetricUnbounded", "PositiveUnbounded", "Bounded"}
    assert sorted(members) == sorted(
        ["Normal", "Student", "Exponential", "Gamma", "LogNormal", "Weibull", "Beta", "Uniform"]
    )
    assert len(members) == len(set(members))


def test_training_data_preserves_continuous_values_and_exact_schema():
    row = dict.fromkeys(FeatureVector.FEATURE_NAMES, 0.0)
    row["sample_size"] = 100.0
    row["normal__ks"] = 1.0
    row["Target"] = "Normal"

    features, target = prepare_training_data(pd.DataFrame([row]))

    assert features.columns.tolist() == FeatureVector.FEATURE_NAMES
    assert features.loc[0, "sample_size"] == 100.0
    assert target.tolist() == ["Normal"]


def test_training_data_preserves_missing_values_as_nan() -> None:
    row = dict.fromkeys(FeatureVector.FEATURE_NAMES, 0.0)
    row["normal__ks"] = np.nan
    row["normal__ad"] = np.inf
    row["Target"] = "Normal"

    features, _ = prepare_training_data(pd.DataFrame([row]))

    assert np.isnan(features.loc[0, "normal__ks"])
    assert np.isnan(features.loc[0, "normal__ad"])
    assert not (features == -1.0).any().any()


def test_training_data_discards_known_historical_excluded_columns() -> None:
    row = dict.fromkeys(FeatureVector.FEATURE_NAMES, 0.0)
    row["beta__mode"] = 123.0
    row["normal__rj"] = 0.99
    row["Target"] = "Normal"

    features, _ = prepare_training_data(pd.DataFrame([row]))

    assert features.columns.tolist() == FeatureVector.FEATURE_NAMES
    assert "beta__mode" not in features
    assert "normal__rj" not in features


def test_training_data_rejects_unknown_extra_columns() -> None:
    row = dict.fromkeys(FeatureVector.FEATURE_NAMES, 0.0)
    row["unknown_feature"] = 1.0
    row["Target"] = "Normal"

    with pytest.raises(ValueError, match="unexpected features"):
        prepare_training_data(pd.DataFrame([row]))


def test_training_loader_selects_balanced_smoke_rows_without_historical_columns(
    tmp_path: Path,
) -> None:
    rows = []
    for target in ("A", "B"):
        for index in range(3):
            row = dict.fromkeys(FeatureVector.FEATURE_NAMES, float(index))
            row["beta__mode"] = 1000.0
            row["Target"] = target
            rows.append(row)
    path = tmp_path / "raw.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    loaded = load_training_dataset(
        path,
        samples_per_class=2,
        expected_targets={"A", "B"},
        chunksize=2,
    )

    assert loaded["Target"].value_counts().to_dict() == {"A": 2, "B": 2}
    assert loaded.columns.tolist() == [*FeatureVector.FEATURE_NAMES, "Target"]
    assert "beta__mode" not in loaded
    assert all(dtype == np.float32 for dtype in loaded[FeatureVector.FEATURE_NAMES].dtypes)


def test_training_reports_stage1_and_each_stage2_accuracy():
    rng = np.random.default_rng(7)
    labels = np.repeat(["A", "B", "C", "D"], 20)
    centers = {"A": -4.0, "B": -2.0, "C": 2.0, "D": 4.0}
    features = pd.DataFrame(
        {
            "shape": [centers[label] + rng.normal(0.0, 0.1) for label in labels],
            "aux": rng.normal(size=len(labels)),
        }
    )
    target = pd.Series(labels)
    family_map = {"negative": ["A", "B"], "positive": ["C", "D"]}
    model = HierarchicalExpertModel(family_map)
    model.fit(features, target, n_stage1=None, n_stage2=None)

    metrics = evaluate_hierarchical_stages(model, features, target)

    assert metrics["stage1_family"]["accuracy"] > 0.95
    assert set(metrics["stage2_by_family"]) == {"negative", "positive"}
    assert all(
        result["accuracy"] > 0.95
        for result in metrics["stage2_by_family"].values()
    )
