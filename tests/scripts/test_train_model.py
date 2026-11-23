import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pysatl_expert.models.feature_selection_contract import build_selection_document
from pysatl_expert.models.feature_vector import FeatureVector
from pysatl_expert.models.hierarchical_model import HierarchicalExpertModel
from scripts.train_model import (
    _resolve_path,
    build_parser,
    evaluate_hierarchical_stages,
    save_training_artifacts,
)


def test_resolve_path_accepts_json_string(tmp_path: Path):
    assert _resolve_path("dataset/training.csv", tmp_path) == (tmp_path / "dataset/training.csv")


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
    model.feature_names = list(FeatureVector.FEATURE_NAMES)
    model.stage1_features = ["normal__ks", "skew"]
    model.stage2_features = {
        "SymmetricUnbounded": ["normal__ks"],
        "PositiveUnbounded": ["gamma__ks"],
        "Bounded": ["beta__ks"],
    }
    selection = {
        "schema_version": 1,
        "family_map": family_map,
        "feature_schema": list(FeatureVector.FEATURE_NAMES),
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


def test_domain_family_map_has_nonempty_unique_members():
    path = (
        Path(__file__).parents[2] / "pysatl_expert" / "config" / "domain_distribution_families.json"
    )
    family_map = json.loads(path.read_text(encoding="utf-8"))
    members = [member for family in family_map.values() for member in family]

    assert family_map
    assert all(
        isinstance(name, str) and name and isinstance(distributions, list) and distributions
        for name, distributions in family_map.items()
    )
    assert all(isinstance(member, str) and member for member in members)
    assert len(members) == len(set(members))


class _FixedPredictions:
    def __init__(self, predictions):
        self.predictions = np.asarray(predictions)
        self.seen_features = None

    def predict(self, features):
        self.seen_features = features.copy()
        assert len(features) == len(self.predictions)
        return self.predictions


def test_training_reports_exact_stage_metrics_and_filters_each_family():
    family_map = {"negative": ["A", "B"], "positive": ["C", "D"]}
    model = HierarchicalExpertModel(family_map)
    model.feature_names = ["route", "within"]
    model.stage1_features = ["route"]
    model.stage2_features = {"negative": ["within"], "positive": ["route", "within"]}
    model.stage1_model = _FixedPredictions(
        [
            "negative",
            "positive",
            "positive",
            "negative",
            "negative",
            "positive",
            "positive",
            "negative",
        ]
    )
    model.stage2_models = {
        "negative": _FixedPredictions(["A", "A", "B", "B"]),
        "positive": _FixedPredictions(["C", "C", "D", "D"]),
    }
    features = pd.DataFrame({"route": range(8), "within": range(10, 18)})
    target = pd.Series(["A", "B", "C", "D", "A", "B", "C", "D"])

    metrics = evaluate_hierarchical_stages(model, features, target)

    assert metrics["stage1_family"]["accuracy"] == 0.5
    assert metrics["stage1_family"]["labels"] == ["negative", "positive"]
    assert metrics["stage1_family"]["confusion_matrix"] == [[2, 2], [2, 2]]
    assert metrics["stage2_by_family"]["negative"]["accuracy"] == 0.5
    assert metrics["stage2_by_family"]["negative"]["labels"] == ["A", "B"]
    assert metrics["stage2_by_family"]["negative"]["confusion_matrix"] == [[1, 1], [1, 1]]
    assert metrics["stage2_by_family"]["positive"]["accuracy"] == 0.5
    assert metrics["stage2_by_family"]["positive"]["labels"] == ["C", "D"]
    assert metrics["stage2_by_family"]["positive"]["confusion_matrix"] == [[1, 1], [1, 1]]
    assert model.stage1_model.seen_features.equals(features[["route"]])
    assert model.stage2_models["negative"].seen_features.equals(
        features.loc[[0, 1, 4, 5], ["within"]]
    )
    assert model.stage2_models["positive"].seen_features.equals(
        features.loc[[2, 3, 6, 7], ["route", "within"]]
    )


def test_training_cli_projects_tuple_schema_and_validates_selection_without_fitting(
    tmp_path, monkeypatch
):
    from scripts import train_model

    family_map = {"demo": ["A", "B"]}
    rows = []
    for split in ("train", "outer_test"):
        for target in ("A", "B"):
            rows.append(
                dict.fromkeys(FeatureVector.FEATURE_NAMES, 1.0) | {"Target": target, "Split": split}
            )
    dataset = tmp_path / "frozen.csv"
    pd.DataFrame(rows).to_csv(dataset, index=False)
    document = build_selection_document(
        family_map=family_map,
        feature_schema=list(FeatureVector.FEATURE_NAMES),
        stage1_features=["sample_size"],
        stage2_features={"demo": ["normal__ks"]},
        test_size=0.2,
        random_state=42,
        stage1_count=1,
        stage2_count={"demo": 1},
        estimators=1,
        dataset_config={
            "kind": "frozen_csv",
            "path": str(dataset),
            "outer_test_size": 0.2,
            "outer_seed": 42,
        },
        stage1_missing_policy="allow_nan_and_report",
        stage2_missing_policy="complete_within_family",
    )
    families, features = tmp_path / "families.json", tmp_path / "features.json"
    families.write_text(json.dumps(family_map))
    features.write_text(json.dumps(document))

    def stop_before_training(self, frame, targets, **kwargs):
        assert frame.columns.tolist() == list(FeatureVector.FEATURE_NAMES)
        assert targets.tolist() == ["A", "B"]
        assert kwargs["stage2_features"] == {"demo": ["normal__ks"]}
        raise RuntimeError("stopped before training")

    monkeypatch.setattr(HierarchicalExpertModel, "fit_selected", stop_before_training)
    with pytest.raises(RuntimeError, match="stopped before training"):
        train_model.main(
            [
                "--families",
                str(families),
                "--features",
                str(features),
                "--output",
                str(tmp_path / "unused.joblib"),
                "--metrics-output",
                str(tmp_path / "unused.json"),
            ]
        )
