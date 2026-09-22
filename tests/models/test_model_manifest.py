import json
from pathlib import Path

import numpy as np
import pytest

from pysatl_expert.models.feature_vector import FeatureVector
from pysatl_expert.models.model_manifest import (
    ModelCompatibilityError,
    build_model_manifest,
    manifest_path_for,
    validate_loaded_model,
    verify_model_manifest,
    write_model_manifest,
)


FAMILY_MAP = {
    "SymmetricUnbounded": ["Normal", "Student"],
    "PositiveUnbounded": ["Exponential", "Gamma", "LogNormal", "Weibull"],
    "Bounded": ["Beta", "Uniform"],
}


class CompatibleModel:
    feature_names = FeatureVector.FEATURE_NAMES
    family_map = FAMILY_MAP
    stage1_features = ["normal__ks", "skew"]
    stage2_features = {
        "SymmetricUnbounded": ["normal__ks"],
        "PositiveUnbounded": ["gamma__ks"],
        "Bounded": ["beta__ks"],
    }
    classes_ = np.array(sorted(member for members in FAMILY_MAP.values() for member in members))


def _selection_document() -> dict:
    return {
        "schema_version": 1,
        "family_map": FAMILY_MAP,
        "feature_schema": FeatureVector.FEATURE_NAMES,
        "dataset": {
            "base_dataset": "/private/training/base.csv",
            "uniform_dataset": "/private/training/uniform.csv",
        },
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
            "stage1_features": CompatibleModel.stage1_features,
            "stage2_features": CompatibleModel.stage2_features,
        },
    }


def test_manifest_records_portable_model_contract(tmp_path: Path) -> None:
    model_path = tmp_path / "expert.joblib"
    model_path.write_bytes(b"trusted model bytes")

    manifest = build_model_manifest(model_path, CompatibleModel(), _selection_document())

    assert manifest["format_version"] == 1
    assert manifest["model"]["filename"] == "expert.joblib"
    assert manifest["model"]["size_bytes"] == len(b"trusted model bytes")
    assert len(manifest["model"]["sha256"]) == 64
    assert manifest["classes"] == sorted(CompatibleModel.classes_.tolist())
    assert manifest["families"] == FAMILY_MAP
    assert manifest["feature_schema"]["size"] == len(FeatureVector.FEATURE_NAMES)
    assert manifest["feature_schema"]["names"] == FeatureVector.FEATURE_NAMES
    assert manifest["selected_features"]["required_total"] == 4
    assert manifest["selected_features"]["gof_statistics"] == 3
    assert manifest["selected_features"]["descriptive"] == ["skew"]
    assert manifest["constraints"]["positive_distributions_loc"] == 0.0
    assert manifest["constraints"]["beta_support"] == [0.0, 1.0]
    assert "/private/" not in json.dumps(manifest)


def test_written_manifest_can_be_verified_without_loading_model(tmp_path: Path) -> None:
    model_path = tmp_path / "expert.joblib"
    model_path.write_bytes(b"trusted model bytes")
    expected = write_model_manifest(model_path, CompatibleModel(), _selection_document())

    actual = verify_model_manifest(model_path)

    assert actual == expected
    assert manifest_path_for(model_path) == tmp_path / "expert.manifest.json"


def test_manifest_verification_rejects_model_tampering(tmp_path: Path) -> None:
    model_path = tmp_path / "expert.joblib"
    model_path.write_bytes(b"original")
    write_model_manifest(model_path, CompatibleModel(), _selection_document())
    model_path.write_bytes(b"tampered")

    with pytest.raises(ModelCompatibilityError, match="SHA-256"):
        verify_model_manifest(model_path)


def test_manifest_verification_uses_its_own_feature_schema(tmp_path: Path) -> None:
    model_path = tmp_path / "expert.joblib"
    model_path.write_bytes(b"trusted model bytes")
    expected = write_model_manifest(model_path, CompatibleModel(), _selection_document())

    assert verify_model_manifest(model_path)["feature_schema"] == expected["feature_schema"]


def test_manifest_verification_rejects_internally_changed_feature_schema(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "expert.joblib"
    model_path.write_bytes(b"trusted model bytes")
    write_model_manifest(model_path, CompatibleModel(), _selection_document())
    manifest_path = manifest_path_for(model_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["feature_schema"]["names"].append("new_upstream_criterion")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ModelCompatibilityError, match="feature schema"):
        verify_model_manifest(model_path)


def test_manifest_verification_rejects_changed_feature_configuration(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "expert.joblib"
    model_path.write_bytes(b"trusted model bytes")
    write_model_manifest(model_path, CompatibleModel(), _selection_document())
    manifest_path = manifest_path_for(model_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["feature_configuration"]["selector"]["stage1_feature_count"] = 99
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ModelCompatibilityError, match="feature configuration hash"):
        verify_model_manifest(model_path)


def test_manifest_verification_rejects_changed_prototype_constraints(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "expert.joblib"
    model_path.write_bytes(b"trusted model bytes")
    write_model_manifest(model_path, CompatibleModel(), _selection_document())
    manifest_path = manifest_path_for(model_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["constraints"]["positive_distributions_loc"] = 10.0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ModelCompatibilityError, match="prototype constraints"):
        verify_model_manifest(model_path)


def test_loaded_model_must_match_verified_manifest(tmp_path: Path) -> None:
    model_path = tmp_path / "expert.joblib"
    model_path.write_bytes(b"trusted model bytes")
    manifest = build_model_manifest(model_path, CompatibleModel(), _selection_document())
    stale_model = CompatibleModel()
    stale_model.stage1_features = ["student__ks"]

    with pytest.raises(ModelCompatibilityError, match="Stage 1"):
        validate_loaded_model(stale_model, manifest)
