from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any


MANIFEST_FORMAT_VERSION = 1
_MAX_MANIFEST_BYTES = 1_000_000
_HASH_CHUNK_BYTES = 1024 * 1024
_RUNTIME_DISTRIBUTIONS = {
    "numpy": "numpy",
    "scipy": "scipy",
    "scikit_learn": "scikit-learn",
    "joblib": "joblib",
}
_PACKAGE_DISTRIBUTIONS = {
    "pysatl_expert": "pysatl-expert",
    "pysatl_criterion": "pysatl-criterion",
}


class ModelCompatibilityError(ValueError):
    """Raised when a model bundle is damaged, stale, or incompatible."""


def _prototype_constraints() -> dict[str, Any]:
    return {
        "positive_distributions_loc": 0.0,
        "beta_support": [0.0, 1.0],
        "validated_sample_size": {"min": 50, "max": 1000},
        "score_interpretation": "uncalibrated_recommendation",
    }


def manifest_path_for(model_path: str | Path) -> Path:
    """Return the mandatory manifest path adjacent to a joblib model."""
    return Path(model_path).with_suffix(".manifest.json")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _installed_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _runtime_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    versions.update(
        {
            key: _installed_version(distribution)
            for key, distribution in _RUNTIME_DISTRIBUTIONS.items()
        }
    )
    return versions


def _package_versions() -> dict[str, str]:
    return {
        key: _installed_version(distribution)
        for key, distribution in _PACKAGE_DISTRIBUTIONS.items()
    }


def _portable_feature_configuration(selection_document: dict[str, Any]) -> dict[str, Any]:
    """Keep only model-relevant fields; training paths are intentionally excluded."""
    required = ("schema_version", "family_map", "feature_schema", "selector", "selection")
    missing = [key for key in required if key not in selection_document]
    if missing:
        raise ModelCompatibilityError(
            "Feature selection document is missing: " + ", ".join(missing)
        )
    return {key: selection_document[key] for key in required}


def _selected_feature_summary(stage1: list[str], stage2: dict[str, list[str]]) -> dict[str, Any]:
    required = set(stage1)
    for features in stage2.values():
        required.update(features)
    gof = sorted(feature for feature in required if "__" in feature)
    descriptive = sorted(required.difference(gof))
    return {
        "stage1": stage1,
        "stage2": stage2,
        "required_total": len(required),
        "gof_statistics": len(gof),
        "descriptive": descriptive,
    }


def build_model_manifest(
    model_path: str | Path,
    model: object,
    selection_document: dict[str, Any],
) -> dict[str, Any]:
    """Build a portable manifest for an existing trusted model file."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")

    portable_config = _portable_feature_configuration(selection_document)
    feature_schema = list(portable_config["feature_schema"])
    family_map = portable_config["family_map"]
    selection = portable_config["selection"]
    stage1 = list(selection["stage1_features"])
    stage2 = {family: list(features) for family, features in selection["stage2_features"].items()}
    classes = sorted(member for members in family_map.values() for member in members)

    manifest = {
        "format_version": MANIFEST_FORMAT_VERSION,
        "model": {
            "filename": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
            "serialization": "joblib",
            "type": f"{type(model).__module__}.{type(model).__qualname__}",
        },
        "runtime": _runtime_versions(),
        "packages": _package_versions(),
        "classes": classes,
        "families": family_map,
        "feature_schema": {
            "size": len(feature_schema),
            "sha256": _canonical_sha256(feature_schema),
            "names": feature_schema,
        },
        "selected_features": _selected_feature_summary(stage1, stage2),
        "feature_configuration": {
            "schema_version": portable_config["schema_version"],
            "sha256": _canonical_sha256(portable_config),
            "selector": portable_config["selector"],
        },
        "constraints": _prototype_constraints(),
    }
    validate_loaded_model(model, manifest)
    return manifest


def write_model_manifest(
    model_path: str | Path,
    model: object,
    selection_document: dict[str, Any],
) -> dict[str, Any]:
    """Atomically write the manifest adjacent to a trusted model file."""
    path = manifest_path_for(model_path)
    manifest = build_model_manifest(model_path, model, selection_document)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary_path.open("w", encoding="utf-8") as target:
        json.dump(manifest, target, ensure_ascii=False, indent=2)
        target.write("\n")
    temporary_path.replace(path)
    return manifest


def _require_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ModelCompatibilityError(f"Model manifest {label} must be an object")
    return value


def _validate_runtime(manifest: dict[str, Any]) -> None:
    recorded = _require_mapping(manifest.get("runtime"), "runtime")
    current = _runtime_versions()
    recorded_python = str(recorded.get("python", ""))
    if recorded_python.split(".")[:2] != current["python"].split(".")[:2]:
        raise ModelCompatibilityError(
            "Model Python version is incompatible: "
            f"trained with {recorded_python}, running {current['python']}"
        )
    for key in _RUNTIME_DISTRIBUTIONS:
        if recorded.get(key) != current[key]:
            raise ModelCompatibilityError(
                f"Model runtime dependency {key} is incompatible: "
                f"trained with {recorded.get(key)!r}, running {current[key]!r}"
            )
    recorded_packages = _require_mapping(manifest.get("packages"), "packages")
    current_packages = _package_versions()
    for key in _PACKAGE_DISTRIBUTIONS:
        if recorded_packages.get(key) != current_packages[key]:
            raise ModelCompatibilityError(
                f"Model package {key} is incompatible: "
                f"trained with {recorded_packages.get(key)!r}, "
                f"running {current_packages[key]!r}"
            )


def _validate_selected_features(manifest: dict[str, Any], feature_names: list[str]) -> None:
    selected = _require_mapping(manifest.get("selected_features"), "selected_features")
    stage1 = selected.get("stage1")
    stage2 = selected.get("stage2")
    if (
        not isinstance(stage1, list)
        or not stage1
        or not all(isinstance(feature, str) for feature in stage1)
    ):
        raise ModelCompatibilityError("Model manifest Stage 1 features are invalid")
    if not isinstance(stage2, dict) or not stage2:
        raise ModelCompatibilityError("Model manifest Stage 2 features are invalid")
    if any(
        not isinstance(features, list)
        or not features
        or not all(isinstance(feature, str) for feature in features)
        for features in stage2.values()
    ):
        raise ModelCompatibilityError("Model manifest Stage 2 features are invalid")
    if len(stage1) != len(set(stage1)) or any(
        len(features) != len(set(features)) for features in stage2.values()
    ):
        raise ModelCompatibilityError("Model manifest selected features contain duplicates")
    unknown = set(stage1)
    for features in stage2.values():
        unknown.update(features)
    unknown.difference_update(feature_names)
    if unknown:
        raise ModelCompatibilityError(
            "Model manifest selects unknown features: " + ", ".join(sorted(unknown))
        )
    expected = _selected_feature_summary(stage1, stage2)
    if selected != expected:
        raise ModelCompatibilityError("Model manifest selected-feature summary is inconsistent")


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ModelCompatibilityError(
            f"Model manifest is missing: {path}. " "Install or download the complete model bundle."
        )
    if path.stat().st_size > _MAX_MANIFEST_BYTES:
        raise ModelCompatibilityError("Model manifest is unexpectedly large")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ModelCompatibilityError(f"Model manifest cannot be read: {exc}") from exc
    return _require_mapping(manifest, "root")


def _validate_model_file(path: Path, manifest: dict[str, Any]) -> None:
    model_info = _require_mapping(manifest.get("model"), "model")
    if model_info.get("filename") != path.name:
        raise ModelCompatibilityError("Model manifest filename does not match the model")
    if model_info.get("serialization") != "joblib":
        raise ModelCompatibilityError("Unsupported model serialization format")
    if model_info.get("size_bytes") != path.stat().st_size:
        raise ModelCompatibilityError("Model file size does not match the manifest")
    if model_info.get("sha256") != _sha256_file(path):
        raise ModelCompatibilityError("Model SHA-256 does not match the manifest")


def _validate_feature_schema(manifest: dict[str, Any]) -> list[str]:
    schema = _require_mapping(manifest.get("feature_schema"), "feature_schema")
    feature_names = schema.get("names")
    if (
        not isinstance(feature_names, list)
        or not feature_names
        or not all(isinstance(name, str) and name for name in feature_names)
        or len(feature_names) != len(set(feature_names))
    ):
        raise ModelCompatibilityError("Model feature schema is invalid")
    if schema.get("size") != len(feature_names):
        raise ModelCompatibilityError("Model feature schema size is inconsistent")
    if schema.get("sha256") != _canonical_sha256(feature_names):
        raise ModelCompatibilityError("Model feature schema hash is inconsistent")
    return feature_names


def _validate_feature_configuration(
    manifest: dict[str, Any], expected_feature_schema: list[str]
) -> None:
    families = _require_mapping(manifest.get("families"), "families")
    if len(families) != 3 or any(
        not isinstance(members, list)
        or not members
        or not all(isinstance(member, str) for member in members)
        for members in families.values()
    ):
        raise ModelCompatibilityError("Model manifest must contain three valid families")
    classes = manifest.get("classes")
    expected_classes = sorted(member for members in families.values() for member in members)
    if (
        classes != expected_classes
        or len(expected_classes) != 8
        or len(classes) != len(set(classes))
    ):
        raise ModelCompatibilityError("Model manifest classes and families are inconsistent")
    _validate_selected_features(manifest, expected_feature_schema)
    if manifest.get("constraints") != _prototype_constraints():
        raise ModelCompatibilityError("Model prototype constraints are incompatible")

    selected = manifest["selected_features"]
    feature_configuration = _require_mapping(
        manifest.get("feature_configuration"), "feature_configuration"
    )
    portable_configuration = {
        "schema_version": feature_configuration.get("schema_version"),
        "family_map": families,
        "feature_schema": expected_feature_schema,
        "selector": feature_configuration.get("selector"),
        "selection": {
            "stage1_features": selected["stage1"],
            "stage2_features": selected["stage2"],
        },
    }
    if feature_configuration.get("sha256") != _canonical_sha256(portable_configuration):
        raise ModelCompatibilityError("Model feature configuration hash is inconsistent")


def verify_model_manifest(model_path: str | Path) -> dict[str, Any]:
    """Verify bundle integrity and internal compatibility before deserialization."""
    path = Path(model_path)
    manifest_path = manifest_path_for(path)
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")
    manifest = _read_manifest(manifest_path)
    if manifest.get("format_version") != MANIFEST_FORMAT_VERSION:
        raise ModelCompatibilityError("Unsupported model manifest format version")
    _validate_model_file(path, manifest)
    feature_schema = _validate_feature_schema(manifest)
    _validate_feature_configuration(manifest, feature_schema)
    _validate_runtime(manifest)
    return manifest


def validate_loaded_model(model: object, manifest: dict[str, Any]) -> None:
    """Validate deserialized model internals against the already verified manifest."""
    expected_type = manifest["model"]["type"]
    actual_type = f"{type(model).__module__}.{type(model).__qualname__}"
    if actual_type != expected_type:
        raise ModelCompatibilityError(
            f"Loaded model type does not match the manifest: {actual_type}"
        )
    if getattr(model, "feature_names", None) != manifest["feature_schema"]["names"]:
        raise ModelCompatibilityError("Loaded model feature schema does not match the manifest")
    if getattr(model, "family_map", None) != manifest["families"]:
        raise ModelCompatibilityError("Loaded model families do not match the manifest")
    classes = sorted(str(value) for value in getattr(model, "classes_", []))
    if classes != manifest["classes"]:
        raise ModelCompatibilityError("Loaded model classes do not match the manifest")
    selected = manifest["selected_features"]
    if getattr(model, "stage1_features", None) != selected["stage1"]:
        raise ModelCompatibilityError("Loaded model Stage 1 features do not match the manifest")
    if getattr(model, "stage2_features", None) != selected["stage2"]:
        raise ModelCompatibilityError("Loaded model Stage 2 features do not match the manifest")
