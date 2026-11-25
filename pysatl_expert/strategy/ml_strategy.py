import logging
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path

import numpy as np
from joblib import load as load_model

from pysatl_expert.models.feature_vector import FeatureVector
from pysatl_expert.models.hierarchical_model import HierarchicalExpertModel
from pysatl_expert.models.model_manifest import validate_loaded_model, verify_model_manifest


logger = logging.getLogger(__name__)


class MLStrategy:
    """Classify raw GoF feature vectors using a pre-trained Random Forest."""

    required_statistics: Sequence[str] = ()

    def __init__(self, model_path: str | Path):
        """Load a trusted model and validate its feature schema."""
        p_model = Path(model_path)
        if not p_model.exists():
            raise FileNotFoundError(f"Model file not found: {p_model}")

        manifest = verify_model_manifest(p_model)
        self._artifact_identity = {
            "sha256": manifest["model"]["sha256"],
            "identifier": manifest["model"]["filename"],
            "feature_schema": {
                "sha256": manifest["feature_schema"]["sha256"],
                "names": list(manifest["feature_schema"]["names"]),
            },
        }
        self._feature_names = list(manifest["feature_schema"]["names"])

        try:
            self.model = load_model(p_model)
            validate_loaded_model(self.model, manifest)
            logger.info("Loaded Random Forest model from %s", model_path)
        except Exception as exc:
            logger.error("Failed to load model: %s", exc)
            raise

        self._class_names = sorted(self.model.classes_.tolist())
        logger.info("Model classes: %s", self._class_names)

    @property
    def feature_names(self) -> list[str]:
        """Return the exact ordered schema stored with the loaded model."""
        return list(self._feature_names)

    @property
    def required_features(self) -> frozenset[str]:
        """Return the full-schema columns actually consumed by the loaded model."""
        if not isinstance(self.model, HierarchicalExpertModel):
            return frozenset(self.feature_names)

        selected = set(self.model.stage1_features or [])
        for features in self.model.stage2_features.values():
            selected.update(features)

        unknown = selected.difference(self.feature_names)
        if unknown:
            raise ValueError(
                "Model selects features outside its bundled schema: " + ", ".join(sorted(unknown))
            )
        return frozenset(selected)

    def _infer(self, raw_vector: np.ndarray) -> tuple[np.ndarray, dict[str, float], dict]:
        """Return canonical probabilities and evidence from one model forward pass."""
        row = raw_vector.reshape(1, -1)
        if isinstance(self.model, HierarchicalExpertModel):
            probability_matrix, evidence_rows = self.model._forward_with_evidence(row)
            probabilities = probability_matrix[0]
            if evidence_rows is None:
                raise ValueError("Model forward pass did not return requested evidence")
            evidence = evidence_rows[0]
        else:
            probabilities = self.model.predict_proba(row)[0]
            evidence = {}
        scores = {
            name: float(probability)
            for name, probability in zip(self._class_names, probabilities, strict=True)
        }
        return probabilities, scores, evidence

    def predict(self, sample: np.ndarray, analysis) -> dict:
        """Return the canonical label and class scores for a catalog analysis."""
        if not analysis.statistics:
            raise ValueError("Catalog analysis has no statistic results for RF input")
        vector = FeatureVector(analysis.sample_stats, analysis.statistics)
        raw_vector = np.asarray(
            vector.as_flat_list(feature_names=self.feature_names), dtype=np.float64
        )
        probabilities, scores, evidence = self._infer(raw_vector)
        winner = self._class_names[int(np.argmax(probabilities))]
        artifact = deepcopy(getattr(self, "_artifact_identity", {}))
        return {
            "label": winner,
            "scores": scores,
            "evidence": evidence,
            "artifact": artifact,
        }
