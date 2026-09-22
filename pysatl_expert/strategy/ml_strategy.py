import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load as load_model

from pysatl_expert.core.strategy import AbstractStrategy
from pysatl_expert.models.feature_vector import FeatureVector
from pysatl_expert.models.hierarchical_model import HierarchicalExpertModel
from pysatl_expert.models.model_manifest import (
    validate_loaded_model,
    verify_model_manifest,
)
from pysatl_expert.models.report import Report


logger = logging.getLogger(__name__)


class MLStrategy(AbstractStrategy):
    """Classify raw GoF feature vectors using a pre-trained Random Forest."""

    def __init__(self, model_path: str | Path):
        """Load a trusted model and validate its feature schema."""
        p_model = Path(model_path)
        if not p_model.exists():
            raise FileNotFoundError(f"Model file not found: {p_model}")

        manifest = verify_model_manifest(p_model)
        self._feature_names = list(manifest["feature_schema"]["names"])

        try:
            self.model = load_model(p_model)
            validate_loaded_model(self.model, manifest)
            logger.info("Loaded Random Forest model from %s", model_path)
        except Exception as exc:
            logger.error("Failed to load model: %s", exc)
            raise

        model_feature_names = getattr(self.model, "feature_names", None)
        if model_feature_names != self._feature_names:
            actual_count = len(model_feature_names) if model_feature_names is not None else 0
            raise ValueError(
                "Model feature schema does not match its manifest: "
                f"expected {len(self._feature_names)}, got {actual_count}"
            )

        self._class_names = sorted(self.model.classes_.tolist())
        logger.info("Model classes: %s", self._class_names)

    @property
    def feature_names(self) -> list[str]:
        """Return the exact ordered schema stored with the loaded model."""
        feature_names = getattr(self, "_feature_names", None)
        if feature_names is None:
            feature_names = getattr(self.model, "feature_names", FeatureVector.FEATURE_NAMES)
        return list(feature_names)

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
                "Model selects features outside its bundled schema: "
                + ", ".join(sorted(unknown))
            )
        return frozenset(selected)

    def _hierarchical_evidence(self, X: np.ndarray) -> dict:
        """Collect actual base-sample forest scores and selected input values."""
        if not isinstance(self.model, HierarchicalExpertModel):
            return {}
        model = self.model
        frame = pd.DataFrame(X, columns=model.feature_names)
        stage1_frame = frame[model.stage1_features]
        stage1_scores = dict(
            zip(
                model.stage1_model.classes_,
                map(float, model.stage1_model.predict_proba(stage1_frame)[0]),
                strict=True,
            )
        )
        stage2_scores, stage2_features = {}, {}
        for family, members in model.family_map.items():
            features = model.stage2_features.get(family, [])
            stage2_features[family] = frame[features].iloc[0].to_dict()
            if family in model.stage2_models:
                forest = model.stage2_models[family]
                stage2_scores[family] = dict(
                    zip(
                        forest.classes_,
                        map(float, forest.predict_proba(frame[features])[0]),
                        strict=True,
                    )
                )
            else:
                stage2_scores[family] = {members[0]: 1.0}
        return dict(
            stage1_scores=stage1_scores,
            stage2_scores=stage2_scores,
            stage1_features=stage1_frame.iloc[0].to_dict(),
            stage2_features=stage2_features,
        )

    def predict_report(
        self, base_fv: FeatureVector, bootstrap_fvs: list[FeatureVector] | None = None
    ) -> Report:
        """Generate a recommendation report from raw statistics and optional resamples."""
        raw_vector = np.asarray(
            base_fv.as_flat_list(feature_names=self.feature_names), dtype=np.float64
        )
        X = raw_vector.reshape(1, -1)

        probabilities = self.model.predict_proba(X)[0]

        winner_idx = np.argmax(probabilities)
        winner = self._class_names[winner_idx]
        base_confidence = float(probabilities[winner_idx])

        final_ranks = {
            name: float(prob) for name, prob in zip(self._class_names, probabilities, strict=True)
        }
        confidence_kind = "model_probability"
        model_ranks = dict(final_ranks)
        evidence = self._hierarchical_evidence(X)
        class_name_by_key = {name.lower(): name for name in self._class_names}
        all_scores: dict[str, dict[str, float]] = {name: {} for name in self._class_names}
        for vector_idx, feature_name in enumerate(self.feature_names):
            if "__" not in feature_name:
                continue
            dist_name, crit_code = feature_name.split("__", maxsplit=1)
            class_name = class_name_by_key.get(dist_name)
            if class_name is None:
                continue
            all_scores[class_name][crit_code] = float(raw_vector[vector_idx])

        bootstrap_stability = None
        bootstrap_ranks = {}
        bootstrap_successful = 0
        if bootstrap_fvs:
            votes = []
            for fv in bootstrap_fvs:
                boot_vector = np.asarray(
                    fv.as_flat_list(feature_names=self.feature_names), dtype=np.float64
                )
                X_boot = boot_vector.reshape(1, -1)
                boot_pred = self.model.predict(X_boot)[0]
                if boot_pred not in self._class_names:
                    raise ValueError(f"Bootstrap predicted an unknown class: {boot_pred!r}")
                votes.append(boot_pred)

            vote_counts = {name: votes.count(name) for name in self._class_names}
            bootstrap_ranks = {name: count / len(votes) for name, count in vote_counts.items()}
            bootstrap_stability = bootstrap_ranks[winner]
            bootstrap_successful = len(votes)

        return Report(
            distribution_name=winner,
            confidence=round(base_confidence, 3),
            all_scores=all_scores,
            final_ranks=final_ranks,
            model_ranks=model_ranks,
            **evidence,
            confidence_kind=confidence_kind,
            model_confidence=round(base_confidence, 3),
            bootstrap_ranks=bootstrap_ranks,
            bootstrap_successful=bootstrap_successful,
            bootstrap_stability=(
                round(bootstrap_stability, 3) if bootstrap_stability is not None else None
            ),
            sample_statistics=base_fv.descriptive_stats,
        )
