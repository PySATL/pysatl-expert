import gc
from collections.abc import Mapping

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import check_random_state

from pysatl_expert.models.feature_vector import FeatureVector


_SELECTION_FOREST_BATCH_SIZE = 10


class HierarchicalExpertModel:
    """Two-stage Random Forest classifier for distribution identification.

    Stage 1 predicts a distribution family. A family-specific Stage 2 model
    then predicts a distribution within that family. Final probabilities are
    products of family and conditional distribution probabilities.

    Feature selection can be separated from final fitting so training reuses
    an explicit, reproducible feature schema.
    """

    def __init__(self, family_map: dict[str, list[str]]):
        """Initialize the hierarchical model with distribution family mappings.

        Args:
            family_map (dict[str, list[str]]): Mapping of family names to member distributions.
        """
        if not family_map or any(not members for members in family_map.values()):
            raise ValueError("Family map must contain non-empty families")

        self.family_map = family_map
        self.dist_to_family: dict[str, str] = {}
        for fam_name, dists in family_map.items():
            for d in dists:
                if d in self.dist_to_family:
                    raise ValueError(f"Distribution {d!r} belongs to more than one family")
                self.dist_to_family[d] = fam_name

        self.feature_names: list[str] | None = None
        self.stage1_model: RandomForestClassifier | None = None
        self.stage1_features: list[str] | None = None

        self.stage2_models: dict[str, RandomForestClassifier] = {}
        self.stage2_features: dict[str, list[str]] = {}

    def _validate_training_data(
        self, X_df: pd.DataFrame, y_series: pd.Series
    ) -> list[str]:
        if len(X_df) != len(y_series) or len(X_df) == 0:
            raise ValueError("Training features and targets must be non-empty and equally sized")
        expected_targets = set(self.dist_to_family)
        actual_targets = set(y_series.unique())
        if actual_targets != expected_targets:
            raise ValueError(
                "Training targets and family map does not match: "
                f"targets={sorted(actual_targets)}, family_map={sorted(expected_targets)}"
            )
        training_features = [
            feature
            for feature in X_df.columns
            if feature not in FeatureVector.EXCLUDED_TRAINING_FEATURES
        ]
        if not training_features:
            raise ValueError("No numerically stable features available for training")
        return training_features

    @staticmethod
    def _validate_forest_resources(n_estimators: int, n_jobs: int) -> None:
        if n_estimators < 1:
            raise ValueError("n_estimators must be positive")
        if n_jobs < 1:
            raise ValueError("n_jobs must be positive")

    def _validate_stage2_budgets(
        self, n_stage2: int | Mapping[str, int] | None
    ) -> None:
        if isinstance(n_stage2, Mapping):
            expected_families = {
                family for family, members in self.family_map.items() if len(members) > 1
            }
            if set(n_stage2) != expected_families:
                raise ValueError("n_stage2 mapping must cover every multi-distribution family")
            if any(count < 1 for count in n_stage2.values()):
                raise ValueError("n_stage2 family budgets must be positive")
        elif n_stage2 is not None and n_stage2 < 1:
            raise ValueError("n_stage2 must be positive or None")

    @staticmethod
    def _batched_feature_importances(
        features: np.ndarray,
        target: pd.Series | np.ndarray,
        *,
        n_estimators: int,
        n_jobs: int,
        batch_size: int,
    ) -> np.ndarray:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        random_state = check_random_state(42)
        tree_importances: list[np.ndarray] = []
        for batch_start in range(0, n_estimators, batch_size):
            selector = RandomForestClassifier(
                n_estimators=min(batch_size, n_estimators - batch_start),
                max_depth=15,
                bootstrap=True,
                random_state=random_state,
                n_jobs=n_jobs,
            )
            selector.fit(features, target)
            tree_importances.extend(
                tree.feature_importances_ for tree in selector.estimators_
            )
            del selector
            gc.collect()
        return np.mean(tree_importances, axis=0)

    @staticmethod
    def _select_top_features(
        features: pd.DataFrame,
        target: pd.Series,
        candidates: list[str],
        count: int | None,
        n_estimators: int,
        n_jobs: int,
    ) -> list[str]:
        if count is not None and len(candidates) < count:
            raise ValueError(f"Insufficient eligible features: {len(candidates)} < {count}")
        if count is None or count >= len(candidates):
            return candidates.copy()
        candidate_frame = features.loc[:, candidates]
        candidate_matrix = candidate_frame.to_numpy(copy=False)
        del candidate_frame
        feature_importances = HierarchicalExpertModel._batched_feature_importances(
            candidate_matrix,
            target,
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            batch_size=_SELECTION_FOREST_BATCH_SIZE,
        )
        del candidate_matrix
        importances = pd.Series(feature_importances, index=candidates)
        return importances.nlargest(count).index.tolist()

    def select_features(
        self,
        X_df: pd.DataFrame,
        y_series: pd.Series,
        n_stage1: int | None = 30,
        n_stage2: int | Mapping[str, int] | None = 20,
        complete_stage2: bool = False,
        n_estimators: int = 200,
        n_jobs: int = 2,
    ) -> tuple[list[str], dict[str, list[str]]]:
        """Train temporary forests and return concrete features for both stages."""
        training_features = self._validate_training_data(X_df, y_series)
        self._validate_forest_resources(n_estimators, n_jobs)
        if n_stage1 is not None and n_stage1 < 1:
            raise ValueError("n_stage1 must be positive or None")
        self._validate_stage2_budgets(n_stage2)

        y_families = y_series.map(self.dist_to_family)
        stage1_features = self._select_top_features(
            X_df,
            y_families,
            training_features,
            n_stage1,
            n_estimators,
            n_jobs,
        )

        stage2_features: dict[str, list[str]] = {}
        for family_name, distribution_members in self.family_map.items():
            if len(distribution_members) <= 1:
                continue
            family_mask = y_series.isin(distribution_members)
            X_family = X_df.loc[family_mask, training_features]
            y_family = y_series[family_mask]
            eligible_features = training_features
            if complete_stage2:
                eligible_features = [
                    feature
                    for feature in training_features
                    if not X_family[feature].isna().any()
                ]
            family_budget = (
                n_stage2[family_name] if isinstance(n_stage2, Mapping) else n_stage2
            )
            stage2_features[family_name] = self._select_top_features(
                X_family,
                y_family,
                eligible_features,
                family_budget,
                n_estimators,
                n_jobs,
            )
        return stage1_features, stage2_features

    def fit_selected(
        self,
        X_df: pd.DataFrame,
        y_series: pd.Series,
        *,
        stage1_features: list[str],
        stage2_features: dict[str, list[str]],
        n_estimators: int = 200,
        n_jobs: int = 2,
    ) -> None:
        """Train final forests on an explicit, previously selected feature set."""
        training_features = self._validate_training_data(X_df, y_series)
        self._validate_forest_resources(n_estimators, n_jobs)
        expected_stage2 = {
            family_name
            for family_name, members in self.family_map.items()
            if len(members) > 1
        }
        missing_stage2 = expected_stage2.difference(stage2_features)
        extra_stage2 = set(stage2_features).difference(expected_stage2)
        if missing_stage2:
            raise ValueError(
                "Feature selection is missing Stage 2 selections for: "
                + ", ".join(sorted(missing_stage2))
            )
        if extra_stage2:
            raise ValueError(
                "Feature selection contains unknown Stage 2 families: "
                + ", ".join(sorted(extra_stage2))
            )

        selections = {"Stage 1": stage1_features, **stage2_features}
        for selection_name, selected in selections.items():
            if not selected:
                raise ValueError(f"{selection_name} feature selection must not be empty")
            if len(selected) != len(set(selected)):
                raise ValueError(f"{selection_name} feature selection contains duplicates")
            unknown = set(selected).difference(training_features)
            if unknown:
                raise ValueError(
                    f"{selection_name} contains unknown feature(s): "
                    + ", ".join(sorted(unknown))
                )

        self.feature_names = X_df.columns.tolist()
        self.stage1_features = list(stage1_features)
        self.stage2_features = {
            family_name: list(selected)
            for family_name, selected in stage2_features.items()
        }
        self.stage2_models = {}
        y_families = y_series.map(self.dist_to_family)
        self.stage1_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=12,
            bootstrap=True,
            random_state=42,
            n_jobs=n_jobs,
        )
        self.stage1_model.fit(X_df[self.stage1_features], y_families)

        for family_name, distribution_members in self.family_map.items():
            if len(distribution_members) <= 1:
                continue
            family_mask = y_series.isin(distribution_members)
            family_model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=12,
                bootstrap=True,
                random_state=42,
                n_jobs=n_jobs,
            )
            selected = self.stage2_features[family_name]
            family_model.fit(X_df.loc[family_mask, selected], y_series[family_mask])
            self.stage2_models[family_name] = family_model

    @property
    def classes_(self) -> np.ndarray:
        """Return sorted array of target distribution class names."""
        return np.array(sorted(list(self.dist_to_family.keys())))

    def fit(
        self,
        X_df: pd.DataFrame,
        y_series: pd.Series,
        n_stage1: int | None = 30,
        n_stage2: int | None = 20,
        n_estimators: int = 200,
        n_jobs: int = 2,
    ):
        """Train Stage 1 family model and Stage 2 sub-family models.

        Args:
            X_df (pd.DataFrame): Feature DataFrame.
            y_series (pd.Series): Target distribution class names Series.
            n_stage1 (int | None): Number of top features for Stage 1 model (None for all).
            n_stage2 (int | None): Number of top features for Stage 2 sub-models (None for all).
            n_estimators (int): Number of trees in every selection and final forest.
            n_jobs (int): Number of parallel workers used by every forest.
        """
        stage1_features, stage2_features = self.select_features(
            X_df,
            y_series,
            n_stage1=n_stage1,
            n_stage2=n_stage2,
            n_estimators=n_estimators,
            n_jobs=n_jobs,
        )
        self.fit_selected(
            X_df,
            y_series,
            stage1_features=stage1_features,
            stage2_features=stage2_features,
            n_estimators=n_estimators,
            n_jobs=n_jobs,
        )

    def predict_proba(self, X_df: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Vectorized batch predict probability matrix across all distribution classes.

        Args:
            X_df (pd.DataFrame | np.ndarray): Input feature matrix.

        Returns:
            np.ndarray: Predicted probability matrix of shape (n_samples, n_classes).
        """
        if isinstance(X_df, np.ndarray):
            if self.feature_names is None:
                raise RuntimeError("Model must be fitted before calling predict_proba")
            if X_df.ndim != 2 or X_df.shape[1] != len(self.feature_names):
                actual_features = X_df.shape[1] if X_df.ndim == 2 else "non-2D"
                raise ValueError(
                    f"Model expected {len(self.feature_names)} features, got {actual_features}"
                )
            X_df = pd.DataFrame(X_df, columns=self.feature_names)

        classes = self.classes_
        class_to_idx = {c: i for i, c in enumerate(classes)}
        n_samples = len(X_df)
        n_classes = len(classes)

        proba_matrix = np.zeros((n_samples, n_classes), dtype=float)

        if self.stage1_model is None or self.stage1_features is None:
            raise RuntimeError("Model must be fitted before calling predict_proba")

        X_s1 = X_df[self.stage1_features]
        fam_probs = self.stage1_model.predict_proba(X_s1)
        fam_classes = self.stage1_model.classes_.tolist()
        fam_cls_idx = {f: i for i, f in enumerate(fam_classes)}

        for fam_name, members in self.family_map.items():
            if fam_name not in fam_cls_idx:
                continue

            f_col_idx = fam_cls_idx[fam_name]
            f_probs = fam_probs[:, f_col_idx]

            if fam_name in self.stage2_models:
                m2 = self.stage2_models[fam_name]
                feats2 = self.stage2_features[fam_name]
                X_s2 = X_df[feats2]
                sub_probs = m2.predict_proba(X_s2)
                sub_classes = m2.classes_.tolist()

                for sub_idx, sub_c in enumerate(sub_classes):
                    if sub_c in class_to_idx:
                        c_idx = class_to_idx[sub_c]
                        proba_matrix[:, c_idx] = f_probs * sub_probs[:, sub_idx]
            else:
                if members and members[0] in class_to_idx:
                    c_idx = class_to_idx[members[0]]
                    proba_matrix[:, c_idx] = f_probs

        return proba_matrix

    def predict(self, X_df: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Vectorized batch prediction for Stage 1 + Stage 2.

        Args:
            X_df (pd.DataFrame | np.ndarray): Input feature matrix.

        Returns:
            np.ndarray: Array of predicted winning distribution class names.
        """
        probs = self.predict_proba(X_df)
        best_indices = np.argmax(probs, axis=1)
        return self.classes_[best_indices]
