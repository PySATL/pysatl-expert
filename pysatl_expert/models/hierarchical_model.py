import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


class HierarchicalExpertModel:
    """Production wrapper for 2-stage hierarchical distribution classification."""

    def __init__(self, family_map: dict[str, list[str]]):
        """Initialize the hierarchical model with distribution family mappings.

        Args:
            family_map (dict[str, list[str]]): Mapping of family names to member distributions.
        """
        self.family_map = family_map
        self.dist_to_family = {}
        for fam_name, dists in family_map.items():
            for d in dists:
                self.dist_to_family[d] = fam_name

        self.feature_names: list[str] | None = None
        self.stage1_model: RandomForestClassifier | None = None
        self.stage1_features: list[str] | None = None

        self.stage2_models: dict[str, RandomForestClassifier] = {}
        self.stage2_features: dict[str, list[str]] = {}

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
    ):
        """Train Stage 1 family model and Stage 2 sub-family models.

        Args:
            X_df (pd.DataFrame): Feature DataFrame.
            y_series (pd.Series): Target distribution class names Series.
            n_stage1 (int | None): Number of top features for Stage 1 model (None for all).
            n_stage2 (int | None): Number of top features for Stage 2 sub-models (None for all).
        """
        self.feature_names = X_df.columns.tolist()
        y_families = y_series.map(self.dist_to_family)

        # --- Stage 1: Family Model ---
        if n_stage1 is not None and n_stage1 < len(X_df.columns):
            rf_stage1 = RandomForestClassifier(
                n_estimators=200, max_depth=15, random_state=42, n_jobs=2
            )
            rf_stage1.fit(X_df, y_families)
            importances1 = pd.Series(rf_stage1.feature_importances_, index=X_df.columns)
            self.stage1_features = importances1.nlargest(n_stage1).index.tolist()
        else:
            self.stage1_features = self.feature_names

        self.stage1_model = RandomForestClassifier(
            n_estimators=200, max_depth=12, random_state=42, n_jobs=2
        )
        self.stage1_model.fit(X_df[self.stage1_features], y_families)

        # --- Stage 2: Sub-Family Models ---
        for fam_name, dist_members in self.family_map.items():
            mask = y_series.isin(dist_members)
            X_fam = X_df[mask]
            y_fam = y_series[mask]

            if len(dist_members) <= 1:
                continue

            if n_stage2 is not None and n_stage2 < len(X_fam.columns):
                rf_fam = RandomForestClassifier(
                    n_estimators=200, max_depth=15, random_state=42, n_jobs=2
                )
                rf_fam.fit(X_fam, y_fam)
                importances2 = pd.Series(rf_fam.feature_importances_, index=X_fam.columns)
                top_fam_features = importances2.nlargest(n_stage2).index.tolist()
            else:
                top_fam_features = X_fam.columns.tolist()

            fam_model = RandomForestClassifier(
                n_estimators=200, max_depth=12, random_state=42, n_jobs=2
            )
            fam_model.fit(X_fam[top_fam_features], y_fam)

            self.stage2_models[fam_name] = fam_model
            self.stage2_features[fam_name] = top_fam_features

    def predict_proba(self, X_df: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Vectorized batch predict probability matrix across all distribution classes.

        Args:
            X_df (pd.DataFrame | np.ndarray): Input feature matrix.

        Returns:
            np.ndarray: Predicted probability matrix of shape (n_samples, n_classes).
        """
        if isinstance(X_df, np.ndarray):
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
