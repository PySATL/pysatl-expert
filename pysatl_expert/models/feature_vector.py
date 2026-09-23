from pysatl_expert.criteria.catalog import CRITERIA_SPECS, RAW_CRITERIA_EXCLUSIONS


_NUMERICALLY_UNSTABLE_TRAINING_FEATURES = frozenset(RAW_CRITERIA_EXCLUSIONS)


class FeatureVector:
    """Encapsulates statistical evidence for ML classifiers and decision strategies.

    Aggregates continuous sample statistics and GoF test results into a fixed-length vector.

    Attributes:
        STAT_KEYS (list[str]): Key names of descriptive sample statistics.
        CRITERIA_SCHEMA (list[tuple[str, str]]): Ordered list of (dist_name, test_code) tuples.
        sample_stats (dict[str, float]): Model statistics from the full descriptive snapshot.
        descriptive_stats (dict[str, float]): Full snapshot retained for presentation.
        candidates_scores (dict[str, dict[str, float]]): Map of GoF test scores per distribution.
    """

    STAT_KEYS = [
        "sample_size",
        "skew",
        "kurtosis",
        "relative_iqr",
        "entropy",
    ]

    CRITERIA_SCHEMA = [(spec.distribution, spec.short_code) for spec in CRITERIA_SPECS]
    FEATURE_NAMES = STAT_KEYS + [spec.feature_name for spec in CRITERIA_SPECS]
    NUMERICALLY_UNSTABLE_TRAINING_FEATURES = _NUMERICALLY_UNSTABLE_TRAINING_FEATURES
    EXCLUDED_TRAINING_FEATURES = _NUMERICALLY_UNSTABLE_TRAINING_FEATURES
    TRAINING_FEATURE_NAMES = [
        feature
        for feature in FEATURE_NAMES
        if feature not in _NUMERICALLY_UNSTABLE_TRAINING_FEATURES
    ]

    def __init__(self, sample_stats: dict, candidates_scores: dict):
        """Initialize the FeatureVector with sample statistics and GoF scores.

        Args:
            sample_stats (dict): Dictionary of calculated descriptive sample statistics.
            candidates_scores (dict): Dictionary of GoF criterion scores per distribution.
        """
        self.sample_stats = {k: v for k, v in sample_stats.items() if k in self.STAT_KEYS}
        self.descriptive_stats = dict(sample_stats)
        self.candidates_scores = {
            k.lower(): {ck.lower(): cv for ck, cv in v.items()}
            for k, v in candidates_scores.items()
        }

    def as_flat_list(
        self,
        missing_value: float = float("nan"),
        *,
        feature_names: list[str] | None = None,
    ) -> list[float]:
        """Convert aggregated values to a requested model feature schema.

        Args:
            missing_value: Fallback for missing or inapplicable values.
            feature_names: Ordered model schema. Defaults to the current training schema.

        Returns:
            Numerical values in the requested schema order.
        """
        schema = self.FEATURE_NAMES if feature_names is None else feature_names
        values = []
        for feature_name in schema:
            if feature_name in self.STAT_KEYS:
                value = self.sample_stats.get(feature_name, missing_value)
            elif "__" in feature_name:
                distribution, criterion = feature_name.split("__", maxsplit=1)
                value = self.candidates_scores.get(distribution, {}).get(criterion, missing_value)
            else:
                value = missing_value
            values.append(float(value))
        return values

    def as_dict(self) -> dict:
        """Convert feature vector data into a structured dictionary.

        Returns:
            dict: Map with 'stats' and 'scores' nested dictionaries.
        """
        return {"stats": self.sample_stats, "scores": self.candidates_scores}
