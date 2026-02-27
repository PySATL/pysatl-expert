class FeatureVector:
    """
    Data Transfer Object (DTO) that standardizes the feature space for decision strategies.

    This class aggregates disparate data points—intrinsic sample statistics and
    multi-distribution goodness-of-fit scores—into a unified structure. Its primary
    purpose is to provide a consistent numerical representation of the statistical
    evidence, suitable for both heuristic analysis and machine learning inference.

    Attributes:
        STAT_KEYS (list): Standardized set of sample-profiling keys.
        CRITERIA_KEYS (list): Predefined sequence of statistical criteria to
            maintain a fixed-length feature vector.
    """

    STAT_KEYS = ["sample_size", "skew", "kurtosis", "coef_of_variation", "relative_iqr", "entropy"]
    CRITERIA_KEYS = [
        "shapiro_wilk",
        "anderson_darling",
        "ks_test",
        "jarque_bera",
        "lilliefors",
        "cramer_von_mises",
        "gini_index",
        "moran_test",
        "ahs_test",
        "msf_test",
        "tiku_singh",
    ]

    def __init__(self, sample_stats: dict, candidates_scores: dict):
        """
        Initializes the vector with filtered sample stats and candidate scores.

        Args:
            sample_stats (dict): Metadata describing the raw data sample.
            candidates_scores (dict): Nested mapping of distribution names to
                their respective criterion scores.
        """
        self.sample_stats = {k: v for k, v in sample_stats.items() if k in self.STAT_KEYS}
        self.candidates_scores = candidates_scores

    def as_flat_list(self) -> list[float]:
        """
        Transforms structured statistical data into a flattened numerical array.

        This method ensures a deterministic order of features, which is critical for
        predictive models. It iterates through fixed keys and sorted distribution
        names to produce a stable input vector for ML classifiers.

        Returns:
            list[float]: A flat list of features representing the entire state
                of the identification experiment.
        """
        flat_vector = []

        for key in self.STAT_KEYS:
            flat_vector.append(self.sample_stats.get(key, 0.0))

        sorted_dist_names = sorted(self.candidates_scores.keys())
        for dist_name in sorted_dist_names:
            dist_scores = self.candidates_scores[dist_name]

            for crit_key in self.CRITERIA_KEYS:
                val = dist_scores.get(crit_key, 0.0)
                flat_vector.append(float(val))

        return flat_vector

    def as_dict(self) -> dict:
        """
        Returns a dictionary representation for logging or reporting purposes.
        """
        return {"stats": self.sample_stats, "scores": self.candidates_scores}
