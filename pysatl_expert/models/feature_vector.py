from pysatl_criterion.distribution.distribution_type import DistributionType
from pysatl_criterion.utils.statistic import get_available_criteria


class FeatureVector:
    """Encapsulates statistical evidence for ML classifiers and decision strategies.

    Aggregates continuous sample statistics and GoF test results into a fixed-length vector.

    Attributes:
        STAT_KEYS (list[str]): Key names of descriptive sample statistics.
        CRITERIA_SCHEMA (list[tuple[str, str]]): Ordered list of (dist_name, test_code) tuples.
        sample_stats (dict[str, float]): Map of calculated descriptive statistics.
        candidates_scores (dict[str, dict[str, float]]): Map of GoF test scores per distribution.
    """

    STAT_KEYS = [
        "min",
        "max",
        "sample_size",
        "skew",
        "kurtosis",
        "coef_of_variation",
        "relative_iqr",
        "entropy",
    ]

    CRITERIA_SCHEMA = []
    BLACKLIST = {
        "bhs",
        "kl_int",
        "kl_sup",
        "cq*",
        "rs",
        "ahs",
        "hp",
        "independencenumber",
        "cliquenumber",
        "avgdegree",
        "edgesnumber",
        "maxdegree",
        "connectedcomponents",
    }

    for dist in DistributionType:
        dist_name = dist.value.lower()
        available_tests = get_available_criteria(dist)

        for crit_code in available_tests:
            clean_code = crit_code.lower()
            if clean_code not in BLACKLIST:
                CRITERIA_SCHEMA.append((dist_name, clean_code))

    CRITERIA_SCHEMA = sorted(CRITERIA_SCHEMA)

    def __init__(self, sample_stats: dict, candidates_scores: dict):
        """Initialize the FeatureVector with sample statistics and GoF scores.

        Args:
            sample_stats (dict): Dictionary of calculated descriptive sample statistics.
            candidates_scores (dict): Dictionary of GoF criterion scores per distribution.
        """
        self.sample_stats = {k: v for k, v in sample_stats.items() if k in self.STAT_KEYS}
        self.candidates_scores = {
            k.lower(): {ck.lower(): cv for ck, cv in v.items()}
            for k, v in candidates_scores.items()
        }

    def as_flat_list(self, missing_value: float = -1.0) -> list[float]:
        """Convert aggregated features into a 1D flat list for ML model input.

        Args:
            missing_value (float): Fallback value for missing/inapplicable tests. Defaults to -1.0.

        Returns:
            list[float]: Flat numerical vector matching the schema order.
        """
        flat_vector = []

        for key in self.STAT_KEYS:
            val = self.sample_stats.get(key, missing_value)
            flat_vector.append(float(val))

        for dist_name, crit_key in self.CRITERIA_SCHEMA:
            if dist_name in self.candidates_scores:
                val = self.candidates_scores[dist_name].get(crit_key, missing_value)
                flat_vector.append(float(val))
            else:
                flat_vector.append(float(missing_value))

        return flat_vector

    def as_dict(self) -> dict:
        """Convert feature vector data into a structured dictionary.

        Returns:
            dict: Map with 'stats' and 'scores' nested dictionaries.
        """
        return {"stats": self.sample_stats, "scores": self.candidates_scores}
