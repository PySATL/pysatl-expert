from pysatl_criterion.util.distribution import DistributionType
from pysatl_criterion.util.statistic import get_available_criteria


class FeatureVector:
    """
    Data Transfer Object  defining the feature space for ML classifiers.

    Aggregates disparate statistical evidence into a high-dimensional,
    fixed-length numerical array.

    The vector is composed of:
    1. Sample Statistics: Fundamental shape and complexity metrics (skew, entropy).
    2. GoF Scores: Results from a dynamic array of criteria defined by the
       global CRITERIA_SCHEMA.

    If a test is mathematically inapplicable or fails, its position in the
    vector is preserved and filled with a 'missing_value' (-1.0), serving
    as a categorical indicator for the decision tree nodes.
    """

    STAT_KEYS = ["sample_size", "skew", "kurtosis", "coef_of_variation", "relative_iqr", "entropy"]

    CRITERIA_SCHEMA = []
    BLACKLIST = ["bhs", "kl_int", "kl_sup", "cq*", "rs", "ahs", "hp"]

    for dist in DistributionType:
        dist_name = dist.value.lower()
        available_tests = get_available_criteria(dist)

        for crit_code in available_tests:
            clean_code = crit_code.lower()
            if clean_code not in BLACKLIST:
                CRITERIA_SCHEMA.append((dist_name, clean_code))

    CRITERIA_SCHEMA = sorted(CRITERIA_SCHEMA)

    def __init__(self, sample_stats: dict, candidates_scores: dict):
        self.sample_stats = {k: v for k, v in sample_stats.items() if k in self.STAT_KEYS}
        self.candidates_scores = {
            k.lower(): {ck.lower(): cv for ck, cv in v.items()}
            for k, v in candidates_scores.items()
        }

    def as_flat_list(self, missing_value: float = -1.0) -> list[float]:
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
        return {"stats": self.sample_stats, "scores": self.candidates_scores}
