from pysatl_criterion.statistics.exponent import (
    GiniExponentialityGofStatistic,
    MoranExponentialityGofStatistic,
)
from pysatl_criterion.statistics.normal import (
    AndersonDarlingNormalityGofStatistic,
    ShapiroWilkNormalityGofStatistic,
)
from pysatl_criterion.statistics.weibull import (
    AndersonDarlingWeibullGofStatistic,
    TikuSinghWeibullGofStatistic,
)

from pysatl_expert.core.criterion_selector import AbstractCriterionSelector
from pysatl_expert.criteria.calculate.generic import GenericCriterion


class SimpleCriterionSelector(AbstractCriterionSelector):
    """
    Standard implementation of the criterion selector.

    Maps supported distributions to their specific Goodness-of-Fit tests
    implemented in the external 'pysatl_criterion' library.
    """

    def __init__(self):
        """
        Initializes the selector with a predefined map of statistical tests.
        """
        super().__init__()
        self._criteria_map = {
            "Normal": [
                GenericCriterion(AndersonDarlingNormalityGofStatistic(), "anderson_darling"),
                GenericCriterion(ShapiroWilkNormalityGofStatistic(), "shapiro_wilk"),
            ],
            "Exponential": [
                GenericCriterion(GiniExponentialityGofStatistic(), "gini_index"),
                GenericCriterion(MoranExponentialityGofStatistic(), "moran_test"),
            ],
            "Weibull": [
                GenericCriterion(AndersonDarlingWeibullGofStatistic(), "anderson_darling"),
                GenericCriterion(TikuSinghWeibullGofStatistic(), "tiku_singh"),
            ],
        }
        self._default_criteria = []

    def get_applicable_criteria(self, data, distribution) -> list:
        """
        Retrieves the list of criteria mapped to the given distribution name.
        """
        return self._criteria_map.get(distribution.name, self._default_criteria)
