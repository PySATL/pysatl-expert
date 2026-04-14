import inspect
import logging

from pysatl_criterion.util.distribution import DistributionType
from pysatl_criterion.util.statistic import get_available_criteria

from pysatl_expert.core.criterion_selector import AbstractCriterionSelector
from pysatl_expert.criteria.calculate.generic import GenericCriterion


logger = logging.getLogger(__name__)


class DynamicCriterionSelector(AbstractCriterionSelector):
    """
    Selector for automated statistical test discovery.

    Dynamically scans the 'pysatl-criterion' library to identify all applicable
    Goodness-of-Fit tests for a given distribution.

    Features:
        - Runtime Safety: Utilizes a 'blacklist' to skip computationally expensive
          tests that might cause system timeouts.
    """

    def __init__(self):
        super().__init__()
        self._criteria_cache = {}
        self.BLACKLIST = ["bhs", "kl_int", "kl_sup", "cq*", "rs", "ahs", "hp"]

    def get_applicable_criteria(self, data, distribution) -> list:
        dist_name = distribution.name.lower()

        if dist_name in self._criteria_cache:
            return self._criteria_cache[dist_name]

        criteria_list = []
        try:
            dist_type = DistributionType(dist_name)
        except ValueError:
            logger.warning(f"'{distribution.name}' distribution not found in DistributionType.")
            return []

        available_short_codes = get_available_criteria(dist_type)
        base_class = dist_type.base_class

        def get_all_concrete_subclasses(cls):
            subclasses = set()
            for subclass in cls.__subclasses__():
                if not inspect.isabstract(subclass) and not subclass.__name__.startswith("Abstract"):
                    subclasses.add(subclass)
                subclasses.update(get_all_concrete_subclasses(subclass))
            return subclasses

        for stat_class in get_all_concrete_subclasses(base_class):
            try:
                if hasattr(stat_class, 'short_code') and stat_class.short_code() in available_short_codes:
                    criterion_name = stat_class.short_code().lower()

                    if criterion_name in self.BLACKLIST:
                        continue

                    instance = stat_class()
                    criterion = GenericCriterion(instance, display_name=criterion_name)
                    criteria_list.append(criterion)
                    available_short_codes.remove(stat_class.short_code())
            except Exception as e:
                logger.debug(f"Initial error {stat_class.__name__}: {e}")

        self._criteria_cache[dist_name] = criteria_list
        return criteria_list
