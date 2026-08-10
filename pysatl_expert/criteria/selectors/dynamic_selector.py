import logging

from pysatl_criterion.distribution.distribution_type import DistributionType
from pysatl_criterion.statistics.goodness_of_fit.beta import AbstractBetaGofStatistic
from pysatl_criterion.statistics.goodness_of_fit.exponent import AbstractExponentialityGofStatistic
from pysatl_criterion.statistics.goodness_of_fit.gamma import AbstractGammaGofStatistic
from pysatl_criterion.statistics.goodness_of_fit.log_normal import AbstractLogNormalGofStatistic
from pysatl_criterion.statistics.goodness_of_fit.normal import AbstractNormalityGofStatistic
from pysatl_criterion.statistics.goodness_of_fit.student import AbstractStudentGofStatistic
from pysatl_criterion.statistics.goodness_of_fit.uniform import AbstractUniformGofStatistic
from pysatl_criterion.statistics.goodness_of_fit.weibull import AbstractWeibullGofStatistic
from pysatl_criterion.utils.statistic import get_available_criteria

from pysatl_expert.core.criterion_selector import AbstractCriterionSelector
from pysatl_expert.criteria.calculate.generic import GenericCriterion
from pysatl_expert.models.feature_vector import FeatureVector


logger = logging.getLogger(__name__)

BASE_MAP = {
    "normal": (DistributionType.NORMAL, AbstractNormalityGofStatistic),
    "exponential": (DistributionType.EXPONENTIAL, AbstractExponentialityGofStatistic),
    "weibull": (DistributionType.WEIBULL, AbstractWeibullGofStatistic),
    "uniform": (DistributionType.UNIFORM, AbstractUniformGofStatistic),
    "student": (DistributionType.STUDENT, AbstractStudentGofStatistic),
    "gamma": (DistributionType.GAMMA, AbstractGammaGofStatistic),
    "beta": (DistributionType.BETA, AbstractBetaGofStatistic),
    "lognormal": (DistributionType.LOG_NORMAL, AbstractLogNormalGofStatistic),
    "log_normal": (DistributionType.LOG_NORMAL, AbstractLogNormalGofStatistic),
}


def _build_criteria_registry() -> dict[str, list[type]]:
    """Build a static registry mapping distribution names to concrete statistic classes."""
    registry = {}
    for dist_name, (dt, base_cls) in BASE_MAP.items():
        if dist_name == "log_normal":
            continue
        try:
            available_codes = set(get_available_criteria(dt))
        except Exception as e:
            logger.warning(f"Could not load criteria codes for {dist_name}: {e}")
            continue

        def walk_subclasses(cls):
            res = []
            for sub in cls.__subclasses__():
                try:
                    if hasattr(sub, "short_code"):
                        code = sub.short_code()
                        if code in available_codes and code.lower() not in FeatureVector.BLACKLIST:
                            res.append(sub)
                except Exception:
                    pass
                res.extend(walk_subclasses(sub))
            return res

        valid_classes = walk_subclasses(base_cls)
        registry[dist_name] = valid_classes
        if dist_name == "lognormal":
            registry["log_normal"] = valid_classes

    return registry


CRITERIA_REGISTRY = _build_criteria_registry()


class DynamicCriterionSelector(AbstractCriterionSelector):
    """Selector backed by a pre-built static registry of PySATL statistics.

    Provides fast, deterministic lookups of goodness-of-fit criteria per distribution.
    """

    def __init__(self, registry: dict[str, list[type]] | None = None):
        """Initialize selector with optional custom criteria registry."""
        super().__init__()
        self.registry = registry if registry is not None else CRITERIA_REGISTRY
        self._cache = {}

    def get_applicable_criteria(self, data, distribution) -> list[GenericCriterion]:
        """Retrieve applicable criteria for a given candidate distribution model."""
        dist_key = distribution.name.lower()

        if dist_key in self._cache:
            return self._cache[dist_key]

        stat_classes = self.registry.get(dist_key, [])
        criteria = []
        for stat_cls in stat_classes:
            try:
                instance = stat_cls()
                criterion_name = stat_cls.short_code().lower()
                criterion = GenericCriterion(instance, display_name=criterion_name)
                criteria.append(criterion)
            except Exception as e:
                logger.debug(f"Could not instantiate {stat_cls.__name__}: {e}")

        self._cache[dist_key] = criteria
        return criteria
