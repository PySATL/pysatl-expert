"""Discover and classify goodness-of-fit statistics available to pysatl-expert."""

from dataclasses import dataclass

from pysatl_criterion import DistributionType
from pysatl_criterion.statistics.goodness_of_fit.goodness_of_fit_statistic import (
    AbstractGoodnessOfFitStatistic,
)
from pysatl_criterion.utils.statistic import get_available_criteria


DISTRIBUTIONS: dict[str, DistributionType] = {
    "normal": DistributionType.NORMAL,
    "exponential": DistributionType.EXPONENTIAL,
    "weibull": DistributionType.WEIBULL,
    "uniform": DistributionType.UNIFORM,
    "student": DistributionType.STUDENT,
    "gamma": DistributionType.GAMMA,
    "beta": DistributionType.BETA,
    "lognormal": DistributionType.LOG_NORMAL,
    "laplace": DistributionType.LAPLACE,
    "pareto": DistributionType.PARETO,
    "loglogistic": DistributionType.LOG_LOGISTIC,
    "inversegamma": DistributionType.INVERSE_GAMMA,
    "exponentiatedweibull": DistributionType.EXPONENTIATED_WEIBULL,
    "hyperbolic": DistributionType.HYPERBOLIC,
}

GLOBAL_BLACKLIST = {
    "bhs",
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
PENDING_VALIDATION_CODES = {"cq*"}

RAW_CRITERIA_EXCLUSIONS = {
    "beta__chi2_pearson": "non-finite values when expected bin frequencies are zero",
    "beta__lillie": "exact duplicate of beta__ks in the completed raw dataset",
    "beta__mode": "mode formula is invalid when either fitted Beta shape is at most one",
    "exponential__ww": "unstable ratio after fitted-location canonicalization",
    "gamma__mt": "non-finite results and values above the float32 range in the balanced pilot",
    "lognormal__glb": "non-finite values in distribution tails",
    "lognormal__rj": "exact duplicate of lognormal__lg in the completed raw dataset",
    "lognormal__sh": "overflow for larger samples",
    "lognormal__zwa": "non-finite values in distribution tails",
    "lognormal__zwc": "non-finite values in distribution tails",
    "normal__glb": "non-finite values in distribution tails",
    "normal__rj": "exact duplicate of normal__lg in the completed raw dataset",
    "normal__sh": "overflow for larger samples",
    "normal__zwa": "non-finite values in distribution tails",
    "normal__zwc": "non-finite values in distribution tails",
    "student__lillie": "exact duplicate of student__ks in the completed raw dataset",
    "uniform__censored_stein_u": (
        "exact duplicate of uniform__stein_u for uncensored generated samples"
    ),
    "uniform__lillie": "exact duplicate of uniform__ks in the completed raw dataset",
    "weibull__lt2": "numerically explosive exponential transform",
    "weibull__lt3": "numerically explosive exponential transform",
    "weibull__lillie": "exact duplicate of weibull__ks in the completed raw dataset",
    "weibull__ls": "values above the float32 range in the balanced pilot",
    "weibull__mt": "non-finite results and values above the float32 range in the balanced pilot",
    "weibull__chi2_pearson": (
        "explodes when equal-width tail bins have near-zero expected probability"
    ),
}


@dataclass(frozen=True)
class CriterionSpec:
    """Connect a feature name to its statistic implementation."""

    distribution: str
    short_code: str
    statistic_class: type[AbstractGoodnessOfFitStatistic]

    @property
    def feature_name(self) -> str:
        """Return the feature name used by datasets and models."""
        return f"{self.distribution}__{self.short_code}"


@dataclass(frozen=True)
class SkippedCriterion:
    """Describe a discovered criterion excluded from the current prototype."""

    distribution: str
    short_code: str
    full_code: str
    reason: str


def _available_classes(
    distribution_type: DistributionType,
) -> dict[str, type[AbstractGoodnessOfFitStatistic]]:
    resolved: dict[str, type[AbstractGoodnessOfFitStatistic]] = {}
    for statistic_class in sorted(
        get_available_criteria(distribution_type),
        key=lambda item: (item.__module__, item.__qualname__),
    ):
        short_code = statistic_class.short_code().lower()
        if short_code in resolved:
            raise ValueError(
                f"Duplicate Criterion short code {short_code!r} for {distribution_type.name}"
            )
        resolved[short_code] = statistic_class
    return resolved


def _discover_catalog() -> tuple[list[CriterionSpec], list[SkippedCriterion]]:
    active: list[CriterionSpec] = []
    skipped: list[SkippedCriterion] = []
    for distribution, distribution_type in DISTRIBUTIONS.items():
        for short_code, statistic_class in sorted(_available_classes(distribution_type).items()):
            feature_name = f"{distribution}__{short_code}"
            reason: str | None
            if short_code in GLOBAL_BLACKLIST:
                reason = (
                    "pending_validation"
                    if short_code in PENDING_VALIDATION_CODES
                    else "blacklisted"
                )
            else:
                reason = RAW_CRITERIA_EXCLUSIONS.get(feature_name)
            if reason is None:
                active.append(CriterionSpec(distribution, short_code, statistic_class))
            else:
                skipped.append(
                    SkippedCriterion(
                        distribution,
                        short_code,
                        statistic_class.code().upper(),
                        reason,
                    )
                )

    active.sort(key=lambda item: (item.distribution, item.short_code))
    skipped.sort(key=lambda item: (item.reason, item.distribution, item.short_code))
    return active, skipped


CRITERIA_SPECS, SKIPPED_CRITERIA = _discover_catalog()
CRITERIA_SPEC_BY_KEY = {(spec.distribution, spec.short_code): spec for spec in CRITERIA_SPECS}
CRITERIA_REGISTRY = {
    distribution: tuple(spec for spec in CRITERIA_SPECS if spec.distribution == distribution)
    for distribution in DISTRIBUTIONS
}
