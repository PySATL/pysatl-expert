"""Central registry of supported candidate distributions."""

from collections.abc import Callable

from pysatl_expert.core.distribution import AbstractDistribution
from pysatl_expert.distributions.beta import BetaDistribution
from pysatl_expert.distributions.exponential import ExponentialDistribution
from pysatl_expert.distributions.exponentiated_weibull import ExponentiatedWeibullDistribution
from pysatl_expert.distributions.gamma import GammaDistribution
from pysatl_expert.distributions.hyperbolic import HyperbolicDistribution
from pysatl_expert.distributions.inverse_gamma import InverseGammaDistribution
from pysatl_expert.distributions.laplace import LaplaceDistribution
from pysatl_expert.distributions.log_logistic import LogLogisticDistribution
from pysatl_expert.distributions.log_normal import LogNormalDistribution
from pysatl_expert.distributions.normal import NormalDistribution
from pysatl_expert.distributions.pareto import ParetoDistribution
from pysatl_expert.distributions.student import StudentDistribution
from pysatl_expert.distributions.uniform import UniformDistribution
from pysatl_expert.distributions.weibull import WeibullDistribution


DISTRIBUTION_REGISTRY: dict[str, Callable[[], AbstractDistribution]] = {
    "Normal": NormalDistribution,
    "Exponential": ExponentialDistribution,
    "Weibull": WeibullDistribution,
    "Uniform": UniformDistribution,
    "Student": StudentDistribution,
    "Gamma": GammaDistribution,
    "Beta": BetaDistribution,
    "LogNormal": LogNormalDistribution,
    "Laplace": LaplaceDistribution,
    "Pareto": ParetoDistribution,
    "LogLogistic": LogLogisticDistribution,
    "InverseGamma": InverseGammaDistribution,
    "ExponentiatedWeibull": ExponentiatedWeibullDistribution,
    "Hyperbolic": HyperbolicDistribution,
}


def create_all_distributions() -> list[AbstractDistribution]:
    """Instantiate every registered distribution adapter."""
    return [factory() for factory in DISTRIBUTION_REGISTRY.values()]
