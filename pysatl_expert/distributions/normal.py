import numpy as np
import scipy.stats as st

from pysatl_expert.core.distribution import AbstractDistribution


class NormalDistribution(AbstractDistribution):
    """
    Two-parameter implementation of the Normal (Gaussian) probability distribution.

    Defined by mean (mu) and standard deviation (std). Features universal
    support (-inf, inf), making it mathematically compatible with any real-valued sample.

    Mapping to SciPy: 'mu' maps to 'loc', 'std' maps to 'scale'.
    """

    def __init__(self):
        """
        Initializes the distribution with universal theoretical support (-inf, inf).
        """
        super().__init__(name="Normal", support=(-np.inf, np.inf))

    def fit(self, data: np.ndarray) -> dict:
        """
        Estimates 'mu' and 'std' parameters via Maximum Likelihood Estimation (MLE).
        """
        mu, std = st.norm.fit(data)
        return {"mu": mu, "std": std}

    def pdf(self, data: np.ndarray, params: dict) -> np.ndarray:
        """
        Evaluates the Gaussian probability density function (PDF).
        """
        return st.norm.pdf(data, loc=params["mu"], scale=params["std"])

    def cdf(self, data: np.ndarray, params: dict) -> np.ndarray:
        """
        Evaluates the cumulative distribution function (CDF) for GoF analysis.
        """
        return st.norm.cdf(data, loc=params["mu"], scale=params["std"])
