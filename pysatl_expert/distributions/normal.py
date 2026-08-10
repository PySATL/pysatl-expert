"""Normal (Gaussian) probability distribution module."""

import numpy as np
import scipy.stats as st

from pysatl_expert.core.distribution import AbstractDistribution


class NormalDistribution(AbstractDistribution):
    """Two-parameter implementation of the Normal (Gaussian) probability distribution.

    Defined by mean (mu) and standard deviation (std) with universal support (-inf, inf).

    Mapping to SciPy: 'mu' maps to 'loc', 'std' maps to 'scale'.
    """

    def __init__(self):
        """Initialize the Normal distribution with support (-inf, inf)."""
        super().__init__(name="Normal", support=(-np.inf, np.inf))

    def fit(self, data: np.ndarray) -> dict:
        """Estimate mean (mu) and standard deviation (std) via MLE.

        Args:
            data (np.ndarray): 1D array of sample observations.

        Returns:
            dict[str, float]: Map containing 'mu' and 'std' parameters.
        """
        mu, std = st.norm.fit(data)
        return {"mu": mu, "std": std}

    def pdf(self, data: np.ndarray, params: dict) -> np.ndarray:
        """Evaluate the Gaussian probability density function (PDF).

        Args:
            data (np.ndarray): Array of values at which to evaluate the PDF.
            params (dict): Estimated parameter dictionary with 'mu' and 'std'.

        Returns:
            np.ndarray: Computed PDF values.
        """
        return st.norm.pdf(data, loc=params["mu"], scale=params["std"])

    def cdf(self, data: np.ndarray, params: dict) -> np.ndarray:
        """Evaluate the Gaussian cumulative distribution function (CDF).

        Args:
            data (np.ndarray): Array of values at which to evaluate the CDF.
            params (dict): Estimated parameter dictionary with 'mu' and 'std'.

        Returns:
            np.ndarray: Computed CDF values.
        """
        return st.norm.cdf(data, loc=params["mu"], scale=params["std"])
