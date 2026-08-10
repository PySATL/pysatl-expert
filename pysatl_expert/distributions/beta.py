"""Beta probability distribution module."""

import numpy as np
import scipy.stats as st

from pysatl_expert.core.distribution import AbstractDistribution


class BetaDistribution(AbstractDistribution):
    """Two-parameter implementation of the Beta probability distribution.

    Defined by shape parameters alpha and beta with strictly bounded theoretical support [0, 1].

    Mapping to SciPy: 'alpha' maps to 'a', 'beta' maps to 'b' with floc=0, fscale=1.
    """

    def __init__(self):
        """Initialize the Beta distribution with support (0, 1)."""
        super().__init__(name="Beta", support=(0, 1))

    def fit(self, data: np.ndarray) -> dict:
        """Estimate shape parameters (alpha, beta) via MLE.

        Args:
            data (np.ndarray): 1D array of sample observations in (0, 1).

        Returns:
            dict[str, float]: Map containing 'alpha' and 'beta' parameters.
        """
        a, b, loc, scale = st.beta.fit(data, floc=0, fscale=1)
        return {"alpha": a, "beta": b}

    def pdf(self, data: np.ndarray, params: dict) -> np.ndarray:
        """Evaluate the Beta probability density function (PDF).

        Args:
            data (np.ndarray): Array of values at which to evaluate the PDF.
            params (dict): Estimated parameter dictionary with 'alpha' and 'beta'.

        Returns:
            np.ndarray: Computed PDF values.
        """
        return st.beta.pdf(data, a=params["alpha"], b=params["beta"])

    def cdf(self, data: np.ndarray, params: dict) -> np.ndarray:
        """Evaluate the Beta cumulative distribution function (CDF).

        Args:
            data (np.ndarray): Array of values at which to evaluate the CDF.
            params (dict): Estimated parameter dictionary with 'alpha' and 'beta'.

        Returns:
            np.ndarray: Computed CDF values.
        """
        return st.beta.cdf(data, a=params["alpha"], b=params["beta"])
