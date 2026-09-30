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
        values = np.asarray(data, dtype=float)
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError("Beta fitting requires a non-empty finite sample")
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError("Beta fitting requires every observation to lie in [0, 1]")

        # Continuous Beta samples are theoretically inside (0, 1), but storage and
        # floating-point generation can round an extreme draw to exactly 0 or 1.
        epsilon = np.finfo(float).eps
        interior = np.clip(values, epsilon, 1.0 - epsilon)
        a, b, _, _ = st.beta.fit(interior, floc=0, fscale=1)
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

    def prepare_criterion_input(self, data: np.ndarray, params: dict) -> tuple[np.ndarray, dict]:
        """Validate and move boundary observations into the open interval."""
        observations = np.asarray(data, dtype=float)
        if np.any(observations < 0.0) or np.any(observations > 1.0):
            raise ValueError("Beta criteria require every observation to lie in [0, 1]")
        epsilon = np.finfo(float).eps
        return np.clip(observations, epsilon, 1.0 - epsilon), params
