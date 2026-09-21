import numpy as np
import scipy.stats as st

from pysatl_expert.core.distribution import AbstractDistribution


class GammaDistribution(AbstractDistribution):
    """
    Fixed-origin implementation of the Gamma probability distribution.

    Defined by a shape parameter (a) and a scale parameter, with loc fixed at zero.
    Support is [0, inf), matching the experiment generator used for training data.

    Mapping to SciPy: 'shape' maps to 'a', with ``floc=0``.
    """

    def __init__(self):
        super().__init__(name="Gamma", support=(0.0, np.inf))

    def fit(self, data: np.ndarray) -> dict:
        shape, _, scale = st.gamma.fit(data, floc=0.0)
        return {"shape": shape, "loc": 0.0, "scale": scale}

    def pdf(self, data: np.ndarray, params: dict) -> np.ndarray:
        loc = params.get("loc", 0)
        return st.gamma.pdf(data, a=params["shape"], loc=loc, scale=params["scale"])

    def cdf(self, data: np.ndarray, params: dict) -> np.ndarray:
        loc = params.get("loc", 0)
        return st.gamma.cdf(data, a=params["shape"], loc=loc, scale=params["scale"])

    def prepare_criterion_input(
        self, data: np.ndarray, params: dict
    ) -> tuple[np.ndarray, dict]:
        """Standardize observations for canonical Gamma criteria."""
        observations = self._standardize_criterion_input(
            data,
            params.get("loc", 0.0),
            params.get("scale", 1.0),
            positive_support=True,
        )
        return observations, {"alpha": float(params["shape"]), "beta": 1.0}
