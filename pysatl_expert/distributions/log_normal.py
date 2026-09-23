import numpy as np
import scipy.stats as st

from pysatl_expert.core.distribution import AbstractDistribution


class LogNormalDistribution(AbstractDistribution):
    """
    Fixed-origin implementation of the Log-Normal probability distribution.

    Defined by a shape parameter (s) and scale, with loc fixed at zero.
    Support is [0, inf), matching the experiment generator used for training data.

    Mapping to SciPy: 's' maps to shape and 'scale' is exp(mean), with ``floc=0``.
    """

    def __init__(self):
        super().__init__(name="LogNormal", support=(0.0, np.inf))

    def fit(self, data: np.ndarray) -> dict:
        shape, _, scale = st.lognorm.fit(data, floc=0.0)
        return {"s": shape, "loc": 0.0, "scale": scale}

    def pdf(self, data: np.ndarray, params: dict) -> np.ndarray:
        loc = params["loc"]
        return st.lognorm.pdf(data, s=params["s"], loc=loc, scale=params["scale"])

    def cdf(self, data: np.ndarray, params: dict) -> np.ndarray:
        loc = params["loc"]
        return st.lognorm.cdf(data, s=params["s"], loc=loc, scale=params["scale"])

    def prepare_criterion_input(self, data: np.ndarray, params: dict) -> tuple[np.ndarray, dict]:
        """Standardize observations for canonical Log-Normal criteria."""
        observations = self._standardize_criterion_input(
            data,
            params["loc"],
            params["scale"],
            positive_support=True,
        )
        return observations, {"s": float(params["s"]), "scale": 1.0}
