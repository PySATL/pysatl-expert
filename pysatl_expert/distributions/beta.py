import numpy as np
import scipy.stats as st

from pysatl_expert.core.distribution import AbstractDistribution


class BetaDistribution(AbstractDistribution):
    """
    Two-parameter implementation of the Beta probability distribution.

    Defined by two positive shape parameters (alpha, beta). Features strictly
    bounded theoretical support of[0, 1], making it ideal for modeling
    proportions, probabilities, or percentages. The pipeline will automatically
    reject any data sample containing values outside this range.

    Mapping to SciPy: 'alpha' maps to 'a', 'beta' maps to 'b', with
    location fixed to 0 and scale fixed to 1.
    """
    def __init__(self):
        super().__init__(name="Beta", support=(0, 1))

    def fit(self, data: np.ndarray) -> dict:
        a, b, loc, scale = st.beta.fit(data, floc=0, fscale=1)
        return {"alpha": a, "beta": b}

    def pdf(self, data: np.ndarray, params: dict) -> np.ndarray:
        return st.beta.pdf(data, a=params["alpha"], b=params["beta"])

    def cdf(self, data: np.ndarray, params: dict) -> np.ndarray:
        return st.beta.cdf(data, a=params["alpha"], b=params["beta"])
