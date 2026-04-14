import numpy as np
import scipy.stats as st

from pysatl_expert.core.distribution import AbstractDistribution


class StudentDistribution(AbstractDistribution):
    """
    Three-parameter implementation of the Student's t-distribution.

    Defined by degrees of freedom (df), location (loc), and scale.
    Features universal theoretical support (-inf, inf). It is particularly
    useful for modeling data with 'heavy tails' compared to the Normal distribution.

    Mapping to SciPy: 'df', 'loc', and 'scale' are fitted dynamically.
    """
    def __init__(self):
        super().__init__(name="Student", support=(-np.inf, np.inf))

    def fit(self, data: np.ndarray) -> dict:
        df, loc, scale = st.t.fit(data)
        return {"df": df, "loc": loc, "scale": scale}

    def pdf(self, data: np.ndarray, params: dict) -> np.ndarray:
        return st.t.pdf(data, df=params["df"], loc=params["loc"], scale=params["scale"])

    def cdf(self, data: np.ndarray, params: dict) -> np.ndarray:
        return st.t.cdf(data, df=params["df"], loc=params["loc"], scale=params["scale"])
