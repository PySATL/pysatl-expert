import numpy as np
import scipy.stats as st

from pysatl_expert.core.distribution import AbstractDistribution


class StudentDistribution(AbstractDistribution):
    """Three-parameter implementation of Student's t-distribution.

    Defined by degrees of freedom (df), location (loc), and scale with domain (-inf, inf).

    Mapping to SciPy: 'df', 'loc', and 'scale' fitted via MLE.
    """

    def __init__(self):
        """Initialize Student's t-distribution with support (-inf, inf)."""
        super().__init__(name="Student", support=(-np.inf, np.inf))

    def fit(self, data: np.ndarray) -> dict:
        """Estimate degrees of freedom (df), location (loc), and scale via MLE.

        Args:
            data (np.ndarray): 1D array of sample observations.

        Returns:
            dict[str, float]: Map containing 'df', 'loc', and 'scale' parameters.
        """
        values = np.asarray(data, dtype=float)
        center = float(np.mean(values))
        reference_scale = float(np.std(values))
        if not np.isfinite(reference_scale) or reference_scale <= 0:
            raise ValueError("Student fitting requires a non-constant finite sample")

        standardized = (values - center) / reference_scale
        df, loc, scale = st.t.fit(standardized)
        return {
            "df": df,
            "loc": center + loc * reference_scale,
            "scale": scale * reference_scale,
        }

    def pdf(self, data: np.ndarray, params: dict) -> np.ndarray:
        """Evaluate the Student's t probability density function (PDF).

        Args:
            data (np.ndarray): Array of values at which to evaluate the PDF.
            params (dict): Estimated parameter dictionary with 'df', 'loc', and 'scale'.

        Returns:
            np.ndarray: Computed PDF values.
        """
        return st.t.pdf(data, df=params["df"], loc=params["loc"], scale=params["scale"])

    def cdf(self, data: np.ndarray, params: dict) -> np.ndarray:
        """Evaluate the Student's t cumulative distribution function (CDF).

        Args:
            data (np.ndarray): Array of values at which to evaluate the CDF.
            params (dict): Estimated parameter dictionary with 'df', 'loc', and 'scale'.

        Returns:
            np.ndarray: Computed CDF values.
        """
        return st.t.cdf(data, df=params["df"], loc=params["loc"], scale=params["scale"])

    def prepare_criterion_input(
        self, data: np.ndarray, params: dict
    ) -> tuple[np.ndarray, dict]:
        """Standardize observations for canonical Student criteria."""
        observations = self._standardize_criterion_input(
            data, params["loc"], params["scale"]
        )
        return observations, {
            "df": float(params["df"]),
            "loc": 0.0,
            "scale": 1.0,
        }
