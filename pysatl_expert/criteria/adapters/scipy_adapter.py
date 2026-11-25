"""SciPy Shapiro-Wilk test exposed through Expert's procedure contract."""

import warnings

import numpy as np
import scipy
from scipy.stats import shapiro

from pysatl_expert.core.recommendation import ProcedureRequest
from pysatl_expert.models.analysis_result import ProcedureResult


class ScipyShapiroAdapter:
    """Normality statistic and SciPy p-value; critical values are unavailable.

    SciPy computes W and p together, even when only W is requested. Its p-value
    may be inaccurate for samples above 5000, so this adapter withholds it.
    """

    def evaluate(
        self, sample: np.ndarray, request: ProcedureRequest, *, allow_calibration: bool
    ) -> ProcedureResult:
        values = np.asarray(sample, dtype=float)
        if values.ndim != 1 or values.size < 3:
            raise ValueError("Shapiro-Wilk requires a one-dimensional sample of at least 3")
        if not np.all(np.isfinite(values)) or np.ptp(values) == 0:
            raise ValueError("Shapiro-Wilk requires a finite, non-constant sample")
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="scipy.stats.shapiro: For N > 5000.*", category=UserWarning
            )
            result = shapiro(values)
        unavailable = {}
        p_value = None
        if request.p_value:
            if not allow_calibration:
                unavailable["p_value"] = "calibration not enabled"
            elif values.size > 5000:
                unavailable["p_value"] = "SciPy warns p-value may be inaccurate for N > 5000"
            else:
                p_value = float(result.pvalue)
        if request.critical_values:
            unavailable["critical_values"] = "SciPy shapiro does not provide critical values"
        return ProcedureResult(
            statistic=float(result.statistic),
            alternative="less",
            p_value=p_value,
            source="scipy.stats.shapiro",
            hypothesis_description="sample drawn from a normal distribution",
            configuration={"scipy_version": scipy.__version__},
            calibration_method="scipy.stats.shapiro" if p_value is not None else None,
            unavailable=unavailable,
        )
