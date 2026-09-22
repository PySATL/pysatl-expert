import math

import scipy.stats as stats


_SCIPY_SPECS = {
    "normal": (stats.norm, ("mu", "std"), ("loc", "scale"), (1,)),
    "lognormal": (stats.lognorm, ("s", "loc", "scale"), ("s", "loc", "scale"), (0, 2)),
    "exponential": (stats.expon, ("loc", "scale"), ("loc", "scale"), (1,)),
    "student": (stats.t, ("df", "loc", "scale"), ("df", "loc", "scale"), (0, 2)),
    "gamma": (stats.gamma, ("shape", "loc", "scale"), ("a", "loc", "scale"), (0, 2)),
    "weibull": (
        stats.weibull_min,
        ("shape", "loc", "scale"),
        ("c", "loc", "scale"),
        (0, 2),
    ),
    "beta": (stats.beta, ("alpha", "beta"), ("a", "b"), (0, 1)),
}


def _finite_parameters(params: dict, *names: str) -> tuple[float, ...] | None:
    """Return required finite parameters, or None when a fitted curve is unavailable."""
    if not all(name in params for name in names):
        return None
    values = tuple(float(params[name]) for name in names)
    return values if all(math.isfinite(value) for value in values) else None


def get_scipy_dist(dist_name: str, params: dict):
    """Return a SciPy distribution only when all fitted parameters are available."""
    name = dist_name.lower().replace("_", "")
    if name == "uniform":
        values = _finite_parameters(params, "a", "b")
        if values is not None and values[1] > values[0]:
            return stats.uniform(loc=values[0], scale=values[1] - values[0])
        return None

    spec = _SCIPY_SPECS.get(name)
    if spec is None:
        return None
    scipy_dist, parameter_names, scipy_parameter_names, positive_indices = spec
    values = _finite_parameters(params, *parameter_names)
    if values is None or any(values[index] <= 0 for index in positive_indices):
        return None

    scipy_params = dict(zip(scipy_parameter_names, values, strict=True))
    if name == "beta":
        scipy_params.update(loc=0.0, scale=1.0)
    return scipy_dist(**scipy_params)
