"""Construct exact Criterion parameter values for catalog calculations."""

from pysatl_criterion.distribution.parameters import ParameterValues


# Source keys are supplied by Expert's distribution adapters. Target names are
# the public names in Criterion's declared parameterizations; no aliases/defaults
# are inferred here.
_FITTED_COORDINATES = {
    "normal": {"normal.mean_var": {"mean": "mean", "var": "var"}},
    "exponential": {"exponential.rate": {"lam": "lam"}},
    "weibull": {"weibull.scale_shape": {"scale": "a", "shape": "k"}},
    "uniform": {"uniform.lower_upper": {"a": "a", "b": "b"}},
    "student": {"student.df_loc_scale": {"df": "df", "loc": "loc", "scale": "scale"}},
    "gamma": {"gamma.shape_rate": {"alfa": "alpha", "beta": "beta"}},  # codespell:ignore alfa
    "beta": {"beta.alpha_beta": {"a": "alpha", "b": "beta"}},
    "lognormal": {"log_normal.shape_scale": {"s": "s", "scale": "scale"}},
    "laplace": {"laplace.loc_scale": {"t": "t", "s": "s"}},
    "pareto": {"pareto.shape_scale": {"shape": "shape", "scale": "scale"}},
    "loglogistic": {"log_logistic.scale_shape": {"alpha": "alpha", "beta": "beta"}},
    "inversegamma": {"inverse_gamma.shape_scale": {"alpha": "alpha", "beta": "beta"}},
    "exponentiatedweibull": {
        "exponentiated_weibull.exponent_shape_scale": {
            "exponent": "exponent",
            "shape": "shape",
            "scale": "scale",
        }
    },
    "hyperbolic": {
        "hyperbolic.shape_skewness_scale_loc": {
            "alpha": "alpha",
            "beta": "beta",
            "delta": "delta",
            "mu": "mu",
        }
    },
}


def family_parameters_for_statistic(statistic_class) -> ParameterValues | None:
    """Return the declared empty-hypothesis schema, if the statistic supports it."""
    supports = tuple(statistic_class.supported_hypotheses())
    if not supports:
        raise ValueError(f"{statistic_class.__name__} declares no supported hypotheses")
    descriptor = statistic_class.distribution()
    default_schema = descriptor.default_parameterization()
    native = [
        support
        for support in supports
        if support.parameterization == default_schema and not support.fixed_parameters
    ]
    if len(native) == 1:
        return default_schema.parse({})
    if len(native) > 1:
        raise ValueError(f"Ambiguous empty hypothesis for {statistic_class.__name__}")
    return None


def fitted_parameters_for_statistic(
    statistic_class, family: str, prepared: dict[str, float]
) -> ParameterValues:
    """Map explicit Expert fit coordinates to one exact supported hypothesis."""
    supports = tuple(statistic_class.supported_hypotheses())
    if not supports:
        raise ValueError(f"{statistic_class.__name__} declares no supported hypotheses")

    try:
        schema_mappings = _FITTED_COORDINATES[family]
    except KeyError:
        raise ValueError(f"No explicit Criterion parameter mapping for {family!r}") from None

    candidates: list[tuple[ParameterValues, int]] = []
    for support in supports:
        mapping = schema_mappings.get(support.parameterization.id)
        if mapping is None:
            continue
        available = {
            criterion_name: prepared[expert_name]
            for criterion_name, expert_name in mapping.items()
            if expert_name in prepared
        }
        fixed_names = {parameter.name for parameter in support.fixed_parameters}
        if not fixed_names <= available.keys():
            continue
        values = support.parameterization.parse({name: available[name] for name in fixed_names})
        if support.supports(values):
            candidates.append((values, len(fixed_names)))

    if not candidates:
        raise ValueError(
            f"{statistic_class.__name__} has no supported hypothesis covered by prepared "
            f"{family} parameters"
        )
    most_specific = max(size for _, size in candidates)
    selected = [values for values, size in candidates if size == most_specific]
    identities = {values.storage_identity() for values in selected}
    if len(identities) != 1:
        raise ValueError(f"Ambiguous fitted hypothesis for {statistic_class.__name__}")
    return selected[0]
