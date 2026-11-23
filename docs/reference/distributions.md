# Distribution fitting

`analyze_sample` accepts explicit names from the centralized catalog in `pysatl_expert.distributions.DISTRIBUTION_REGISTRY`: Normal, Exponential, Weibull, Uniform, Student, Gamma, Beta, LogNormal, Laplace, Pareto, LogLogistic, InverseGamma, ExponentiatedWeibull, and Hyperbolic. It fits each requested distribution to the supplied sample and returns fitted parameters with sample summaries, requested raw statistics, and per-operation failures. It does not rank or select a distribution.

The available raw statistic IDs depend on the installed Criterion revision. Use `pysatl_expert.criteria.catalog.CRITERIA_SPECS` to inspect the active catalog. A statistic value is not a p-value, calibrated result, or decision. Fitted parameters describe the fit on this sample; they do not establish that a distribution is appropriate.

::: pysatl_expert.distributions
::: pysatl_expert.distributions.exponential
::: pysatl_expert.distributions.normal
::: pysatl_expert.distributions.weibull
