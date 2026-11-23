# PySATL Expert

PySATL Expert analyzes one finite numeric sample. Its general API fits explicitly chosen distributions, calculates explicitly chosen raw statistics, and stores results and failures in an `AnalysisResult`. Ready recommendation methods are built on top of that analysis.

## Analyze a sample

This model-free example uses fixed values. It performs no model training, resampling, or Monte Carlo calibration:

```python
from pysatl_expert.app import analyze_sample

result = analyze_sample(
    [0.1, 0.4, 0.8, 1.2, 1.5],
    distributions=["Normal"],
    statistics=["normal__ks"],
)
print(result.sample_stats)
print(result.parameters)
print(result.statistics)  # raw statistic values, not p-values
print(result.failures)
```

The caller chooses distribution names (from `pysatl_expert.distributions.DISTRIBUTION_REGISTRY`, covering 14 continuous families) and statistic IDs. Available IDs depend on the installed Criterion version; inspect `pysatl_expert.criteria.catalog.CRITERIA_SPECS`. A fitted parameter estimate and a raw statistic do not provide calibrated evidence or establish a scientific decision.

## Ready methods and procedures

A ready method declares its `required_statistics` (and optional `procedure_requests`) and implements `predict(sample, analysis)`. The pipeline validates declarations before computation and executes registered `StatisticProcedure` or `ProcedureAdapter` objects.

P-values and critical values require a matching procedure adapter and explicit permission (`allow_calibration=True`). Results retain the procedure identity and calibration method where available. In particular, fitted-parameter raw statistics cannot use the Criterion adapter's simple-hypothesis calibrator; required calibration remains unavailable and blocks a dependent method.

The Criterion adapter targets `pysatl-criterion` `main` (`v1.0.0`, commit `3685d7e746fd7c175e18dc9b113f1db0d4fe3455`): it reads parameter values, supported hypotheses, alternatives, and versioned statistic configuration. `parameter_mode` distinguishes a family hypothesis, explicitly known parameters, and parameters fitted from this sample. Catalog values remain raw statistics and retain their full procedure identity in `AnalysisResult.procedure_results`; `statistics` provides a detached numeric view. A failed fit still permits supported family statistics.

See [method contracts](docs/reference/core.md), [statistics and procedures](docs/reference/criteria.md), and [results and reporting](docs/reference/models.md).

## RF and command line

The `pysatl-expert` CSV command and `recommend_ml` are an RF-backed path. They require a trusted model bundle consisting of a model file and adjacent manifest, compatible with the installed dependencies, Criterion revision, and feature schema. No trained bundle is included with the package.

```console
poetry install
poetry run pysatl-expert --help
```

Criterion is fetched over public HTTPS at the exact commit pinned in
`pyproject.toml` and `poetry.lock`; installation does not require a GitHub SSH key.

For the RF path, consult [the ML adapter](docs/reference/strategy.md) and [public entry points](docs/reference/main.md) before supplying a compatible bundle. A hash check detects accidental artifact mismatch; it does not authenticate who supplied a bundle. Load only artifacts from a trusted source.

## Results and reports

Analysis and recommendation results retain their own evidence and failure information. Render an existing result without repeating analysis or prediction:

```python
from pysatl_expert.reporting.result_report import (
    ReportOptions,
    render_result_report,
    write_result_pdf,
)

print(render_result_report(result))
write_result_pdf(result, "output/result.pdf", ReportOptions(detail="summary"))
```

See [results and reporting](docs/reference/models.md) for result types and report options.

## Tests

Run the non-research tests, including small real sklearn/RF integration cases:

```console
poetry run pytest -m "not research" --ignore=tests/benchmarks
```

Run the Random Forest selection-batching equivalence check separately:

```console
poetry run pytest -m research tests/models/test_hierarchical_model.py
```

The `research` marker does not change the default selection: plain `pytest`
still includes the research check.
