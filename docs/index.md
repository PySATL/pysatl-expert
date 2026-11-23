# PySATL Expert

PySATL Expert analyzes one finite numeric sample. Start with model-free analysis: choose distributions and raw statistic IDs explicitly, then inspect the returned summaries, fitted parameters, statistics, and failures.

```python
from pysatl_expert.app import analyze_sample

result = analyze_sample(
    [0.1, 0.4, 0.8, 1.2, 1.5],
    distributions=["Normal"],
    statistics=["normal__ks"],
)
print(result.sample_stats, result.parameters)
print(result.statistics, result.failures)
```

This example uses fixed data and performs no model training, resampling, or calibration. Statistic values are raw values, not p-values or decisions. Available IDs are listed by `pysatl_expert.criteria.catalog.CRITERIA_SPECS` for the installed Criterion version (`v1.0.0`).

## Ready methods and procedures

Ready methods declare their required statistics (and optional `procedure_requests`) and implement `predict(sample, analysis)`. Connecting a predictor does not train it or validate its quality.

Calibration requires a matching `ProcedureAdapter` and explicit permission (`allow_calibration=True`). Fitted-parameter statistics cannot use the Criterion adapter's simple-hypothesis calibrator. Procedure identity and calibration details are retained in `AnalysisResult.procedure_results`.

## RF and CLI mode

The CSV CLI and `recommend_ml` are RF-backed paths that require a trusted model bundle and adjacent manifest compatible with the installed Criterion revision (`3685d7e746fd7c175e18dc9b113f1db0d4fe3455`) and feature schema. No trained model is included with the package.

See [public entry points](reference/main.md), [method contracts](reference/core.md), [statistics and procedures](reference/criteria.md), [result types and reporting](reference/models.md), and [the RF adapter](reference/strategy.md).
