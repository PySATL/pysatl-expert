# Results and reporting

`AnalysisResult` stores sample summaries, fitted parameters, raw catalog
statistics, procedure results, and reasons for unavailable evidence.
`RecommendationResult` retains its analysis even when prediction fails.

Catalog procedure facts have one canonical record in `procedure_results`.
`catalog_ids` maps distribution/test names to those IDs. `statistics` is a
detached numeric view of these facts plus any legacy numeric constructor data;
editing this view does not change stored results. Duplicate catalog ID aliases
are rejected; canonical facts take precedence over legacy values for the same
catalog entry.

The former `identify_best` / `predict_report` / `Report` path has been replaced
by canonical result objects. Render a stored result with one reporting API;
rendering does not fit distributions, call procedures, or predict again.

## Result types

::: pysatl_expert.models.analysis_result

## Reporting an existing result

`ReportOptions` supports the `summary` and `full` detail presets, explicit
ordered sections, numeric precision, and an optional prediction formatter.
Text and PDF use the same selected facts. PNG plots use the original sample
snapshot retained in the result; without a snapshot, the plot shows an explicit
unavailable panel. Reporting never recomputes inference.

```python
from pysatl_expert.reporting.result_report import (
    ReportOptions,
    render_result_report,
    write_result_pdf,
)

options = ReportOptions(sections=("sample", "recommendation", "procedures"))
print(render_result_report(result, options))
write_result_pdf(result, "output/report.pdf", options)
```

::: pysatl_expert.reporting.result_report.ReportOptions
::: pysatl_expert.reporting.result_report.render_result_report
::: pysatl_expert.reporting.result_report.write_result_pdf

## Trusted model artifacts

This section applies to the RF path. The adapter verifies the adjacent
manifest and model digest before joblib deserialization. A hash detects
accidental mismatch; it does not authenticate the sender. Load bundles only
from a trusted source. The package does not include a trained model or training
dataset. A usable bundle must match the installed dependency revision and its
exact feature schema.

::: pysatl_expert.models.model_manifest
