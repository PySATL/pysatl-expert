# Public entry points

## Model-free analysis

`analyze_sample` is the general first step. It accepts explicit distribution names and statistic IDs and returns an `AnalysisResult`. It does not require an RF model, choose a distribution, or calibrate raw statistics.

::: pysatl_expert.app.analyze_sample

## Ready methods and reusable pipelines

Use the ready-method contracts in [method contracts](core.md). The pipeline can run supplied methods and explicit procedures while retaining their canonical results.

::: pysatl_expert.app.build_pipeline

## RF recommendation

`recommend_ml` is an RF-backed path. It validates the trusted model and adjacent manifest before sample analysis. It requires a bundle compatible with this installation's Criterion revision and the exact model input schema; the package does not ship such a model.

::: pysatl_expert.app.recommend_ml

## Convenience reporting

The convenience function `pysatl_expert.app.evaluate_sample` returns the
canonical result that it renders. For custom sections or precision, render an existing result explicitly with
[`ReportOptions`](models.md#reporting-an-existing-result).

## Pipeline operations

::: pysatl_expert.pipeline.DistributionPipeline
    options:
      members:
        - analyze_catalog
        - analyze_procedures
        - recommend
