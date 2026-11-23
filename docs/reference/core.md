# Method contracts

`recommend` runs a caller-supplied (or pipeline-configured) ready method against
recorded sample analysis. Methods declare `required_statistics` (and optional
`procedure_requests`) and expose `predict(sample, analysis)`. Expert does not
train the method or assess its scientific quality.

## Runnable technical example

This model-free example demonstrates orchestration with a technical stub:

```python
from types import SimpleNamespace

from pysatl_expert.core.pipeline_components import PipelineComponents
from pysatl_expert.models.feature_extractor import FeatureExtractor
from pysatl_expert.pipeline import DistributionPipeline


class ReadyMean:
    required_statistics = ("sample_mean",)

    def predict(self, sample, analysis):
        return {"technical_mean": analysis.special_statistics["sample_mean"]}


pipeline = DistributionPipeline(
    PipelineComponents([], None, ReadyMean(), FeatureExtractor())
)
result = pipeline.recommend(
    [1.0, 2.0, 3.0],
    procedures={"sample_mean": SimpleNamespace(calculate=lambda sample: sum(sample) / len(sample))},
)
assert result.error is None
assert result.prediction == {"technical_mean": 2.0}
```

## Ready methods and procedures

::: pysatl_expert.core.recommendation

## Pipeline components

::: pysatl_expert.core.pipeline_components
