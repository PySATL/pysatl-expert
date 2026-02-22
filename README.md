# PySATL Expert


**Pysatl-expert** is experimental expert system designed for the automatic identification of the distribution law of a random variable.

Currently, the system uses an ensemble approach based on bootstrap aggregation and a heuristic decision-making strategy.

---
## Features
- **Modular architecture:** Easily add new distributions, goodness-of-fit criteria, and selection strategies.
- **Domain pre-validation:** Automatically filters distributions based on their theoretical support.
- **Bootstrap analysis:** Estimating the confidence level of identification by repeatedly resampling the source data.
- **ML-Ready:** The system generates standardized feature vectors  ready for training machine learning models.

---

## Architecture

* `core/`: Abstract interfaces and base classes of the system.
* `distributions/`: Implementations of statistical distributions (Normal, Exponential, Weibull).
* `criteria/`: Wrappers over statistical tests.
* `models/`: Reports and feature vectors.
* `strategy/`: The logic for making the final decision.

---

## Decision strategy
The identification algorithm is driven by the `HeuristicStrategy`, which relies on three core principles:
#### 1. Score Unification
   Statistical criteria return results on different scales, the strategy aligns them to a single standard -- the Penalty Score.
   * The lower the score - the better
   * Metrics where "higher is better" are inverted
   * Metrics with low variability are scaled so that their contribution is comparable to more sweeping criteria.
#### 2. Complexity Penalty
   * If two distributions describe the data with equal accuracy, the simpler model (with fewer parameters) is preferred.
#### 3. Bootstrap Consensus
   * The final decision is made based on a vote based on the results of multiple bootstrap iterations. 
   Confidence of the system is calculated as the percentage of votes cast for the winning allocation.

> **Note**:
The current implementation of `HeuristicStrategy` is a **baseline solution**,
with parameters selected empirically.
> 
>The main goal of this strategy is to validate the pipeline and test the hypothesis that the selected set of features is sufficient to separate distribution classes.
>Thanks to the modular architecture, future plans include replacing the heuristic algorithm with a trainable ML classifier (e.g., Random Forest),
which will be able to automatically identify nonlinear relationships between statistical metrics without manually adjusting weights.

---

## Example

```python
import numpy as np
from pysatl_expert.pipeline import DistributionPipeline
from pysatl_expert.core.pipeline_components import PipelineComponents
from pysatl_expert.distributions.normal import NormalDistribution
from pysatl_expert.distributions.exponential import ExponentialDistribution
from pysatl_expert.criteria.selectors.simple_selector import SimpleCriterionSelector
from pysatl_expert.strategy.heuristic_strategy import HeuristicStrategy
from pysatl_expert.models.feature_extractor import FeatureExtractor

# 1. Assembling the system components
components = PipelineComponents(
    distributions=[NormalDistribution(), ExponentialDistribution()],
    criterion_selector=SimpleCriterionSelector(),
    strategy=HeuristicStrategy(),
    feature_extractor=FeatureExtractor()
)

# 2. Initializing the pipeline
pipeline = DistributionPipeline(components)

# 3. Running identification (with 100 bootstrap iterations)
data = np.random.normal(loc=5, scale=2, size=150)
report = pipeline.identify_best(data, n_bootstraps=100)

print(report)
```
