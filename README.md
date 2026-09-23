# PySATL Expert

`pysatl-expert` is an experimental system that recommends a probability
distribution for one empirical numeric sample. It combines parameter estimation,
goodness-of-fit statistics from `pysatl-criterion`, and a two-stage Random Forest
model.

The current model distinguishes eight distributions: Normal, Student, Exponential,
Gamma, LogNormal, Weibull, Beta, and Uniform.

## Prototype scope

The result is an uncalibrated model recommendation, not a probability that a
statistical hypothesis is true.

Exponential, Gamma, LogNormal, and Weibull use the prototype policy `loc = 0`.
Beta uses fixed support `[0, 1]`. Arbitrarily shifted samples are therefore not a
supported identification scenario for those distributions.

## Installation

`pysatl-criterion` is installed from its GitHub repository. Its resolved revision
is recorded in `poetry.lock`.

```bash
poetry install
```

## Run an analysis

The command accepts a CSV containing exactly one numeric column and requires a
trusted model bundle:

```bash
poetry run pysatl-expert \
  --input sample.csv \
  --model path/to/model.joblib \
  --output output/report.pdf
```

For a header row such as `value`, add `--skip-header-rows 1`. Run
`poetry run pysatl-expert --help` for delimiter, bootstrap, and output options.
The command does not overwrite an existing report.

The default report calculates only the statistics required by the loaded model.
`--pdf-mode full` displays those selected model inputs; it does not calculate every
registered criterion or change the ranking.

## Use from Python

Build the pipeline once when analyzing several samples:

```python
from pysatl_expert.app import build_pipeline
from pysatl_expert.reporting.pdf import generate_pdf_report

pipeline = build_pipeline(model_path="path/to/model.joblib")
report = pipeline.identify_best(sample, n_bootstraps=0, random_state=42)
generate_pdf_report(sample, report, output_path="report.pdf", mode="summary")
```

`identify_best` does not write files or print output. `evaluate_sample` remains a
convenience wrapper for one-off console, PDF, and PNG output.

## Model bundle

A model bundle contains two adjacent files:

- `<model>.joblib` — the serialized hierarchical model;
- `<model>.manifest.json` — its integrity and compatibility metadata.

The manifest must have the same stem as the model. Before loading the trusted
joblib file, the application verifies the model hash, feature schema, and runtime
versions recorded in the manifest. Obtain both files from the same release.

## Reproducible training

Training uses one local frozen CSV, which is intentionally excluded from Git. It
must contain all active feature columns, `Target`, and a `Split` column with exactly
`train` and `outer_test` values.

Set the local CSV path in `pysatl_expert/config/feature_selection.json`, then run:

```bash
# Select fixed Stage 1 and Stage 2 feature sets from the train partition.
poetry run python scripts/select_features.py

# Train and evaluate the final model using that fixed selection.
poetry run python scripts/train_model.py
```

`select_features.py` ranks candidate inputs automatically while respecting the
project's explicit exclusions. It writes the selected feature names and training
configuration to `selected_features.json`. `train_model.py` validates that document,
fits the final Stage 1 and Stage 2 forests, evaluates once on `outer_test`, and writes
the model, metrics, and manifest.
