from pathlib import Path

import pandas as pd
import pytest

from pysatl_expert.models.feature_vector import FeatureVector
from scripts.training_data import (
    load_frozen_training_dataset,
    load_frozen_training_partition,
)


def test_load_frozen_training_dataset_returns_the_two_partitions(tmp_path: Path):
    rows = []
    for target, split, value in (
        ("Normal", "train", 1.0),
        ("Uniform", "train", 2.0),
        ("Normal", "outer_test", 3.0),
    ):
        row = {feature: value for feature in FeatureVector.FEATURE_NAMES}
        row.update({"Target": target, "Split": split})
        rows.append(row)
    path = tmp_path / "frozen_training.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    training, outer_test = load_frozen_training_dataset(path)

    assert training["Target"].tolist() == ["Normal", "Uniform"]
    assert outer_test["Target"].tolist() == ["Normal"]
    assert list(training.columns) == [*FeatureVector.FEATURE_NAMES, "Target"]


def test_load_frozen_training_dataset_rejects_unknown_split(tmp_path: Path):
    row = {feature: 1.0 for feature in FeatureVector.FEATURE_NAMES}
    row.update({"Target": "Normal", "Split": "validation"})
    path = tmp_path / "frozen_training.csv"
    pd.DataFrame([row]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="Split column"):
        load_frozen_training_dataset(path)


def test_load_frozen_training_partition_streams_train_rows(tmp_path: Path):
    rows = []
    for target, split, value in (
        ("Normal", "train", 1.0),
        ("Uniform", "train", 2.0),
        ("Normal", "outer_test", 3.0),
    ):
        row = {feature: value for feature in FeatureVector.FEATURE_NAMES}
        row.update({"Target": target, "Split": split})
        rows.append(row)
    path = tmp_path / "frozen_training.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    training, outer_test_rows = load_frozen_training_partition(path, chunksize=2)

    assert training["Target"].tolist() == ["Normal", "Uniform"]
    assert outer_test_rows == 1
    assert "Split" not in training
