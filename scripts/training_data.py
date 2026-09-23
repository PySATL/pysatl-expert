"""Load frozen datasets for feature selection and HRF training."""

from pathlib import Path

import numpy as np
import pandas as pd

from pysatl_expert.models.feature_vector import FeatureVector


def load_frozen_training_dataset(
    path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load one frozen CSV and return its train and outer-test partitions."""
    selected_columns = [*FeatureVector.FEATURE_NAMES, "Target", "Split"]
    numeric_types = dict.fromkeys(FeatureVector.FEATURE_NAMES, np.float32)
    frame = pd.read_csv(path, usecols=selected_columns, dtype=numeric_types)
    if np.isinf(frame[FeatureVector.FEATURE_NAMES].to_numpy()).any():
        raise ValueError(f"Infinite features in {path}")

    if frame["Split"].isna().any():
        raise ValueError("Frozen dataset Split column must not contain missing values")
    split_values = set(frame["Split"].unique())
    if split_values != {"train", "outer_test"}:
        raise ValueError("Frozen dataset Split column must contain train and outer_test rows")

    feature_columns = [*FeatureVector.FEATURE_NAMES, "Target"]
    training = frame.loc[frame["Split"] == "train", feature_columns].reset_index(drop=True)
    outer_test = frame.loc[frame["Split"] == "outer_test", feature_columns].reset_index(drop=True)
    if training.empty or outer_test.empty:
        raise ValueError("Frozen dataset must contain non-empty train and outer-test splits")
    return training, outer_test


def load_frozen_training_partition(
    path: Path,
    *,
    chunksize: int = 50_000,
) -> tuple[pd.DataFrame, int]:
    """Load only train rows from a frozen CSV and count outer-test rows."""
    if chunksize < 1:
        raise ValueError("chunksize must be positive")

    selected_columns = [*FeatureVector.FEATURE_NAMES, "Target", "Split"]
    numeric_types = dict.fromkeys(FeatureVector.FEATURE_NAMES, np.float32)
    training_chunks: list[pd.DataFrame] = []
    outer_test_rows = 0
    split_values: set[str] = set()
    for frame in pd.read_csv(
        path,
        usecols=selected_columns,
        dtype=numeric_types,
        chunksize=chunksize,
    ):
        if np.isinf(frame[FeatureVector.FEATURE_NAMES].to_numpy()).any():
            raise ValueError(f"Infinite features in {path}")
        if frame["Split"].isna().any():
            raise ValueError("Frozen dataset Split column must not contain missing values")
        split_values.update(frame["Split"].unique())
        training_chunks.append(
            frame.loc[frame["Split"] == "train", [*FeatureVector.FEATURE_NAMES, "Target"]]
        )
        outer_test_rows += int((frame["Split"] == "outer_test").sum())

    if split_values != {"train", "outer_test"}:
        raise ValueError("Frozen dataset Split column must contain train and outer_test rows")
    training = pd.concat(training_chunks, ignore_index=True)
    if training.empty or outer_test_rows == 0:
        raise ValueError("Frozen dataset must contain non-empty train and outer-test splits")
    return training, outer_test_rows
