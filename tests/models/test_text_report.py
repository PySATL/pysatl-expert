import numpy as np
import pytest

from pysatl_expert.models.feature_extractor import FeatureExtractor
from pysatl_expert.models.report import Report
from pysatl_expert.reporting.formatting import format_raw_statistic
from pysatl_expert.reporting.text import generate_text_report


def _report(data: np.ndarray, *args, **kwargs) -> Report:
    """Build a report with the same statistics snapshot as the pipeline."""
    return Report(
        *args,
        sample_statistics=FeatureExtractor().calculate_sample_stats(data),
        **kwargs,
    )


def test_text_report_shows_raw_values_without_hypothesis_decisions():
    data = np.array([0.1, 0.2, 0.4])
    report = _report(
        data,
        "Normal",
        0.8,
        {"Normal": {"zero": 0.0, "one": 1.0, "negative": -1.0, "missing": np.nan}},
        final_ranks={"Normal": 0.8, "Student": 0.2},
    )
    text = generate_text_report(data, report)
    assert "zero: 0" in text
    assert "one: 1" in text
    assert "negative: -1" in text
    assert "missing: —" in text
    assert "Finite statistics: 3 / 4" in text
    assert "H0" not in text
    assert "Passed" not in text
    assert "Rejected" not in text
    assert "Probability Ranking" not in text


@pytest.mark.parametrize("value", [None, np.nan, np.inf, -np.inf])
def test_text_report_nonfinite_statistics_have_dash(value):
    data = np.array([1.0, 2.0, 3.0])
    report = _report(data, "Gamma", 0.6, {"Gamma": {"stat": value}})
    text = generate_text_report(data, report)
    assert "stat: —" in text
    assert "Finite statistics: 0 / 1" in text


def test_text_report_keeps_original_ranking_separate_from_bootstrap():
    data = np.array([1.0, 2.0, 4.0])
    report = _report(
        data,
        "Normal",
        0.75,
        {},
        final_ranks={"Normal": 0.75, "Student": 0.25},
        model_ranks={"Normal": 0.75, "Student": 0.25},
        bootstrap_ranks={"Normal": 0.25, "Student": 0.75},
        bootstrap_requested=4,
        bootstrap_successful=4,
        bootstrap_stability=0.25,
    )
    text = generate_text_report(data, report)
    assert "Model Score Ranking" in text
    assert "Normal  :  75.00%" in text
    assert "Bootstrap Stability:  25.00%" in text


def test_text_report_without_winner_statistics_does_not_invent_summary():
    data = np.array([1.0, 2.0, 3.0])
    report = _report(data, "Normal", 0.8, {"Gamma": {"ad": 1.0}})
    text = generate_text_report(data, report)
    assert "Raw GoF statistics" not in text


def test_text_report_preserves_legacy_flat_scores_as_raw_values():
    data = np.array([1.0, 2.0, 3.0])
    report = _report(data, "Normal", 0.8, {"zero": 0.0, "one": 1.0})
    text = generate_text_report(data, report)
    assert "zero: 0" in text
    assert "one: 1" in text
    assert "H0" not in text


def test_raw_formatting_marks_nonfinite_values_as_unavailable():
    assert format_raw_statistic(np.nan) == "—"


def test_text_report_uses_statistics_snapshot_instead_of_input_data():
    report = Report(
        "Normal",
        0.8,
        {},
        sample_statistics={
            "sample_size": 3,
            "min": 10.0,
            "max": 20.0,
            "mean": 15.0,
            "standard_deviation": 5.0,
            "skew": 0.0,
            "kurtosis": -1.5,
        },
    )

    text = generate_text_report(np.array([100.0, 200.0, 300.0]), report)

    assert "[10.0000, 20.0000]" in text
    assert "Mean = 15.0000 | Std = 5.0000" in text
