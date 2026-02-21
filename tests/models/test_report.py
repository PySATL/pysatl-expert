from unittest.mock import MagicMock

from pysatl_expert.models.report import Report


def test_report_init():
    all_scores = {"aic": 10.5}
    params = {"mu": 0, "sigma": 1}
    ranks = {"norm": 1, "expon": 2}
    report = Report("norm", 0.95, all_scores, params, ranks)

    assert report.distribution_name == "norm"
    assert report.confidence == 0.95
    assert report.all_scores == all_scores
    assert report.parameters == params
    assert report.final_ranks == ranks


def test_report_str_json_success():
    all_scores = {"aic": 10.5, "bic": 12.0}
    report = Report("norm", 0.9, all_scores)
    output = str(report)

    assert "Winner:      norm" in output
    assert "Confidence:  0.9" in output
    assert '"aic": 10.5' in output


def test_report_str_numpy_serialization():
    mock_numpy_val = MagicMock()
    mock_numpy_val.item.return_value = 42
    all_scores = {"val": mock_numpy_val}

    report = Report("norm", 0.8, all_scores)
    output = str(report)

    assert "42" in output
    mock_numpy_val.item.assert_called_once()


def test_report_str_type_error_fallback():
    all_scores = {complex(1, 2): "unserializable_key"}
    report = Report("norm", 0.7, all_scores)
    output = str(report)

    assert "Detailed Scores:" in output
    assert "(1+2j)" in output


def test_report_repr():
    report = Report("norm", 0.9, {"a": 1})
    assert repr(report) == str(report)


def test_report_safe_serialize_no_item():
    all_scores = {"test": "simple_string"}
    report = Report("norm", 1.0, all_scores)
    output = str(report)
    assert "simple_string" in output


def test_report_str_unserializable_value_no_item():
    report = Report(
        distribution_name="norm", confidence=1.0, all_scores={"test_val": complex(1, 2)}
    )

    output = str(report)
    assert "Winner:      norm" in output
    assert "(1+2j)" in output
