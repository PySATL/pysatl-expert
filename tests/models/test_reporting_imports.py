import numpy as np
import pytest

from pysatl_expert.models.analysis_result import AnalysisResult, RecommendationResult
from pysatl_expert.reporting.common import get_scipy_dist
from pysatl_expert.reporting.plot import generate_plot_report
from pysatl_expert.reporting.result_report import (
    ReportOptions,
    render_result_report,
    write_result_pdf,
)


def _result(snapshot=(0.0, 0.5, 1.0), params=None):
    analysis = AnalysisResult(
        sample_stats=(
            {"n": len(snapshot), "mean": float(np.mean(snapshot))} if snapshot is not None else {}
        ),
        parameters={"Uniform": params} if params is not None else {},
        statistics={},
        failures={},
        sample_snapshot=snapshot,
    )
    return RecommendationResult(
        analysis,
        {"label": "Uniform", "scores": {"Uniform": 0.9, "Beta": 0.1}},
    )


def test_supported_reporting_entrypoints_import_and_run(tmp_path):
    result = _result()

    assert "Answer: Uniform" in render_result_report(result)
    pdf_path = tmp_path / "report.pdf"
    assert write_result_pdf(result, pdf_path) == pdf_path
    assert pdf_path.read_bytes().startswith(b"%PDF-")


@pytest.mark.parametrize(
    ("name", "params"),
    [
        ("Normal", {"mu": 2.0, "std": 3.0}),
        ("LogNormal", {"s": 0.5, "loc": 0.0, "scale": 2.0}),
        ("Exponential", {"loc": 0.0, "scale": 2.0}),
        ("Uniform", {"a": -1.0, "b": 3.0}),
        ("Student", {"df": 4.0, "loc": 1.0, "scale": 2.0}),
        ("Gamma", {"shape": 2.0, "loc": 0.0, "scale": 3.0}),
        ("Weibull", {"shape": 1.5, "loc": 0.0, "scale": 2.0}),
        ("Beta", {"alpha": 2.0, "beta": 5.0}),
    ],
)
def test_fitted_distribution_adapter_accepts_each_runtime_fit_format(name, params):
    assert get_scipy_dist(name, params) is not None


@pytest.mark.parametrize(
    ("name", "params"),
    [
        ("Normal", {}),
        ("LogNormal", {"s": 0.5, "scale": 2.0}),
        ("Exponential", {"loc": 0.0}),
        ("Uniform", {"a": 1.0, "b": 1.0}),
        ("Student", {"df": 4.0, "loc": 0.0, "scale": 0.0}),
        ("Gamma", {"shape": -1.0, "loc": 0.0, "scale": 2.0}),
        ("Weibull", {"shape": 1.5, "loc": 0.0}),
        ("Beta", {"alpha": 2.0}),
        ("Normal", {"mu": "invalid", "std": 1.0}),
    ],
)
def test_fitted_distribution_adapter_marks_incomplete_parameters_unavailable(name, params):
    assert get_scipy_dist(name, params) is None


def test_plot_report_writes_png_to_nested_parent_and_closes_figure(tmp_path):
    import matplotlib.pyplot as plt

    result = _result(params={"a": 0.0, "b": 1.0})
    initial_figures = plt.get_fignums()
    path = tmp_path / "nested" / "report.png"

    assert generate_plot_report(result, path) == path
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert plt.get_fignums() == initial_figures


def test_pdf_and_png_share_the_same_result_chart_helper(tmp_path, monkeypatch):
    from pysatl_expert.reporting import plot

    generated = []
    original = plot._build_result_figure

    def capture(result):
        generated.append(result)
        return original(result)

    monkeypatch.setattr(plot, "_build_result_figure", capture)
    result = _result(params={"a": 0.0, "b": 1.0})
    write_result_pdf(result, tmp_path / "report.pdf", ReportOptions())
    generate_plot_report(result, tmp_path / "report.png", ReportOptions())
    assert generated == [result, result]


def test_plot_without_snapshot_shows_unavailable_reason_saves_png_and_closes_figure(
    tmp_path, monkeypatch
):
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    captured_text = []
    savefig = Figure.savefig

    def capture(figure, *args, **kwargs):
        captured_text.extend(text.get_text() for text in figure.texts)
        return savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    result = _result(snapshot=None)
    output = tmp_path / "unavailable.png"
    initial_figures = plt.get_fignums()

    generate_plot_report(result, output)

    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert any("Charts unavailable: no sample snapshot" in text for text in captured_text)
    assert plt.get_fignums() == initial_figures
