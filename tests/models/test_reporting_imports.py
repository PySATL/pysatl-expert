import numpy as np
import pytest

from pysatl_expert.models.report import Report


def test_pdf_and_plot_have_reporting_home():
    from pysatl_expert.reporting.pdf import generate_pdf_report
    from pysatl_expert.reporting.plot import generate_plot_report

    assert generate_pdf_report.__module__ == "pysatl_expert.reporting.pdf"
    assert generate_plot_report.__module__ == "pysatl_expert.reporting.plot"


def test_fitted_distribution_adapter_is_shared_between_reports():
    from pysatl_expert.reporting import pdf, plot
    from pysatl_expert.reporting.common import get_scipy_dist

    assert pdf._get_scipy_dist is get_scipy_dist
    assert plot._get_scipy_dist is get_scipy_dist
    fitted = get_scipy_dist("Uniform", {"a": 2.0, "b": 5.0})
    np.testing.assert_allclose(fitted.cdf([2.0, 3.5, 5.0]), [0.0, 0.5, 1.0])


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
    from pysatl_expert.reporting.common import get_scipy_dist

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
    ],
)
def test_fitted_distribution_adapter_never_invents_missing_parameters(name, params):
    from pysatl_expert.reporting.common import get_scipy_dist

    assert get_scipy_dist(name, params) is None


def test_plot_report_writes_png_and_closes_figure(tmp_path):
    import matplotlib.pyplot as plt

    from pysatl_expert.reporting.plot import generate_plot_report

    report = Report(
        "Normal", 0.8, {}, parameters={"mu": 0.0, "std": 1.0},
        final_ranks={"Normal": 0.8, "Student": 0.2},
    )
    initial_figures = plt.get_fignums()
    path = tmp_path / "report.png"
    assert generate_plot_report(np.linspace(-2.0, 2.0, 20), report, path) == path
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert plt.get_fignums() == initial_figures


def test_plot_report_does_not_draw_a_fitted_curve_without_fitted_parameters(
    tmp_path, monkeypatch
):
    from matplotlib.axes import Axes

    from pysatl_expert.reporting.plot import generate_plot_report

    labels = []
    original_plot = Axes.plot

    def capture(axis, *args, **kwargs):
        labels.append(kwargs.get("label"))
        return original_plot(axis, *args, **kwargs)

    monkeypatch.setattr(Axes, "plot", capture)
    report = Report("Normal", 0.8, {}, final_ranks={"Normal": 0.8})

    generate_plot_report(np.linspace(-2.0, 2.0, 20), report, tmp_path / "report.png")

    assert all(
        label is None or ("Fitted" not in label and "Theoretical" not in label)
        for label in labels
    )
