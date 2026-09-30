import numpy as np
import pytest

from pysatl_expert.models.report import Report
from pysatl_expert.reporting.common import get_scipy_dist


def test_evaluate_sample_uses_explicit_model(monkeypatch):
    from pysatl_expert import app

    captured = {}

    def strategy(**kwargs):
        captured.update(kwargs)
        return object()

    class FakePipeline:
        def __init__(self, components):
            pass

        def identify_best(self, data, n_bootstraps, random_state):
            return Report("Normal", 0.8, {}, final_ranks={"Normal": 0.8})

    monkeypatch.setattr(app, "MLStrategy", strategy)
    monkeypatch.setattr(app, "DistributionPipeline", FakePipeline)
    monkeypatch.setattr(app, "generate_text_report", lambda data, report: "report")

    app.evaluate_sample(
        np.array([1.0, 2.0]),
        model_path="custom_model.joblib",
        n_bootstraps=0,
    )

    assert captured == {"model_path": "custom_model.joblib"}


def test_app_import_does_not_require_optional_plot_dependencies():
    from pysatl_expert.app import evaluate_sample

    assert callable(evaluate_sample)


def test_evaluate_sample_default_disables_bootstrap():
    import inspect

    from pysatl_expert.app import evaluate_sample

    assert inspect.signature(evaluate_sample).parameters["n_bootstraps"].default == 0


def test_evaluate_sample_requires_model_path():
    import inspect

    from pysatl_expert.app import evaluate_sample

    assert inspect.signature(evaluate_sample).parameters["model_path"].default is (
        inspect.Parameter.empty
    )


@pytest.mark.parametrize("pdf_mode", ["summary", "full"])
def test_evaluate_sample_forwards_pdf_mode(monkeypatch, pdf_mode):
    from pysatl_expert import app

    captured = {}

    class FakePipeline:
        def __init__(self, components):
            pass

        def identify_best(self, data, **kwargs):
            return Report("Normal", 0.8, {})

    def generate(data, report, **kwargs):
        captured.update(kwargs)
        return kwargs["output_path"]

    monkeypatch.setattr(app, "MLStrategy", lambda **kwargs: object())
    monkeypatch.setattr(app, "DistributionPipeline", FakePipeline)
    monkeypatch.setattr(app, "generate_text_report", lambda data, report: "report")
    monkeypatch.setattr(app, "generate_pdf_report", generate)
    app.evaluate_sample(
        np.array([1.0, 2.0]),
        model_path="model.joblib",
        save_pdf=True,
        pdf_mode=pdf_mode,
        pdf_path="report.pdf",
    )
    assert captured == {"output_path": "report.pdf", "mode": pdf_mode}


def test_evaluate_sample_forwards_configurable_input_bootstrap(monkeypatch):
    from pysatl_expert import app

    captured = {}

    class FakePipeline:
        def __init__(self, components):
            pass

        def identify_best(self, data, n_bootstraps, random_state):
            captured["n_bootstraps"] = n_bootstraps
            captured["random_state"] = random_state
            return Report("Normal", 0.8, {}, final_ranks={"Normal": 0.8})

    monkeypatch.setattr(app, "MLStrategy", lambda **kwargs: object())
    monkeypatch.setattr(app, "DistributionPipeline", FakePipeline)
    monkeypatch.setattr(app, "generate_text_report", lambda data, report: "report")

    app.evaluate_sample(
        np.array([1.0, 2.0]),
        model_path="model.joblib",
        save_plot=False,
        n_bootstraps=20,
        random_state=17,
    )

    assert captured == {"n_bootstraps": 20, "random_state": 17}


def test_beta_visualization_uses_fitted_alpha_and_beta_parameters():
    distribution = get_scipy_dist("Beta", {"alpha": 2.5, "beta": 4.0})

    assert distribution.kwds["a"] == 2.5
    assert distribution.kwds["b"] == 4.0


def test_build_pipeline_is_reusable_and_does_not_print(monkeypatch, capsys):
    from pysatl_expert import app

    loads = []

    def strategy(**kwargs):
        loads.append(kwargs)
        return object()

    class FakePipeline:
        def __init__(self, components):
            self.components = components

        def identify_best(self, data, **kwargs):
            return Report("Normal", 0.8, {})

    monkeypatch.setattr(app, "MLStrategy", strategy)
    monkeypatch.setattr(app, "DistributionPipeline", FakePipeline)
    pipeline = app.build_pipeline(model_path="trusted.joblib")
    for sample in ([1.0, 2.0], [3.0, 4.0]):
        assert pipeline.identify_best(np.asarray(sample)).distribution_name == "Normal"

    assert len(loads) == 1
    assert loads[0]["model_path"] == "trusted.joblib"
    assert capsys.readouterr().out == ""
    assert len(pipeline.components.distributions) == 8
