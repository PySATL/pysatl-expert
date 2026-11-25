import inspect
from types import SimpleNamespace

import numpy as np
import pytest


def test_recommend_ml_forwards_explicit_model_and_options(monkeypatch):
    from pysatl_expert import app

    marker = object()
    captured = {}

    class Pipeline:
        components = SimpleNamespace(method="primary")

        def analyze_catalog(self, sample):
            return None

        def recommend(self, sample, **kwargs):
            captured["sample"] = np.array(sample, copy=True)
            captured.update(kwargs)
            return marker

    monkeypatch.setattr(
        app, "build_pipeline", lambda **kwargs: captured.update(kwargs) or Pipeline()
    )
    procedures = {"custom": object()}
    result = app.recommend_ml(
        [0.3, 0.1, 0.2],
        model_path="trusted.joblib",
        procedures=procedures,
        allow_calibration=True,
        additional_statistics=("normal__ks",),
        n_bootstraps=0,
        random_state=17,
    )

    assert result is marker
    assert captured["model_path"] == "trusted.joblib"
    assert captured["additional_statistics"] == ("normal__ks",)
    assert set(captured["procedures"]) == set(procedures)
    assert captured["procedures"]["custom"] is procedures["custom"]
    assert captured["allow_calibration"] is True
    assert captured["n_bootstraps"] == 0
    assert captured["random_state"] == 17
    np.testing.assert_array_equal(captured["sample"], [0.3, 0.1, 0.2])


def test_build_pipeline_unions_model_features_with_additional_catalog_statistics(
    monkeypatch,
):
    from pysatl_expert import app

    class FakeMethod:
        required_features = {"normal__ks", "sample_size"}

    monkeypatch.setattr(app, "MLStrategy", lambda **_: FakeMethod())
    pipeline = app.build_pipeline(model_path="trusted.joblib", additional_statistics={"normal__ad"})
    assert pipeline.components.method.required_features == {"normal__ks", "sample_size"}
    assert pipeline.components.criterion_selector.feature_names == {
        "normal__ks",
        "normal__ad",
        "sample_size",
    }


def test_build_pipeline_rejects_unknown_additional_statistics(monkeypatch):
    from pysatl_expert import app

    monkeypatch.setattr(app, "MLStrategy", lambda **_: SimpleNamespace(required_features=set()))
    with pytest.raises(ValueError, match="Unknown catalog statistics"):
        app.build_pipeline(model_path="trusted.joblib", additional_statistics={"not_a_test"})


def test_evaluate_sample_delegates_and_renders_the_canonical_result(monkeypatch, capsys):
    from pysatl_expert import app

    result = object()
    captured = {}
    delegated = {}

    def recommend(data, **kwargs):
        delegated["data"] = data
        delegated.update(kwargs)
        return result

    monkeypatch.setattr(app, "recommend_ml", recommend)
    monkeypatch.setattr(app, "render_result_report", lambda value, options: "canonical report")
    monkeypatch.setattr(
        app,
        "generate_plot_report",
        lambda value, output_path, options: captured.update(plot=(value, output_path, options))
        or output_path,
    )
    monkeypatch.setattr(
        app,
        "write_result_pdf",
        lambda value, path, options: captured.update(pdf=(value, path, options)) or path,
    )

    returned = app.evaluate_sample(
        np.array([0.1, 0.2, 0.3]),
        model_path="trusted.joblib",
        save_plot=True,
        plot_path="plots/result.png",
        save_pdf=True,
        pdf_path="reports/result.pdf",
        n_bootstraps=4,
        random_state=12,
        pdf_mode="full",
    )

    assert returned is result
    np.testing.assert_array_equal(delegated["data"], [0.1, 0.2, 0.3])
    assert delegated["n_bootstraps"] == 4
    assert delegated["random_state"] == 12
    output = capsys.readouterr().out
    assert "canonical report" in output
    plot_value, plot_path, plot_options = captured["plot"]
    pdf_value, pdf_path, pdf_options = captured["pdf"]
    assert plot_value is result and pdf_value is result
    assert plot_path == "plots/result.png" and pdf_path == "reports/result.pdf"
    assert plot_options is pdf_options
    assert plot_options.detail == "full"


def test_evaluate_sample_keeps_existing_entrypoint_defaults():
    from pysatl_expert.app import evaluate_sample

    assert inspect.signature(evaluate_sample).parameters["n_bootstraps"].default == 0
    assert inspect.signature(evaluate_sample).parameters["model_path"].default is (
        inspect.Parameter.empty
    )
