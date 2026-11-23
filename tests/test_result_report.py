from collections.abc import Mapping
from copy import deepcopy

import numpy as np
import pytest
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties

from pysatl_expert.models.analysis_result import (
    AnalysisResult,
    ProcedureResult,
    RecommendationResult,
)
from pysatl_expert.reporting._pdf_layout import _draw_pdf_pages
from pysatl_expert.reporting._report_content import _ReportContent, _Section
from pysatl_expert.reporting.result_report import (
    ReportOptions,
    render_result_report,
    write_result_pdf,
)


def analysis(marker="primary"):
    return AnalysisResult(
        sample_stats={"n": 3, "mean": 0.25},
        parameters={marker: 1.5},
        statistics={"Beta": {"ks": 0.2}},
        failures={"fit:Gamma": "fit failed"},
        procedure_results={
            "procedure": ProcedureResult(
                statistic=0.4,
                alternative="greater",
                p_value=1e-12,
                alpha=0.05,
                source="adapter",
                hypothesis=(("shape", 2.0),),
                configuration={"mode": "small"},
                parameter_mode="fitted",
                calibration_method="external-calibrator",
                calibration_error="optional calibration unavailable",
                unavailable={"critical_values": "provider did not return bounds"},
            )
        },
    )


def test_options_are_immutable_and_explicit_sections_override_order():
    opts = ReportOptions(detail="summary", sections=["recommendation", "sample"], precision=2)
    result = RecommendationResult(analysis(), "Normal")
    text = render_result_report(result, opts)
    assert text.index("Recommendation\n") < text.index("Sample summary\n")
    assert "Fit parameters\n" not in text
    assert opts.sections == ("recommendation", "sample")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"detail": "brief"},
        {"sections": ["unknown"]},
        {"sections": ["sample", "sample"]},
        {"sections": "sample"},
        {"precision": -1},
        {"precision": True},
    ],
)
def test_invalid_options_are_rejected(kwargs):
    with pytest.raises(ValueError):
        ReportOptions(**kwargs)


def test_summary_keeps_failures_and_unavailable_status_and_small_p_value():
    summary = render_result_report(analysis())
    full = render_result_report(analysis(), ReportOptions(detail="full"))
    assert "1.0000e-12" in full
    assert "Diagnostics\n" not in summary
    assert "Analysis failures (analysis): fit:Gamma: fit failed" in summary
    assert "Unavailable procedure results (analysis): procedure.critical_values" in summary
    assert "Procedure calibration errors" in summary
    assert "Diagnostics\n" in full
    assert "Sample size | 3" in full
    assert "Sample size | 3.0000" not in full
    assert "Analysis failures (analysis): fit:Gamma: fit failed" in full
    assert "Unavailable procedure.critical_values" in full
    assert "Parameter mode: fitted" in full
    assert "Procedure calibration error" in full


def test_catalog_fact_is_reported_once_with_provenance():
    from pysatl_expert.app import analyze_sample

    result = analyze_sample(
        [0.1, 0.3, 0.8, 1.2, 1.7, 2.1],
        distributions=["Normal"],
        statistics=["normal__ad"],
    )
    text = render_result_report(result, ReportOptions(detail="full"))

    assert "Normal.ad" not in text
    rows = [line for line in text.splitlines() if "normal__ad" in line and "|" in line]
    assert len(rows) == 1
    assert "AD_NORMALITY_GOODNESS_OF_FIT" in text
    assert "Parameter mode: family" in text


def test_arbitrary_prediction_requires_explicit_formatter():
    class Secret:
        def __str__(self):
            raise AssertionError("must not call arbitrary __str__")

    result = RecommendationResult(analysis(), Secret())
    text = render_result_report(result)
    assert "unsupported type Secret" in text
    custom = render_result_report(
        result, ReportOptions(prediction_formatter=lambda obj: "custom answer")
    )
    assert "custom answer" in custom


def test_canonical_label_scores_render_as_recommendation_and_shared_class_score_table():
    scores = {
        "Beta": 0.025837551965687582,
        "Exponential": 0.00014732995853231298,
        "Gamma": 0.007349934888968124,
        "LogNormal": 0.00014435818885693655,
        "Normal": 3.3920800726341644e-05,
        "Student": 9.114768996378478e-06,
        "Uniform": 0.9589322594827658,
        "Weibull": 0.0075455299454668424,
    }
    result = RecommendationResult(analysis(), {"label": "Uniform", "scores": scores})

    text = render_result_report(result)

    assert "Answer: Uniform" in text
    header = next(
        line for line in text.splitlines() if line.strip().startswith("Class") and "|" in line
    )
    assert [cell.strip() for cell in header.split("|")] == ["Class", "Score"]
    assert "Beta" in text and "0.0258" in text
    assert "Uniform" in text and "0.9589" in text
    assert "Weibull" in text and "0.0075" in text
    assert "scores:" not in text

    custom = render_result_report(
        result, ReportOptions(prediction_formatter=lambda prediction: "custom scores")
    )
    assert "Answer: custom scores" in custom
    assert not any(line.strip().startswith("Class") and "|" in line for line in custom.splitlines())


def test_suppressed_recommendation_does_not_inspect_prediction_mapping():
    class UnreadablePrediction(Mapping):
        def __getitem__(self, key):
            raise AssertionError("suppressed prediction was inspected")

        def __iter__(self):
            raise AssertionError("suppressed prediction was inspected")

        def __len__(self):
            raise AssertionError("suppressed prediction was inspected")

    result = RecommendationResult(analysis(), UnreadablePrediction())

    text = render_result_report(result, ReportOptions(sections=[]))

    assert "Answer: available" in text
    assert "Recommendation\n" not in text


def test_procedure_table_heading_stays_with_multiple_rows_after_low_content(tmp_path, monkeypatch):
    captured_pages = []

    class CapturingPdf:
        def __init__(self, _path):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def savefig(self, figure):
            axis = figure.axes[0]
            texts = [item.get_text() for item in axis.texts]
            for table in axis.tables:
                texts.extend(cell.get_text().get_text() for cell in table.get_celld().values())
            captured_pages.append(texts)

    monkeypatch.setattr("pysatl_expert.reporting._pdf_layout.PdfPages", CapturingPdf)
    content = _ReportContent(
        title="Pagination regression",
        status_lines=(),
        sections=(
            _Section("provenance", "Methods", tuple(f"Stored note {index}" for index in range(21))),
            _Section(
                "procedures",
                "Procedure results",
                (),
                ("Procedure / statistic", "Statistic"),
                tuple((f"Normal.metric_{index}", "0.5000") for index in range(8)),
            ),
        ),
    )

    _draw_pdf_pages(tmp_path / "captured.pdf", content, FontProperties(family="DejaVu Sans"))

    procedure_pages = [
        page
        for page in captured_pages
        if "Procedure results" in page or "Procedure results (continued)" in page
    ]
    first_page = procedure_pages[0]
    procedure_rows = [text for text in first_page if text.startswith("Normal.metric_")]
    assert "Procedure / statistic" in first_page
    assert "Normal.metric_0" in procedure_rows
    assert len(procedure_rows) > 1


def test_standalone_recommendation_error_survives_section_override():
    result = RecommendationResult(analysis(), None, error="prediction could not be produced")
    text = render_result_report(result, ReportOptions(sections=[]))
    assert "Answer: error - prediction could not be produced" in text
    assert "Recommendation\n" not in text


def test_rendering_does_not_mutate_result():
    result = RecommendationResult(analysis(), {"label": np.int64(2), "scores": {np.int64(2): 0.8}})
    before = deepcopy(result)
    render_result_report(result, ReportOptions(detail="full"))
    assert result == before


def test_pdf_is_written_only_to_requested_path_and_contains_report_text(tmp_path):
    result = RecommendationResult(analysis(), "Normal")
    target = tmp_path / "nested" / "result.pdf"
    returned = write_result_pdf(
        result, target, ReportOptions(sections=["recommendation", "procedures"])
    )
    assert returned == target
    assert target.read_bytes().startswith(b"%PDF-")
    assert target.stat().st_size > 1000


def test_empty_result_sections_use_explicit_unavailable_label():
    result = AnalysisResult({}, {}, {}, {})
    text = render_result_report(result)
    assert "No data is recorded in sample summary for this result." in text


def test_analysis_summary_uses_only_stored_analysis_sections_and_procedure_values():
    result = AnalysisResult(
        sample_stats={"n": 24, "mean": 2.5},
        parameters={"Normal": {"loc": 2.5, "scale": 1.2}},
        statistics={},
        failures={},
        procedure_results={"shapiro": ProcedureResult(statistic=0.98, alternative="two-sided")},
    )
    text = render_result_report(result)
    assert "Sample size" in text and "Mean" in text
    assert "Fit parameters" in text and "Normal" in text and "scale" in text
    assert "Procedure results" in text and "shapiro" in text
    assert "Recommendation" not in text
    assert "p-value" not in text and "alpha" not in text and "Critical values" not in text
    assert "No data" not in text


def test_explicit_unavailable_sections_are_contextual_and_internal_strings_english():
    text = render_result_report(
        AnalysisResult({}, {}, {}, {}), ReportOptions(sections=["recommendation"])
    )
    assert "No recommendation data is stored for this result." in text
    assert not any(0x0400 <= ord(character) <= 0x04FF for character in text)


def test_default_sections_follow_canonical_order_and_skip_unavailable_charts():
    base = analysis()
    base.sample_snapshot = (0.0, 0.5, 1.0)
    result = RecommendationResult(base, "Normal")
    text = render_result_report(result)
    headings = [
        "Sample summary\n",
        "Recommendation\n",
        "Fit parameters\n",
        "Procedure results\n",
        "Distribution charts\n",
    ]
    positions = [text.index(heading) for heading in headings]
    assert positions == sorted(positions)
    assert "Distribution charts\n" in text

    missing_snapshot = AnalysisResult({}, {}, {}, {})
    missing_text = render_result_report(missing_snapshot)
    assert "Distribution charts\n" not in missing_text
    assert "Charts unavailable" not in missing_text
    explicit = render_result_report(missing_snapshot, ReportOptions(sections=["plots"]))
    assert "Charts unavailable: no sample snapshot" in explicit


def test_full_t_only_result_does_not_claim_missing_provenance_or_calibration():
    result = AnalysisResult(
        sample_stats={"sample_size": 12},
        parameters={},
        statistics={},
        failures={},
        procedure_results={"test": ProcedureResult(statistic=0.75)},
    )
    text = render_result_report(result, ReportOptions(detail="full"))
    assert "Procedure results" in text and "0.7500" in text
    assert "Methods and provenance" not in text
    assert "Diagnostics" not in text
    assert "Not available" not in text


def _rf_result(snapshot=(0.1, 0.4, 0.9)):
    stored = AnalysisResult(
        sample_stats={"sample_size": len(snapshot), "mean": 0.4667},
        parameters={"Normal": {"mu": 0.45, "std": 0.3}},
        statistics={"Normal": {"ks": 0.2}},
        failures={},
        sample_snapshot=snapshot,
    )
    prediction = {
        "label": "Normal",
        "scores": {"Normal": 0.8, "Gamma": 0.2},
        "evidence": {
            "stage1_scores": {"continuous": 0.9, "bounded": 0.1},
            "stage2_scores": {"continuous": {"Normal": 0.7, "Gamma": 0.3}},
            "stage1_features": {"sample_size": 3},
            "stage2_features": {"continuous": {"Normal__ks": 0.2}},
        },
        "artifact": {
            "identifier": "model.joblib",
            "sha256": "abc123",
            "feature_schema": {"sha256": "schema123", "names": ["sample_size", "Normal__ks"]},
        },
    }
    return RecommendationResult(stored, prediction)


def test_rf_four_key_prediction_renders_scores_evidence_and_artifact_without_duplicate_rank_table():
    result = _rf_result()
    summary = render_result_report(result)
    full = render_result_report(result, ReportOptions(detail="full"))

    assert "Answer: Normal" in summary
    assert any(
        line.strip().startswith("Class") and "Score" in line for line in summary.splitlines()
    )
    assert "Model evidence" in summary
    assert "Stage 1" in summary and "continuous" in summary and "0.9000" in summary
    assert "Artifact: model.joblib" in summary and "abc123" in summary
    assert "Feature schema columns: 2" in summary and "schema123" in summary
    assert "Normal__ks" not in summary
    assert "Normal__ks" in full and "0.2000" in full
    assert (
        sum(line.strip().startswith("Class") and "Score" in line for line in summary.splitlines())
        == 1
    )


def test_bootstrap_diagnostics_keep_counts_shares_and_errors_separate_from_primary_label():
    result = _rf_result()
    result.diagnostics["bootstrap"] = {
        "requested": 4,
        "successful": 3,
        "vote_counts": {"Normal": 2, "Gamma": 1},
        "vote_shares": {"Normal": 2 / 3, "Gamma": 1 / 3},
        "stability": 2 / 3,
        "failures": ["one resample failed"],
    }

    text = render_result_report(result)

    assert "Answer: Normal" in text
    assert "Bootstrap requested: 4" in text
    assert "Bootstrap successful: 3" in text
    assert "Bootstrap winner stability: 0.6667" in text
    assert any("Normal" in line and "2" in line and "0.6667" in line for line in text.splitlines())
    assert any("Gamma" in line and "1" in line and "0.3333" in line for line in text.splitlines())
    assert "one resample failed" in text


def test_plot_report_rejects_options_that_suppress_plots(tmp_path):
    from pysatl_expert.reporting.plot import generate_plot_report

    output = tmp_path / "suppressed.png"
    with pytest.raises(ValueError, match="plots"):
        generate_plot_report(_rf_result(), output, ReportOptions(sections=["recommendation"]))
    assert not output.exists()


def test_png_bypasses_text_collection_and_figure_reserves_footer_band(tmp_path, monkeypatch):
    from pysatl_expert.reporting import plot

    monkeypatch.setattr(
        "pysatl_expert.reporting.result_report._collect",
        lambda *_args: pytest.fail("PNG must not build text/PDF content"),
    )
    figure = plot._build_result_figure(_rf_result())
    try:
        assert all(axis.get_position().y0 >= 0.09 for axis in figure.axes)
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)


@pytest.mark.parametrize(
    "order",
    [
        ("plots", "sample"),
        ("recommendation", "plots", "sample"),
        ("sample", "plots"),
    ],
)
def test_pdf_chart_page_obeys_explicit_section_position(tmp_path, monkeypatch, order):
    saved_pages = []

    class CapturingPdf:
        def __init__(self, _path):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def savefig(self, figure):
            texts = [text.get_text() for text in figure.texts]
            for axis in figure.axes:
                texts.extend(item.get_text() for item in axis.texts)
                texts.extend([axis.get_title()])
            saved_pages.append(texts)

    monkeypatch.setattr("pysatl_expert.reporting._pdf_layout.PdfPages", CapturingPdf)
    sections = tuple(
        _Section(name, f"{name.upper()}_SECTION", (f"{name}_detail",)) for name in order
    )
    content = _ReportContent("Order regression", (), sections)
    chart = Figure()
    chart_axis = chart.add_subplot(111)
    chart_axis.set_title("CHART_PAGE_MARKER")

    _draw_pdf_pages(tmp_path / "ordered.pdf", content, FontProperties(family="DejaVu Sans"), chart)

    chart_page = next(
        index for index, texts in enumerate(saved_pages) if "CHART_PAGE_MARKER" in texts
    )
    for name in order:
        section_page = next(
            index for index, texts in enumerate(saved_pages) if f"{name.upper()}_SECTION" in texts
        )
        if name == "plots":
            assert section_page == chart_page
            assert "plots_detail" in saved_pages[chart_page]
        elif order.index(name) < order.index("plots"):
            assert section_page < chart_page
        else:
            assert section_page > chart_page


def test_pdf_does_not_build_or_write_chart_page_when_plots_are_suppressed(tmp_path, monkeypatch):
    from pysatl_expert.reporting import plot

    monkeypatch.setattr(
        plot, "_build_result_figure", lambda _result: pytest.fail("unexpected chart")
    )
    target = tmp_path / "no-charts.pdf"
    result = AnalysisResult({}, {}, {}, {})

    write_result_pdf(result, target, ReportOptions(sections=["sample"]))

    assert target.read_bytes().startswith(b"%PDF-")
