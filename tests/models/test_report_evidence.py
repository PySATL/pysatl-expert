import numpy as np

from pysatl_expert.models.feature_extractor import FeatureExtractor
from pysatl_expert.models.feature_vector import FeatureVector
from pysatl_expert.models.hierarchical_model import HierarchicalExpertModel
from pysatl_expert.models.report import Report
from pysatl_expert.reporting.formatting import format_raw_statistic
from pysatl_expert.reporting.pdf import (
    _bootstrap_comparison,
    _fitted_candidate_curves,
)
from pysatl_expert.strategy.ml_strategy import MLStrategy


def test_evidence_fields_are_optional_and_independent():
    report = Report("Normal", 0.8, {})
    other = Report("Normal", 0.8, {})
    report.stage1_scores["Symmetric"] = 0.8
    assert other.stage1_scores == {}
    assert report.stage2_scores == {}
    assert report.candidate_parameters == {}


def test_bootstrap_comparison_is_sorted_by_original_scores():
    report = Report(
        "Normal",
        0.75,
        {},
        final_ranks={"Student": 0.25, "Normal": 0.75},
        bootstrap_ranks={"Student": 2 / 3, "Normal": 1 / 3},
        bootstrap_requested=4,
        bootstrap_successful=3,
    )
    rows, status = _bootstrap_comparison(report)
    assert rows == [["Normal", "75.00%", "33.33%"], ["Student", "25.00%", "66.67%"]]
    assert status == "Successful repeats: 3 of 4"


def test_bootstrap_comparison_distinguishes_disabled_and_failed():
    report = Report("Normal", 0.75, {}, final_ranks={"Normal": 0.75}, bootstrap_requested=0)
    assert _bootstrap_comparison(report)[1] == "Not performed"
    report.bootstrap_requested = 20
    rows, status = _bootstrap_comparison(report)
    assert status == "Successful repeats: 0 of 20; unavailable"
    assert rows == [["Normal", "75.00%", "—"]]


def test_hierarchy_and_parameters_share_page_with_neutral_footer(tmp_path, monkeypatch):
    from matplotlib.backends.backend_pdf import PdfPages

    from pysatl_expert.reporting.pdf import generate_pdf_report

    data = np.random.default_rng(42).normal(5, 2, 100)
    pages = []
    original = PdfPages.savefig

    def capture(pdf, figure, **kwargs):
        texts = [text.get_text() for text in figure.texts]
        for ax in figure.axes:
            for table in ax.tables:
                texts.extend(cell.get_text().get_text() for cell in table.get_celld().values())
        pages.append("\n".join(texts))
        return original(pdf, figure, **kwargs)

    monkeypatch.setattr(PdfPages, "savefig", capture)
    report = Report(
        "Normal",
        0.8,
        {},
        parameters={"loc": 5, "scale": 2},
        final_ranks={"Normal": 0.8, "Student": 0.2},
        stage1_scores={"Symmetric": 1.0},
        stage2_scores={"Symmetric": {"Normal": 0.8, "Student": 0.2}},
        sample_statistics=FeatureExtractor().calculate_sample_stats(data),
    )
    generate_pdf_report(data, report, tmp_path / "combined.pdf")
    assert "How the hierarchy" in pages[1]
    assert "Fitted parameters" in pages[1]
    assert "loc=5" in pages[1]
    text = "\n".join(pages)
    assert "Statistical Expert System" not in text
    assert "not hypothesis-test" not in text
    assert "no hypothesis decisions" not in text


def test_raw_values_are_not_decisions_and_missing_has_dash():
    assert [format_raw_statistic(v) for v in [0, 1, -1, np.nan, np.inf]] == [
        "0",
        "1",
        "-1",
        "—",
        "—",
    ]


def test_candidate_curves_do_not_invent_missing_parameters():
    report = Report(
        "Uniform",
        0.8,
        {},
        final_ranks={"Uniform": 0.8, "Beta": 0.2},
        candidate_parameters={"Uniform": {"a": 0, "b": 1}},
    )
    curves = _fitted_candidate_curves(report, None)
    assert [name for name, _ in curves] == ["Uniform"]


def test_hierarchy_evidence_matches_actual_forests():
    import pandas as pd

    rng = np.random.default_rng(12)
    frame = pd.DataFrame(
        rng.normal(size=(40, len(FeatureVector.FEATURE_NAMES))), columns=FeatureVector.FEATURE_NAMES
    )
    targets = pd.Series(["Normal", "Student", "Beta", "Uniform"] * 10)
    model = HierarchicalExpertModel(
        {"Symmetric": ["Normal", "Student"], "Bounded": ["Beta", "Uniform"]}
    )
    model.fit(frame, targets, n_stage1=3, n_stage2=2, n_estimators=3, n_jobs=1)
    strategy = object.__new__(MLStrategy)
    strategy.model = model
    strategy._class_names = model.classes_.tolist()
    values = frame.iloc[0].to_dict()
    candidate_scores = {}
    for dist, code in FeatureVector.CRITERIA_SCHEMA:
        candidate_scores.setdefault(dist, {})[code] = values[f"{dist}__{code}"]
    fv = FeatureVector({key: values[key] for key in FeatureVector.STAT_KEYS}, candidate_scores)
    report = strategy.predict_report(fv)
    assert report.stage1_features == frame[model.stage1_features].iloc[0].to_dict()
    for family, scores in report.stage2_scores.items():
        selected = model.stage2_features[family]
        expected = model.stage2_models[family].predict_proba(frame[selected].iloc[:1])[0]
        assert list(scores.values()) == list(expected)
        for name, conditional in scores.items():
            assert report.model_ranks[name] == report.stage1_scores[family] * conditional
