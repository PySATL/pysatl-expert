import numpy as np
import pytest

from pysatl_expert.models.feature_extractor import FeatureExtractor
from pysatl_expert.models.report import Report
from pysatl_expert.reporting.pdf import generate_pdf_report


@pytest.fixture
def sample_data():
    np.random.seed(42)
    return np.random.normal(loc=5.0, scale=2.0, size=150)


def _report(data: np.ndarray, *args, **kwargs) -> Report:
    """Build a report with the same statistics snapshot as the pipeline."""
    return Report(
        *args,
        sample_statistics=FeatureExtractor().calculate_sample_stats(data),
        **kwargs,
    )


def test_generate_pdf_report_creates_valid_pdf(sample_data, tmp_path):
    report = _report(
        sample_data,
        distribution_name="Normal",
        confidence=0.875,
        all_scores={"Normal": {"ad": 1.0, "ks": 1.0, "cvm": 1.0, "kurt": 0.0}},
        parameters={"loc": 5.02, "scale": 1.98},
        final_ranks={"Normal": 0.875, "Student": 0.125},
        confidence_kind="model_probability",
    )

    pdf_path = tmp_path / "test_report.pdf"
    result_path = generate_pdf_report(sample_data, report, output_path=pdf_path)

    assert result_path == pdf_path
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 5000

    with pdf_path.open("rb") as f:
        header = f.read(5)
    assert header == b"%PDF-"


def test_generate_pdf_report_handles_bootstrap_stability(sample_data, tmp_path):
    report = _report(
        sample_data,
        distribution_name="Exponential",
        confidence=0.92,
        all_scores={"Exponential": {"ks": 1.0, "ad": 1.0}},
        parameters={"loc": 0.0, "scale": 3.5},
        final_ranks={"Exponential": 0.92, "Gamma": 0.08},
        confidence_kind="bootstrap_stability",
    )

    pdf_path = tmp_path / "bootstrap_report.pdf"
    result_path = generate_pdf_report(sample_data, report, output_path=pdf_path)

    assert result_path.exists()
    assert result_path.stat().st_size > 5000


def test_generate_pdf_report_handles_empty_scores_and_none_parameters(sample_data, tmp_path):
    report = _report(
        sample_data,
        distribution_name="Uniform",
        confidence=0.50,
        all_scores={},
        parameters=None,
        final_ranks=None,
    )

    pdf_path = tmp_path / "minimal_report.pdf"
    result_path = generate_pdf_report(sample_data, report, output_path=pdf_path)

    assert result_path.exists()
    assert result_path.stat().st_size > 3000


def test_pdf_full_mode_adds_only_selected_feature_inputs(
    sample_data, tmp_path, monkeypatch
):
    from matplotlib.backends.backend_pdf import PdfPages

    pages = []
    original = PdfPages.savefig

    def capture(pdf, figure, **kwargs):
        pages.append([text.get_text() for text in figure.texts])
        return original(pdf, figure, **kwargs)

    monkeypatch.setattr(PdfPages, "savefig", capture)
    report = _report(
        sample_data,
        "Normal",
        0.8,
        {"Normal": {"ks": 0.1}},
        parameters={"loc": 5, "scale": 2},
        final_ranks={"Normal": 0.8},
    )
    generate_pdf_report(sample_data, report, tmp_path / "summary.pdf")
    summary = list(pages)
    assert len(summary) == 2
    pages.clear()
    generate_pdf_report(sample_data, report, tmp_path / "full.pdf", mode="full")
    assert len(pages) == 3
    assert pages[1] == summary[1]
    assert "Selected features" in pages[2][0]
    assert all("Raw criterion statistics" not in page[0] for page in pages)
    assert report.final_ranks == {"Normal": 0.8}


def test_pdf_rejects_unknown_mode_without_creating_output(sample_data, tmp_path):
    output = tmp_path / "invalid.pdf"
    with pytest.raises(ValueError, match="mode"):
        generate_pdf_report(sample_data, Report("Normal", 0.8, {}), output, mode="invalid")
    assert not output.exists()
