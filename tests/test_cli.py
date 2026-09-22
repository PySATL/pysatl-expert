import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def runner():
    import pysatl_expert.cli

    return importlib.reload(pysatl_expert.cli)


def test_import_does_not_execute_cli(runner):
    assert callable(runner.main)


def test_console_entrypoint_discards_report_and_returns_success(runner, monkeypatch):
    report = object()
    monkeypatch.setattr(runner, "main", lambda: report)

    assert runner.entrypoint() == 0


def test_cli_requires_explicit_model_bundle(runner):
    with pytest.raises(SystemExit):
        runner.build_parser().parse_args([])

    model_path = Path("custom-model.joblib")
    args = runner.build_parser().parse_args(["--model", str(model_path)])

    assert args.input == Path("sample.csv")
    assert args.model == model_path
    assert args.output == Path("output/report.pdf")
    assert args.pdf_mode == "summary"
    assert args.bootstraps == 0


def test_loads_explicit_single_column_file(runner, tmp_path):
    path = tmp_path / "sample.csv"
    path.write_text("value\n1.5\n2.5\n", encoding="utf-8")
    np.testing.assert_array_equal(runner.load_sample(path, skip_header_rows=1), [1.5, 2.5])


def test_rejects_multiple_columns(runner, tmp_path):
    path = tmp_path / "sample.csv"
    path.write_text("1,2\n3,4\n", encoding="utf-8")
    with pytest.raises(ValueError, match="one column"):
        runner.load_sample(path)


def test_rejects_existing_output_before_loading_model(runner, tmp_path):
    path = tmp_path / "report.pdf"
    path.write_bytes(b"existing report")
    with pytest.raises(FileExistsError, match="choose another --output path"):
        runner.main(["--model", "model.joblib", "--output", str(path)])
    assert path.read_bytes() == b"existing report"


def test_nonnumeric_input_is_not_silently_removed(runner, tmp_path):
    path = tmp_path / "sample.csv"
    path.write_text("1\nwrong\n2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="wrong"):
        runner.load_sample(path)


def test_main_forwards_settings_and_returns_result(runner, tmp_path, monkeypatch):
    sample = np.array([0.1, 0.2, 0.3])
    input_path = tmp_path / "sample.csv"
    np.savetxt(input_path, sample)
    output_path = tmp_path / "report.pdf"
    report = object()
    captured = {}

    class Pipeline:
        def identify_best(self, data, **kwargs):
            np.testing.assert_array_equal(data, sample)
            captured.update(kwargs)
            return report

    def build(**kwargs):
        captured.update(kwargs)
        return Pipeline()

    def pdf(data, result, **kwargs):
        assert result is report
        captured.update(kwargs)

    monkeypatch.setattr(runner, "build_pipeline", build)
    monkeypatch.setattr(runner, "generate_pdf_report", pdf)
    model_path = tmp_path / "model.joblib"
    assert runner.main(
        [
            "--input",
            str(input_path),
            "--model",
            str(model_path),
            "--output",
            str(output_path),
            "--bootstraps",
            "3",
            "--random-state",
            "17",
        ]
    ) is report
    assert captured == {
        "model_path": model_path,
        "n_bootstraps": 3,
        "random_state": 17,
        "output_path": output_path,
        "mode": "summary",
    }


def test_main_accepts_cli_paths_and_full_report_mode(runner, tmp_path, monkeypatch):
    sample_path = tmp_path / "input.csv"
    sample_path.write_text("1\n2\n3\n", encoding="utf-8")
    output_path = tmp_path / "report.pdf"
    model_path = tmp_path / "expert.joblib"
    captured = {}

    class Pipeline:
        def identify_best(self, data, **kwargs):
            np.testing.assert_array_equal(data, [1.0, 2.0, 3.0])
            captured.update(kwargs)
            return object()

    monkeypatch.setattr(
        runner,
        "build_pipeline",
        lambda **kwargs: captured.update(kwargs) or Pipeline(),
    )
    monkeypatch.setattr(
        runner,
        "generate_pdf_report",
        lambda data, report, **kwargs: captured.update(kwargs),
    )

    runner.main(
        [
            "--input",
            str(sample_path),
            "--model",
            str(model_path),
            "--output",
            str(output_path),
            "--pdf-mode",
            "full",
            "--bootstraps",
            "4",
            "--random-state",
            "17",
        ]
    )

    assert captured == {
        "model_path": model_path,
        "n_bootstraps": 4,
        "random_state": 17,
        "output_path": output_path,
        "mode": "full",
    }
