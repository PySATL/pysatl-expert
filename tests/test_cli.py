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


def test_console_entrypoint_discards_result_and_returns_success(runner, monkeypatch):
    result = object()
    monkeypatch.setattr(runner, "main", lambda: result)

    assert runner.entrypoint() == 0


def test_console_entrypoint_reports_incompatible_model_without_traceback(
    runner, monkeypatch, capsys
):
    message = "Criterion source revision is not validated for this model"

    def reject():
        raise runner.ModelCompatibilityError(message)

    monkeypatch.setattr(runner, "main", reject)

    assert runner.entrypoint() == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == f"error: {message}\n"


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
    assert args.random_state == 42


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


def test_main_passes_canonical_result_and_settings_to_pdf_writer(runner, tmp_path, monkeypatch):
    sample = np.array([0.1, 0.2, 0.3])
    input_path = tmp_path / "sample.csv"
    np.savetxt(input_path, sample)
    output_path = tmp_path / "nested" / "report.pdf"
    result = object()
    captured = {}

    class Pipeline:
        def analyze_catalog(self, values):
            return None

        def recommend(self, data, **kwargs):
            np.testing.assert_array_equal(data, sample)
            captured["recommend"] = kwargs
            return result

    def build(**kwargs):
        captured["build"] = kwargs
        return Pipeline()

    def write(value, path, options):
        captured["write"] = (value, path, options)
        return path

    monkeypatch.setattr(runner, "build_pipeline", build)
    monkeypatch.setattr(runner, "write_result_pdf", write)
    model_path = tmp_path / "model.joblib"
    returned = runner.main(
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
            "--pdf-mode",
            "full",
        ]
    )

    assert returned is result
    assert captured["build"] == {"model_path": model_path}
    options_passed = captured["recommend"]
    assert "method" not in options_passed
    assert options_passed["n_bootstraps"] == 3
    assert options_passed["random_state"] == 17
    assert options_passed["analysis_provider"].__name__ == "analyze_catalog"
    written_result, written_path, options = captured["write"]
    assert written_result is result
    assert written_path == output_path
    assert options.detail == "full"
    assert output_path.parent.is_dir()
