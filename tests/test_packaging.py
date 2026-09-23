from pathlib import Path


try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib


def _pyproject() -> dict:
    path = Path(__file__).parents[1] / "pyproject.toml"
    return tomllib.loads(path.read_text(encoding="utf-8"))


def test_runtime_criterion_dependency_is_portable() -> None:
    poetry = _pyproject()["tool"]["poetry"]

    assert poetry["dependencies"]["pysatl-criterion"] == {
        "git": "https://github.com/PySATL/pysatl-criterion.git",
        "branch": "main",
    }
    assert "pysatl-criterion" not in poetry["group"]["dev"]["dependencies"]


def test_installed_package_exposes_analysis_cli() -> None:
    poetry = _pyproject()["tool"]["poetry"]

    assert poetry["scripts"]["pysatl-expert"] == "pysatl_expert.cli:entrypoint"


def test_production_wheel_excludes_training_only_files() -> None:
    poetry = _pyproject()["tool"]["poetry"]

    assert poetry["exclude"] == [
        "pysatl_expert/config",
        "pysatl_expert/models/feature_selection_contract.py",
    ]


def test_runtime_dependencies_do_not_include_unused_reporting_tools() -> None:
    dependencies = _pyproject()["tool"]["poetry"]["dependencies"]

    assert "seaborn" in dependencies
    assert "tqdm" not in dependencies
