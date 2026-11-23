"""Reports for already computed result objects.

This module only reads result snapshots. It does not call a pipeline, model, fit,
statistical procedure, or calibration provider.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from matplotlib.font_manager import FontProperties

from pysatl_expert.models.analysis_result import (
    AnalysisResult,
    RecommendationResult,
)
from pysatl_expert.reporting import _pdf_layout
from pysatl_expert.reporting._report_content import _ReportContent, _Section


SECTION_NAMES = (
    "sample",
    "recommendation",
    "model_evidence",
    "bootstrap",
    "parameters",
    "procedures",
    "provenance",
    "diagnostics",
    "plots",
)
_FULL = SECTION_NAMES
_MISSING = "Not available"


@dataclass(frozen=True)
class ReportOptions:
    """Frozen presentation settings. Explicit sections override detail presets."""

    detail: str = "summary"
    sections: tuple[str, ...] | None = None
    precision: int = 4
    prediction_formatter: Callable[[Any], str] | None = None

    def __post_init__(self) -> None:
        if self.detail not in {"summary", "full"}:
            raise ValueError("detail must be 'summary' or 'full'")
        if (
            isinstance(self.precision, bool)
            or not isinstance(self.precision, int)
            or not 0 <= self.precision <= 12
        ):
            raise ValueError("precision must be an integer from 0 to 12")
        if self.sections is not None:
            if isinstance(self.sections, str):
                raise ValueError("sections must be an ordered sequence of section names")
            sections = tuple(self.sections)
            unknown = [
                section
                for section in sections
                if not isinstance(section, str) or section not in SECTION_NAMES
            ]
            if unknown:
                raise ValueError(f"unknown report section(s): {', '.join(map(str, unknown))}")
            if len(set(sections)) != len(sections):
                raise ValueError("sections cannot contain duplicates")
            object.__setattr__(self, "sections", sections)
        if self.prediction_formatter is not None and not callable(self.prediction_formatter):
            raise ValueError("prediction_formatter must be callable")


Result: TypeAlias = AnalysisResult | RecommendationResult


def _sample_label(name: str) -> str:
    labels = {
        "n": "Sample size",
        "sample_size": "Sample size",
        "mean": "Mean",
        "std": "Standard deviation",
        "standard_deviation": "Standard deviation",
        "var": "Variance",
        "variance": "Variance",
        "min": "Minimum",
        "max": "Maximum",
        "median": "Median",
        "skewness": "Skewness",
        "skew": "Skewness",
        "kurtosis": "Kurtosis",
        "iqr": "Interquartile range",
        "relative_iqr": "Relative interquartile range",
        "entropy": "Histogram entropy",
    }
    return labels.get(name.lower(), name)


def _to_scalar(value: Any) -> Any:
    """Convert supported NumPy scalar values without inspecting arbitrary objects."""
    if isinstance(value, np.generic):
        scalar = value.item()
        if isinstance(scalar, (str, bool, int, float)):
            return scalar
    return value


def _format_number(value: Real, precision: int) -> str:
    if isinstance(value, Integral):
        return str(int(value))
    number = float(value)
    if not isfinite(number):
        return "Not available (non-finite number)"
    threshold = 10 ** (-precision) if precision else 1
    if number != 0 and abs(number) < threshold:
        return f"{number:.{precision}e}"
    return f"{number:.{precision}f}"


def _format_value(value: Any, precision: int) -> str:
    value = _to_scalar(value)
    if value is None:
        return _MISSING
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, Real):
        return _format_number(value, precision)
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        if not value:
            return "Empty"
        return "; ".join(f"{key}: {_format_value(item, precision)}" for key, item in value.items())
    if isinstance(value, (tuple, list)):
        if not value:
            return "Empty"
        return ", ".join(_format_value(item, precision) for item in value)
    return f"Unsupported value ({type(value).__name__})"


def _is_scored_prediction(prediction: Mapping) -> bool:
    """Recognize both the generic label/scores mapping and the canonical RF form."""
    keys = set(prediction)
    accepted = ({"label", "scores"}, {"label", "scores", "evidence", "artifact"})
    return keys in accepted and isinstance(prediction.get("scores"), Mapping)


def _parse_scored_prediction(
    prediction: Mapping, precision: int
) -> tuple[str, tuple[tuple[str, str], ...]] | str | None:
    """Parse canonical label/scores mapping into formatted label and rows, or an error."""
    if not _is_scored_prediction(prediction):
        return None
    label = _to_scalar(prediction["label"])
    if not isinstance(label, (str, bool, int, float)):
        return None
    rows: list[tuple[str, str]] = []
    for key, score in prediction["scores"].items():
        scalar_key = _to_scalar(key)
        scalar_score = _to_scalar(score)
        if not isinstance(scalar_key, (str, bool, int, float)) or not isinstance(
            scalar_score, Real
        ):
            return "Answer not rendered: scores must contain numeric values and simple labels"
        rows.append((_format_value(scalar_key, precision), _format_value(scalar_score, precision)))
    return _format_value(label, precision), tuple(rows)


def _prediction_text(prediction: Any, options: ReportOptions) -> str:
    prediction = _to_scalar(prediction)
    if options.prediction_formatter is not None:
        try:
            formatted = options.prediction_formatter(prediction)
        except Exception as exc:
            return f"Answer not rendered: formatter failed ({type(exc).__name__})"
        if not isinstance(formatted, str):
            return "Answer not rendered: formatter must return a string"
        return formatted
    if prediction is None:
        return _MISSING
    if isinstance(prediction, (str, bool, int, float)):
        return _format_value(prediction, options.precision)
    if isinstance(prediction, Mapping):
        parsed = _parse_scored_prediction(prediction, options.precision)
        if isinstance(parsed, str):
            return parsed
        if parsed is not None:
            label_text, rows = parsed
            details = "; ".join(f"{k}={v}" for k, v in rows) if rows else "Empty"
            return f"Label: {label_text}; scores: {details}"
    return (
        f"Answer not rendered: unsupported type {type(prediction).__name__}; "
        "provide prediction_formatter"
    )


def _prediction_score_table(prediction: Any, options: ReportOptions):
    """Return shared score-table cells for the canonical label/scores mapping."""
    if options.prediction_formatter is not None or not isinstance(prediction, Mapping):
        return None
    parsed = _parse_scored_prediction(prediction, options.precision)
    if not isinstance(parsed, tuple):
        return None
    label_text, rows = parsed
    return label_text, ("Class", "Score"), rows


def _artifact_lines(artifact: Mapping, precision: int) -> list[str]:
    lines = []
    for key, title in (("identifier", "Artifact"), ("sha256", "Artifact SHA-256")):
        value = artifact.get(key)
        if value is not None:
            lines.append(f"{title}: {_format_value(value, precision)}")
    schema = artifact.get("feature_schema")
    if isinstance(schema, Mapping):
        digest = schema.get("sha256")
        if digest is not None:
            lines.append(f"Feature schema SHA-256: {_format_value(digest, precision)}")
        names = schema.get("names")
        if isinstance(names, (tuple, list)):
            lines.append(f"Feature schema columns: {len(names)}")
    return lines


def _score_evidence_rows(evidence: Mapping, precision: int) -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []
    family_scores = evidence.get("stage1_scores")
    conditional_scores = evidence.get("stage2_scores")
    if isinstance(family_scores, Mapping):
        rows.extend(
            ("Stage 1", str(family), "Family score", _format_value(score, precision))
            for family, score in family_scores.items()
        )
    if isinstance(conditional_scores, Mapping):
        for family, scores in conditional_scores.items():
            if isinstance(scores, Mapping):
                rows.extend(
                    ("Stage 2", str(family), str(candidate), _format_value(score, precision))
                    for candidate, score in scores.items()
                )
    return rows


def _selected_input_rows(evidence: Mapping, precision: int) -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []
    for key, stage in (("stage1_features", "Stage 1"), ("stage2_features", "Stage 2")):
        selected = evidence.get(key)
        if not isinstance(selected, Mapping):
            continue
        for group, features in selected.items():
            if isinstance(features, Mapping):
                rows.extend(
                    (f"{stage} input", str(group), str(name), _format_value(value, precision))
                    for name, value in features.items()
                )
            else:
                rows.append(
                    (f"{stage} input", "Selected", str(group), _format_value(features, precision))
                )
    return rows


def _model_evidence_content(prediction: Any, precision: int, detail: str):
    if not isinstance(prediction, Mapping):
        return [], (), ()
    evidence = prediction.get("evidence")
    artifact = prediction.get("artifact")
    if not isinstance(evidence, Mapping) and not isinstance(artifact, Mapping):
        return [], (), ()

    lines = _artifact_lines(artifact, precision) if isinstance(artifact, Mapping) else []
    rows: list[tuple[str, str, str, str]] = []
    if isinstance(evidence, Mapping):
        rows.extend(_score_evidence_rows(evidence, precision))
        if detail == "full":
            rows.extend(_selected_input_rows(evidence, precision))
    columns = ("Stage", "Family", "Candidate / feature", "Score / raw input") if rows else ()
    return lines, columns, tuple(rows)


def _bootstrap_content(recommendation: RecommendationResult, precision: int):
    diagnostics = recommendation.diagnostics
    bootstrap = diagnostics.get("bootstrap") if isinstance(diagnostics, Mapping) else None
    if not isinstance(bootstrap, Mapping) or not bootstrap.get("requested"):
        return [], (), ()
    requested = bootstrap.get("requested")
    successful = bootstrap.get("successful", 0)
    stability = bootstrap.get("stability")
    lines = [
        f"Bootstrap requested: {_format_value(requested, precision)}",
        f"Bootstrap successful: {_format_value(successful, precision)}",
        f"Bootstrap winner stability: {_format_value(stability, precision)}",
    ]
    failures = bootstrap.get("failures", ())
    if isinstance(failures, (tuple, list)):
        lines.extend(f"Bootstrap evaluation failure: {failure}" for failure in failures)
    vote_counts = bootstrap.get("vote_counts")
    vote_shares = bootstrap.get("vote_shares")
    rows = []
    if isinstance(vote_counts, Mapping) or isinstance(vote_shares, Mapping):
        labels = dict.fromkeys([*(vote_counts or {}), *(vote_shares or {})])
        for label in labels:
            count = vote_counts.get(label) if isinstance(vote_counts, Mapping) else None
            share = vote_shares.get(label) if isinstance(vote_shares, Mapping) else None
            rows.append(
                (str(label), _format_value(count, precision), _format_value(share, precision))
            )
    columns = ("Candidate", "Top-1 votes", "Top-1 share") if rows else ()
    return lines, columns, tuple(rows)


def _provenance_lines(analysis: AnalysisResult, precision: int) -> list[str]:
    lines: list[str] = []
    for statistic_id, result in analysis.procedure_results.items():
        if result.source:
            lines.append(f"  {statistic_id}: source={result.source}")
        if result.calibration_method is not None:
            lines.append(f"    Procedure calibration method: {result.calibration_method}")
        metadata = (
            ("Alternative", result.alternative),
            (
                "Hypothesis",
                _format_value(result.hypothesis, precision) if result.hypothesis else None,
            ),
            ("Hypothesis conditions", result.hypothesis_description),
            ("Parameterization", result.parameterization),
            ("Parameter mode", result.parameter_mode),
            (
                "Preprocessing",
                _format_value(result.preprocessing, precision) if result.preprocessing else None,
            ),
            (
                "Configuration",
                _format_value(result.configuration, precision) if result.configuration else None,
            ),
        )
        lines.extend(f"    {label}: {value}" for label, value in metadata if value)
        calibrated = []
        if result.p_value is not None:
            calibrated.append(f"p-value={_format_value(result.p_value, precision)}")
        if result.critical_values is not None:
            calibrated.append(f"critical values={_format_value(result.critical_values, precision)}")
            calibrated.append(f"alpha={_format_value(result.alpha, precision)}")
        if calibrated:
            lines.append(f"    Calibration result: {'; '.join(calibrated)}")
    return lines


def _diagnostic_lines(analysis: AnalysisResult) -> list[str]:
    lines = [f"  Failure {key}: {reason}" for key, reason in analysis.failures.items()]
    for key, result in analysis.procedure_results.items():
        lines.extend(
            f"  Unavailable {key}.{field}: {reason}" for field, reason in result.unavailable.items()
        )
        if result.calibration_error:
            lines.append(f"  Procedure calibration error {key}: {result.calibration_error}")
    return lines


def _analysis_lines(
    analysis: AnalysisResult, section: str, label: str, precision: int
) -> list[str]:
    body = (
        _provenance_lines(analysis, precision)
        if section == "provenance"
        else _diagnostic_lines(analysis)
    )
    if not body:
        return []
    heading = "Analysis" if label == "analysis" else f"Analysis: {label}"
    return [heading, *body]


def _empty_values() -> dict[str, list[str]]:
    return {name: [] for name in SECTION_NAMES}


def _recommendation_content(
    recommendation: RecommendationResult,
    options: ReportOptions,
    include_details: bool = True,
) -> tuple[list[str], list[str], tuple[str, ...], tuple[tuple[str, ...], ...]]:
    score_table = (
        _prediction_score_table(recommendation.prediction, options) if include_details else None
    )
    if include_details:
        if score_table is None:
            answer = _prediction_text(recommendation.prediction, options)
        else:
            answer, _, _ = score_table
        detail = [f"Answer: {answer}"]
    else:
        detail = []
    if recommendation.error:
        detail.append(f"Recommendation error: {recommendation.error}")
        state = f"Answer: error - {recommendation.error}"
    else:
        availability = "available" if recommendation.prediction is not None else _MISSING
        state = f"Answer: {availability}"
    columns, rows = (score_table[1], score_table[2]) if score_table is not None else ((), ())
    return detail, [state], columns, rows


def _recommendation_extras(
    recommendation: RecommendationResult,
    options: ReportOptions,
    values: dict[str, list[str]],
    tables: dict[str, tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]],
) -> None:
    if options.sections is None or "model_evidence" in options.sections:
        prediction = recommendation.prediction
        # Avoid inspecting arbitrary Mapping implementations when the caller has
        # suppressed evidence output; the canonical RF predictor returns a dict.
        if isinstance(prediction, dict):
            evidence_lines, evidence_columns, evidence_rows = _model_evidence_content(
                prediction, options.precision, options.detail
            )
            values["model_evidence"].extend(evidence_lines)
            if evidence_columns:
                tables["model_evidence"] = evidence_columns, evidence_rows
    bootstrap_lines, bootstrap_columns, bootstrap_rows = _bootstrap_content(
        recommendation, options.precision
    )
    values["bootstrap"].extend(bootstrap_lines)
    if bootstrap_columns:
        tables["bootstrap"] = bootstrap_columns, bootstrap_rows


def _section_is_selected(options: ReportOptions, section: str) -> bool:
    chosen = options.sections
    if chosen is None:
        chosen = _FULL if options.detail == "full" else ("recommendation",)
    return section in chosen


def _result_parts(
    result: Result, options: ReportOptions
) -> tuple[
    str,
    list[str],
    dict[str, list[str]],
    tuple[str, AnalysisResult],
    dict[str, tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]],
]:
    values = _empty_values()
    tables: dict[str, tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]] = {}
    if isinstance(result, AnalysisResult):
        return "Statistical analysis report", [], values, ("analysis", result), tables
    if isinstance(result, RecommendationResult):
        _recommendation_extras(result, options, values, tables)
        include_recommendation = _section_is_selected(options, "recommendation")
        detail, status, columns, rows = _recommendation_content(
            result, options, include_details=include_recommendation
        )
        values["recommendation"].extend(detail)
        if columns:
            tables["recommendation"] = columns, rows
        return (
            "Recommendation report",
            status,
            values,
            ("primary answer", result.analysis),
            tables,
        )
    raise TypeError("result must be AnalysisResult or RecommendationResult")


def _analysis_status_lines(label: str, analysis: AnalysisResult) -> list[str]:
    status: list[str] = []
    if analysis.failures:
        status.append(
            f"Analysis failures ({label}): "
            + "; ".join(f"{key}: {reason}" for key, reason in analysis.failures.items())
        )
    unavailable = [
        f"{key}.{field}: {reason}"
        for key, procedure in analysis.procedure_results.items()
        for field, reason in procedure.unavailable.items()
    ]
    if unavailable:
        status.append(f"Unavailable procedure results ({label}): " + "; ".join(unavailable))
    calibration_errors = [
        f"{key}: {procedure.calibration_error}"
        for key, procedure in analysis.procedure_results.items()
        if procedure.calibration_error
    ]
    if calibration_errors:
        status.append(f"Procedure calibration errors ({label}): " + "; ".join(calibration_errors))
    return status


def _table_for(name: str, analysis: AnalysisResult, precision: int):
    if name == "sample":
        sample_rows: list[tuple[str, ...]] = [
            (_sample_label(str(key)), _format_value(value, precision))
            for key, value in analysis.sample_stats.items()
        ]
        return ("Measure", "Value"), sample_rows
    if name == "parameters":
        param_rows: list[tuple[str, ...]] = []
        for distribution, values in analysis.parameters.items():
            if isinstance(values, Mapping):
                param_rows.extend(
                    (str(distribution), str(key), _format_value(value, precision))
                    for key, value in values.items()
                )
            else:
                param_rows.append((str(distribution), "Estimate", _format_value(values, precision)))
        return ("Distribution", "Parameter", "Estimate"), param_rows
    if name == "procedures":
        return _procedure_table(analysis, precision)
    return (), []


def _procedure_table(analysis: AnalysisResult, precision: int):
    procedures = list(analysis.procedure_results.values())
    has_p = any(proc.p_value is not None for proc in procedures)
    has_critical = any(proc.critical_values is not None for proc in procedures)
    has_alpha = any(proc.alpha is not None for proc in procedures)
    columns: tuple[str, ...] = ("Procedure / statistic", "Statistic")
    columns += ("p-value",) if has_p else ()
    columns += ("Critical values",) if has_critical else ()
    columns += ("alpha",) if has_alpha else ()
    rows: list[tuple[str, ...]] = [
        (f"{stat_id}.{key}", _format_value(value, precision))
        for stat_id, stats in analysis.statistics.items()
        for key, value in stats.items()
        if key not in analysis.catalog_ids.get(stat_id, {})
    ]
    rows.extend(
        _procedure_row(proc_id, proc, has_p, has_critical, has_alpha, precision)
        for proc_id, proc in analysis.procedure_results.items()
    )
    normalized = [row + ("",) * (len(columns) - len(row)) for row in rows]
    return columns, normalized


def _procedure_row(proc_id, procedure, has_p, has_critical, has_alpha, precision):
    row = [proc_id, _format_value(procedure.statistic, precision)]
    if has_p:
        row.append(
            _format_value(procedure.p_value, precision) if procedure.p_value is not None else ""
        )
    if has_critical:
        row.append(
            _format_value(procedure.critical_values, precision)
            if procedure.critical_values is not None
            else ""
        )
    if has_alpha:
        row.append(_format_value(procedure.alpha, precision) if procedure.alpha is not None else "")
    return tuple(row)


_SECTION_TITLES = {
    "sample": "Sample summary",
    "recommendation": "Recommendation",
    "model_evidence": "Model evidence",
    "bootstrap": "Bootstrap diagnostics",
    "parameters": "Fit parameters",
    "procedures": "Procedure results",
    "provenance": "Methods and provenance",
    "diagnostics": "Diagnostics",
    "plots": "Distribution charts",
}


def _empty_section_lines(name: str, title: str):
    messages = {
        "recommendation": "No recommendation data is stored for this result.",
        "provenance": "No method provenance is recorded for this result.",
        "diagnostics": "No diagnostic messages are recorded for this result.",
        "plots": "Distribution charts use only the sample snapshot stored in the result.",
        "model_evidence": "No model evidence is stored for this result.",
        "bootstrap": "No bootstrap diagnostics were requested.",
    }
    if name in messages:
        return (messages[name],), name in {"model_evidence", "bootstrap"}
    return (f"No data is recorded in {title.lower()} for this result.",), True


def _selected_sections(
    values: dict[str, list[str]],
    options: ReportOptions,
    defaults: tuple[str, ...],
    analysis: AnalysisResult,
    tables: dict[str, tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]],
) -> tuple[_Section, ...]:
    chosen = options.sections if options.sections is not None else defaults
    sections = []
    for name in chosen:
        if name in tables:
            columns, rows = tables[name]
        else:
            columns, rows = _table_for(name, analysis, options.precision)
        section_lines = tuple(values[name])
        if not section_lines and not rows:
            section_lines, clear_columns = _empty_section_lines(name, _SECTION_TITLES[name])
            if clear_columns:
                columns = ()
        sections.append(_Section(name, _SECTION_TITLES[name], section_lines, columns, tuple(rows)))
    return tuple(sections)


def _default_sections(
    result: Result,
    analysis: AnalysisResult,
    values: dict[str, list[str]],
    options: ReportOptions,
) -> list[str]:
    defaults = ["sample"]
    if analysis.sample_snapshot is not None:
        defaults.append("plots")
    if analysis.parameters:
        defaults.append("parameters")
    if analysis.statistics or analysis.procedure_results:
        defaults.append("procedures")
    if isinstance(result, RecommendationResult):
        defaults.append("recommendation")
        prediction = result.prediction
        if isinstance(prediction, dict) and (
            isinstance(prediction.get("evidence"), Mapping)
            or isinstance(prediction.get("artifact"), Mapping)
        ):
            defaults.append("model_evidence")
        bootstrap = result.diagnostics.get("bootstrap")
        if isinstance(bootstrap, Mapping) and bootstrap.get("requested"):
            defaults.append("bootstrap")
    if options.detail == "full":
        if values["provenance"]:
            defaults.append("provenance")
        if values["diagnostics"]:
            defaults.append("diagnostics")
    return [name for name in SECTION_NAMES if name in defaults]


def _collect(result: Result, options: ReportOptions) -> _ReportContent:
    if not isinstance(result, (AnalysisResult, RecommendationResult)):
        raise TypeError("result must be one of the supported result models")
    title, status, values, (label, analysis), tables = _result_parts(result, options)
    for section in ("provenance", "diagnostics"):
        values[section].extend(_analysis_lines(analysis, section, label, options.precision))
    status.extend(_analysis_status_lines(label, analysis))
    defaults = _default_sections(result, analysis, values, options)
    if analysis.sample_snapshot is not None:
        values["plots"].append("Charts are generated from the stored sample snapshot.")
    else:
        values["plots"].append("Charts unavailable: no sample snapshot is stored in this result.")
    sections = _selected_sections(values, options, tuple(defaults), analysis, tables)
    return _ReportContent(
        title, tuple(status or ["Analysis completed; no recorded errors."]), sections
    )


def render_result_report(result: Result, options: ReportOptions | None = None) -> str:
    """Render a text report from an existing result without recomputation."""
    settings = options or ReportOptions()
    content = _collect(result, settings)
    lines = [content.title, "=" * len(content.title), "Status"]
    lines.extend(f"  {line}" for line in content.status_lines)
    for section in content.sections:
        lines.extend(["", section.title, "-" * len(section.title)])
        lines.extend(section.lines)
        if section.columns:
            widths = [
                max([len(cell), *(len(row[index]) for row in section.rows)])
                for index, cell in enumerate(section.columns)
            ]
            lines.append(
                " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(section.columns))
            )
            lines.append("-+-".join("-" * width for width in widths))
            lines.extend(
                " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(row))
                for row in section.rows
            )
    return "\n".join(lines)


def write_result_pdf(
    result: Result, path: str | Path, options: ReportOptions | None = None
) -> Path:
    """Write a paginated PDF at the explicit path and return that path."""
    content = _collect(result, options or ReportOptions())
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    font = FontProperties(family="DejaVu Sans")
    chart_figure = None
    if any(section.name == "plots" for section in content.sections):
        from pysatl_expert.reporting.plot import _build_result_figure

        chart_figure = _build_result_figure(result)
    try:
        _pdf_layout._draw_pdf_pages(output, content, font, chart_figure)
    finally:
        if chart_figure is not None:
            from matplotlib import pyplot as plt

            plt.close(chart_figure)
    return output
