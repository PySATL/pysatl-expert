"""Charts rendered only from stored canonical result snapshots."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from scipy import stats

from pysatl_expert.models.analysis_result import AnalysisResult, RecommendationResult
from pysatl_expert.reporting.common import get_scipy_dist


logger = logging.getLogger(__name__)


def _primary(result):
    if isinstance(result, RecommendationResult):
        return result.analysis, result.prediction
    if isinstance(result, AnalysisResult):
        return result, None
    raise TypeError("result must be an AnalysisResult or RecommendationResult")


def _candidate(prediction: Any) -> str | None:
    if isinstance(prediction, Mapping):
        prediction = prediction.get("label")
    return prediction if isinstance(prediction, str) else None


def _scores(prediction: Any) -> Mapping | None:
    if isinstance(prediction, Mapping) and isinstance(prediction.get("scores"), Mapping):
        return prediction["scores"]
    return None


def _unavailable_figure(fig: Figure, axes, message: str) -> Figure:
    for axis in axes.flat:
        axis.axis("off")
    fig.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)
    return fig


def _draw_density(ax, data, x_grid, distribution, candidate_label):
    ax.hist(data, bins="auto", density=True, color="#bfdbfe", edgecolor="#1d4ed8", alpha=0.65)
    ax.set_title("Sample density and fitted PDF")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    if distribution is None:
        ax.text(0.5, 0.5, "Fitted PDF unavailable", transform=ax.transAxes, ha="center")
        return
    try:
        ax.plot(x_grid, distribution.pdf(x_grid), color="#b91c1c", lw=2, label=candidate_label)
        ax.legend(loc="best")
    except Exception as exc:
        logger.warning("Could not draw fitted PDF from stored parameters: %s", exc)
        ax.text(0.5, 0.5, "Fitted PDF unavailable", transform=ax.transAxes, ha="center")


def _draw_cdf(ax, sorted_data, x_grid, distribution, candidate_label):
    ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    ax.step(sorted_data, ecdf, where="post", color="#1d4ed8", label="Empirical CDF")
    ax.set_title("Empirical and fitted CDF")
    ax.set_xlabel("Value")
    ax.set_ylabel("Cumulative probability")
    if distribution is None:
        ax.text(0.5, 0.5, "Fitted CDF unavailable", transform=ax.transAxes, ha="center")
        return
    try:
        ax.plot(x_grid, distribution.cdf(x_grid), "--", color="#b91c1c", label=candidate_label)
        ax.legend(loc="best")
    except Exception as exc:
        logger.warning("Could not draw fitted CDF from stored parameters: %s", exc)
        ax.text(0.5, 0.5, "Fitted CDF unavailable", transform=ax.transAxes, ha="center")


def _draw_qq(ax, data, distribution, candidate):
    ax.set_title(f"Q-Q plot: {candidate or 'candidate unavailable'}")
    if distribution is None:
        ax.text(
            0.5,
            0.5,
            "Q-Q plot unavailable: fitted parameters are not stored",
            transform=ax.transAxes,
            ha="center",
        )
        ax.set_axis_off()
        return
    try:
        (theoretical, observed), (slope, intercept, correlation) = stats.probplot(
            data, dist=distribution, plot=None
        )
        ax.scatter(theoretical, observed, s=20, alpha=0.6, color="#1d4ed8")
        line_x = np.asarray([np.min(theoretical), np.max(theoretical)])
        ax.plot(line_x, slope * line_x + intercept, "--", color="#b91c1c")
        ax.set_xlabel("Theoretical quantiles")
        ax.set_ylabel("Sample quantiles")
        ax.text(0.03, 0.97, f"r²={correlation**2:.3f}", va="top", transform=ax.transAxes)
    except Exception as exc:
        logger.warning("Could not draw Q-Q chart from stored parameters: %s", exc)
        ax.text(0.5, 0.5, "Q-Q plot unavailable", transform=ax.transAxes, ha="center")


def _draw_scores(ax, prediction, candidate):
    scores = _scores(prediction)
    ax.set_title("Stored model scores" if scores else "Model ranking unavailable")
    items = []
    if scores:
        for label, score in scores.items():
            try:
                numeric = float(score)
            except (TypeError, ValueError):
                continue
            if np.isfinite(numeric):
                items.append((str(label), numeric))
    items.sort(key=lambda item: item[1])
    if not items:
        message = "No model scores are stored" if not scores else "No finite scores are stored"
        ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center")
        return
    labels = [label for label, _ in items]
    values = [value for _, value in items]
    colors = ["#1d4ed8" if label == candidate else "#cbd5e1" for label in labels]
    ax.barh(labels, values, color=colors, edgecolor="#94a3b8", linewidth=0.5)
    ax.set_xlabel("Model score (not probability of truth)")


def _build_result_figure(result) -> Figure:
    """Build the shared four-panel fit/evidence chart without recomputation."""
    analysis, prediction = _primary(result)
    candidate = _candidate(prediction)
    if candidate is None and len(analysis.parameters) == 1:
        candidate = next(iter(analysis.parameters))
    params = analysis.parameters.get(candidate) if candidate is not None else None
    if not isinstance(params, Mapping):
        params = None

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=180)
    fig.suptitle("PySATL Expert · Stored result diagnostics", fontsize=15, fontweight="bold")
    snapshot = analysis.sample_snapshot
    if snapshot is None:
        return _unavailable_figure(
            fig, axes, "Charts unavailable: no sample snapshot is stored in this result."
        )

    data = np.asarray(snapshot, dtype=float)
    if data.ndim != 1 or data.size == 0 or not np.isfinite(data).all():
        return _unavailable_figure(
            fig, axes, "Charts unavailable: stored sample snapshot is invalid."
        )

    sorted_data = np.sort(data)
    lower, upper = float(sorted_data[0]), float(sorted_data[-1])
    if lower == upper:
        padding = max(abs(lower) * 0.05, 0.5)
        lower, upper = lower - padding, upper + padding
    x_grid = np.linspace(lower, upper, 500)
    distribution = get_scipy_dist(candidate, dict(params)) if candidate and params else None
    candidate_label = candidate or "selected candidate"

    _draw_density(axes[0, 0], data, x_grid, distribution, candidate_label)
    _draw_cdf(axes[0, 1], sorted_data, x_grid, distribution, candidate_label)
    _draw_qq(axes[1, 0], data, distribution, candidate)
    _draw_scores(axes[1, 1], prediction, candidate)

    # Keep a shared lower strip clear for the PDF footer; PNG uses the same
    # geometry so its axis labels are not clipped at the export boundary.
    fig.tight_layout(rect=(0, 0.09, 1, 0.94))
    return fig


def generate_plot_report(
    result,
    output_path: str | Path = "distribution_report.png",
    options=None,
) -> Path:
    """Write the canonical result's charts to PNG, using stored data only."""
    from pysatl_expert.reporting.result_report import ReportOptions

    settings = options or ReportOptions()
    if settings.sections is not None and "plots" not in settings.sections:
        raise ValueError("PNG output requires the 'plots' section in ReportOptions")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure = _build_result_figure(result)
    try:
        figure.savefig(output, dpi=300)
    finally:
        plt.close(figure)
    logger.info("Result charts saved to '%s'", output)
    return output
