"""PDF presentation of an analysis result and selected model inputs."""

import logging
from pathlib import Path

import numpy as np
import scipy.stats as stats

from pysatl_expert.models.report import Report
from pysatl_expert.reporting.common import get_scipy_dist as _get_scipy_dist
from pysatl_expert.reporting.formatting import format_raw_statistic as _format_raw_statistic


logger = logging.getLogger(__name__)


def _bootstrap_comparison(report: Report) -> tuple[list, str]:
    """Compare original scores and top-1 shares without reordering by bootstrap."""
    ranks = report.model_ranks or report.final_ranks or {}
    rows = [
        [
            name,
            f"{score:.2%}",
            f"{report.bootstrap_ranks[name]:.2%}"
            if report.bootstrap_successful and name in report.bootstrap_ranks
            else "—",
        ]
        for name, score in sorted(ranks.items(), key=lambda item: -item[1])
    ]
    if not report.bootstrap_requested and not report.bootstrap_successful:
        if report.bootstrap_requested is None and report.bootstrap_stability is not None:
            return rows, "Repeat count unavailable"
        return rows, "Not performed"
    requested = report.bootstrap_requested
    count = f"Successful repeats: {report.bootstrap_successful}"
    if requested is not None:
        count += f" of {requested}"
    if not report.bootstrap_successful:
        count += "; unavailable"
    return rows, count


def _render_pdf_bootstrap_comparison(ax, report: Report, colors: dict):
    rows, status = _bootstrap_comparison(report)
    ax.axis("off")
    ax.set_title("4. Original Ranking & Bootstrap", fontsize=9, fontweight="bold")
    table = ax.table(
        cellText=rows or [["—"] * 3],
        colLabels=["Candidate", "Model score", "Bootstrap\nTop-1 share"],
        colWidths=[0.36, 0.30, 0.34],
        cellLoc="center",
        bbox=[0, 0.18, 1, 0.70],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    for (r, _), cell in table.get_celld().items():
        cell.set_edgecolor(colors["border"])
        cell.set_facecolor(colors["navy"] if r == 0 else "white")
        if r == 0:
            cell.set_text_props(color="white", fontweight="bold")
    ax.text(0, 0.07, status, transform=ax.transAxes, fontsize=7)


def _render_pdf_summary_card(
    ax_card, report: Report, st: dict[str, float | int], colors: dict[str, str]
):
    """Render structured executive identification table on Page 1."""
    ax_card.axis("off")

    winner = report.distribution_name
    params = report.parameters or {}

    param_items = [
        f"{k} = {v:.4f}" if isinstance(v, (int, float)) else f"{k} = {v}" for k, v in params.items()
    ]
    param_str = ",   ".join(param_items) if param_items else "—"

    model_score_text = (
        f"{report.model_confidence:.1%}" if report.model_confidence is not None else "—"
    )
    ranked = sorted((report.final_ranks or report.model_ranks).items(), key=lambda item: -item[1])
    flow_text = " / ".join(f"{name}: {score:.1%}" for name, score in ranked[:3]) or "—"

    quick_stats = (
        f"N = {st['sample_size']}   |   Domain = [{st['min']:.3f}, {st['max']:.3f}]   |   "
        f"Mean = {st['mean']:.3f}   |   Std Dev = {st['standard_deviation']:.3f}"
    )

    rows = [
        ["Recommendation", f"{winner.upper()}"],
        [
            "Model Confidence",
            f"{model_score_text}  (model score; not a calibrated probability)",
        ],
        ["Bootstrap", _bootstrap_comparison(report)[1]],
    ]
    if report.bootstrap_stability is not None:
        rows.append(
            [
                "Bootstrap Stability",
                (
                    f"{report.bootstrap_stability * 100:.1f}%  "
                    "(Consensus across non-parametric resamples)"
                ),
            ]
        )
    rows.extend(
        [
            ["Top 3 ranking", flow_text],
            ["Fitted Parameters", param_str],
            ["Sample Summary", quick_stats],
        ]
    )

    tbl = ax_card.table(
        cellText=rows,
        colWidths=[0.24, 0.76],
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.8)
    scale_y = 1.30 if len(rows) > 5 else 1.45
    tbl.scale(1.0, scale_y)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(colors["border"])
        cell.set_linewidth(0.8)
        if c == 0:
            cell.set_facecolor(colors["card_bg"])
            cell.set_text_props(color=colors["muted"], fontweight="bold")
        else:
            cell.set_facecolor("#ffffff" if r % 2 == 0 else "#fcfcfd")
            if r == 0:
                cell.set_text_props(color=colors["dark_text"], fontweight="bold", fontsize=10.0)
            elif r == 1:
                cell.set_text_props(color=colors["empirical"], fontweight="bold")
            elif r == 2 and report.bootstrap_stability is not None:
                cell.set_text_props(color=colors["green"], fontweight="bold")
            else:
                cell.set_text_props(color=colors["dark_text"])


def _fitted_candidate_curves(report: Report, scipy_dist) -> list:
    """Never invent parameters for candidates whose fit was not obtained."""
    winner = report.distribution_name
    ranks = report.final_ranks or report.model_ranks
    curves = []
    for name, _ in sorted(ranks.items(), key=lambda item: -item[1])[:3]:
        params = report.candidate_parameters.get(name)
        if not params and name == winner:
            params = report.parameters
        if params:
            fitted = _get_scipy_dist(name, params)
            if fitted is not None:
                curves.append((name, fitted))
    if not curves and scipy_dist is not None:
        curves.append((winner, scipy_dist))
    return curves


def _render_pdf_diagnostics(
    fig1, gs_diag, data_arr: np.ndarray, report: Report, scipy_dist, colors: dict[str, str]
):
    """Render 4-panel diagnostic plots on Page 1."""
    winner = report.distribution_name
    n_samples = len(data_arr)
    d_min, d_max = float(np.min(data_arr)), float(np.max(data_arr))
    data_sorted = np.sort(data_arr)
    curves = _fitted_candidate_curves(report, scipy_dist)
    curve_colors = [colors["theoretical"], colors["green"], "#7c3aed"]

    # Panel 1: Histogram + PDF
    ax1 = fig1.add_subplot(gs_diag[0, 0])
    ax1.set_title(
        "1. Empirical Histogram & Fitted PDF",
        fontsize=9.0,
        fontweight="bold",
        color=colors["dark_text"],
        pad=6,
    )
    ax1.hist(
        data_arr,
        bins=min(30, int(np.sqrt(n_samples))),
        density=True,
        color="#bfdbfe",
        edgecolor=colors["empirical"],
        alpha=0.55,
        label="Empirical Data",
    )
    for (name, fitted), curve_color in zip(curves, curve_colors, strict=False):
        x_grid = np.linspace(d_min, d_max, 400)
        try:
            pdf_vals = fitted.pdf(x_grid)
            ax1.plot(x_grid, pdf_vals, color=curve_color, lw=1.9, label=f"Fitted {name} PDF")
        except Exception as exc:
            logger.warning("Could not plot %s PDF: %s", name, exc)
    ax1.set_xlabel("Value (x)", fontsize=8)
    ax1.set_ylabel("Probability Density", fontsize=8)
    ax1.legend(loc="upper right", fontsize=7.5, frameon=True, framealpha=0.9)
    ax1.grid(True, linestyle="--", alpha=0.35)

    # Panel 2: ECDF vs Theoretical CDF
    ax2 = fig1.add_subplot(gs_diag[0, 1])
    ax2.set_title(
        "2. Empirical CDF vs Theoretical CDF",
        fontsize=9.0,
        fontweight="bold",
        color=colors["dark_text"],
        pad=6,
    )
    ecdf_y = np.arange(1, n_samples + 1) / n_samples
    ax2.plot(data_sorted, ecdf_y, color=colors["empirical"], lw=1.8, label="Empirical CDF (ECDF)")
    for (name, fitted), curve_color in zip(curves, curve_colors, strict=False):
        try:
            cdf_vals = fitted.cdf(data_sorted)
            ax2.plot(
                data_sorted,
                cdf_vals,
                color=curve_color,
                lw=1.8,
                linestyle="--",
                label=f"Fitted {name} CDF",
            )
        except Exception as exc:
            logger.warning("Could not plot %s CDF: %s", name, exc)
    ax2.set_xlabel("Value (x)", fontsize=8)
    ax2.set_ylabel("Cumulative Probability", fontsize=8)
    ax2.legend(loc="lower right", fontsize=7.5, frameon=True, framealpha=0.9)
    ax2.grid(True, linestyle="--", alpha=0.35)

    # Panel 3: Q-Q Plot
    ax3 = fig1.add_subplot(gs_diag[1, 0])
    ax3.set_title(
        f"3. Quantile-Quantile (Q-Q) Plot vs {winner}",
        fontsize=9.0,
        fontweight="bold",
        color=colors["dark_text"],
        pad=6,
    )
    if scipy_dist is not None:
        try:
            (osm, osr), (slope, intercept, r) = stats.probplot(data_arr, dist=scipy_dist, plot=None)
            ax3.scatter(
                osm,
                osr,
                color=colors["empirical"],
                alpha=0.55,
                edgecolors="none",
                s=16,
                label="Sample Quantiles",
            )
            line_x = np.array([np.min(osm), np.max(osm)])
            ax3.plot(
                line_x,
                slope * line_x + intercept,
                color=colors["theoretical"],
                linestyle="--",
                lw=1.8,
                label=f"Ref Line (R² = {r**2:.3f})",
            )
            ax3.set_xlabel("Theoretical Quantiles", fontsize=8)
            ax3.set_ylabel("Sample Quantiles", fontsize=8)
            ax3.legend(loc="upper left", fontsize=7.5, frameon=True, framealpha=0.9)
            ax3.grid(True, linestyle="--", alpha=0.35)
        except Exception as exc:
            logger.warning("Could not plot %s Q-Q diagnostics: %s", winner, exc)
            ax3.text(0.5, 0.5, "Q-Q Plot unavailable", ha="center", va="center", fontsize=8)
    else:
        ax3.text(0.5, 0.5, "Q-Q Plot unavailable", ha="center", va="center", fontsize=8)

    # Panel 4: Probabilities Bar Chart
    ax4 = fig1.add_subplot(gs_diag[1, 1])
    if report.bootstrap_requested or report.bootstrap_successful:
        _render_pdf_bootstrap_comparison(ax4, report, colors)
        return
    is_bootstrap = report.confidence_kind == "bootstrap_stability"
    rank_title = (
        "4. ML Decision Ranking (Bootstrap Votes)" if is_bootstrap else "4. ML Candidate Scores"
    )
    ax4.set_title(rank_title, fontsize=9.0, fontweight="bold", color=colors["dark_text"], pad=6)
    if report.final_ranks:
        sorted_ranks = sorted(report.final_ranks.items(), key=lambda x: x[1])
        dists = [x[0] for x in sorted_ranks]
        probs = [x[1] * 100 for x in sorted_ranks]
        bar_colors = [colors["empirical"] if d == winner else "#e2e8f0" for d in dists]
        bars = ax4.barh(
            dists, probs, color=bar_colors, edgecolor=colors["border"], lw=0.6, height=0.65
        )
        xlabel = "Bootstrap Vote Share (%)" if is_bootstrap else "Model Score (%)"
        ax4.set_xlabel(xlabel, fontsize=8)
        ax4.set_xlim(0, max(max(probs) * 1.25, 10))
        for bar in bars:
            w = bar.get_width()
            ax4.text(
                w + 1.2,
                bar.get_y() + bar.get_height() / 2,
                f"{w:.1f}%",
                va="center",
                fontsize=7.5,
                fontweight="bold" if w > 50 else "normal",
                color=colors["dark_text"],
            )
        ax4.grid(True, axis="x", linestyle="--", alpha=0.35)


def _pdf_table_pages(pdf, title, headers, rows, note, colors, candidate_rows=None):
    """Paginate evidence rather than silently dropping overflowing statistics."""
    import matplotlib.pyplot as plt

    for offset in range(0, max(1, len(rows)), 40):
        page_rows = rows[offset : offset + 40] or [["—"] * len(headers)]
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.08, 0.955, title, fontsize=16, fontweight="bold")
        fig.text(0.08, 0.92, note, fontsize=8, va="top", linespacing=1.7)
        ax = fig.add_axes([0.08, 0.13, 0.84, 0.69])
        ax.axis("off")
        table = ax.table(
            cellText=page_rows,
            colLabels=headers,
            cellLoc="left",
            bbox=[0, 1 - (len(page_rows) + 1) / 41, 1, (len(page_rows) + 1) / 41],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7.3)
        tables = [table]
        if candidate_rows is not None and offset == 0:
            lower_top = 0.82 - 0.69 * (len(page_rows) + 1) / 41 - 0.06
            fig.text(0.08, lower_top, "Top candidate parameters", fontsize=11, fontweight="bold")
            params_ax = fig.add_axes([0.08, lower_top - 0.13, 0.84, 0.10])
            params_ax.axis("off")
            params_table = params_ax.table(
                cellText=candidate_rows or [["—", "—"]],
                colLabels=["Top candidate", "Fitted parameters"],
                cellLoc="left",
                bbox=[0, 0, 1, 1],
            )
            params_table.auto_set_font_size(False)
            params_table.set_fontsize(7.3)
            tables.append(params_table)
            fig.text(
                0.08, lower_top - 0.19, "Prototype limitations", fontsize=11, fontweight="bold"
            )
            fig.text(
                0.08,
                lower_top - 0.22,
                "8 distribution classes; synthetic training data.\n"
                "Positive distributions: loc=0. Beta: support [0,1].\n"
                "Beta(1,1)=Uniform(0,1): the distribution labels overlap.\n"
                "—: value not obtained.",
                fontsize=8,
                va="top",
                linespacing=1.7,
            )
        for styled_table in tables:
            for (r, _), cell in styled_table.get_celld().items():
                cell.set_edgecolor(colors["border"])
                cell.set_linewidth(0.5)
                cell.set_facecolor(
                    colors["navy"] if r == 0 else (colors["card_bg"] if r % 2 else "white")
                )
                if r == 0:
                    cell.set_text_props(color="white", fontweight="bold")
        fig.text(
            0.08,
            0.05,
            "PySATL",
            fontsize=8,
            color=colors["muted"],
        )
        fig.text(0.92, 0.05, f"Page {pdf.get_pagecount() + 1}", ha="right", fontsize=8)
        pdf.savefig(fig)
        plt.close(fig)


def generate_pdf_report(
    data: np.ndarray,
    report: Report,
    output_path: str | Path = "distribution_report.pdf",
    *,
    mode: str = "summary",
) -> Path:
    """Generate a paginated recommendation PDF in English.

    Includes recommendations, actual hierarchy scores and selected model inputs. It
    does not perform hypothesis-test decisions.

    Args:
        data: 1D empirical sample array.
        report: Report object returned by DistributionPipeline.identify_best.
        output_path: Target path for the output PDF. Defaults to 'distribution_report.pdf'.
        mode: 'summary' includes recommendations, parameters and hierarchy scores;
            'full' additionally includes selected model inputs.

    Returns:
        Path: Absolute path to the generated PDF file.
    """
    if mode not in {"summary", "full"}:
        raise ValueError("PDF mode must be 'summary' or 'full'")
    try:
        import datetime

        import matplotlib.gridspec as gridspec
        import matplotlib.lines as mlines
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ModuleNotFoundError as exc:
        raise RuntimeError("PDF report generation requires matplotlib") from exc

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data_arr = np.asarray(data, dtype=float)
    stats_dict = report.sample_statistics
    required_statistics = {
        "sample_size",
        "min",
        "max",
        "mean",
        "median",
        "standard_deviation",
        "variance",
        "skew",
        "kurtosis",
        "iqr",
        "relative_iqr",
        "entropy",
    }
    missing_statistics = required_statistics.difference(stats_dict)
    if missing_statistics:
        raise ValueError(
            "Report is missing descriptive sample statistics: "
            + ", ".join(sorted(missing_statistics))
        )
    winner = report.distribution_name
    scipy_dist = _get_scipy_dist(winner, report.parameters) if report.parameters else None

    colors = {
        "dark_text": "#0f172a",
        "body": "#1e293b",
        "muted": "#475569",
        "gray": "#64748b",
        "border": "#cbd5e1",
        "card_bg": "#f8fafc",
        "empirical": "#1d4ed8",
        "empirical_fill": "#dbeafe",
        "theoretical": "#b91c1c",
        "green": "#15803d",
        "red": "#b91c1c",
        "navy": "#0f172a",
        "accent_blue": "#1d4ed8",
    }

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
            "font.size": 9,
            "text.color": colors["dark_text"],
            "axes.labelcolor": colors["dark_text"],
            "xtick.color": colors["dark_text"],
            "ytick.color": colors["dark_text"],
        }
    )

    with PdfPages(output_path) as pdf:
        # --- Page 1: Executive Summary & Diagnostic Visualizations ---
        fig1 = plt.figure(figsize=(8.27, 11.69), dpi=300)
        fig1.text(
            0.08,
            0.958,
            "PySATL Expert System",
            fontsize=18,
            fontweight="bold",
            color=colors["navy"],
        )
        fig1.text(
            0.08,
            0.938,
            "Automated Empirical Distribution Identification Report",
            fontsize=10.5,
            color=colors["accent_blue"],
        )
        date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        fig1.text(0.92, 0.958, f"Date: {date_str}", fontsize=8.5, color=colors["gray"], ha="right")
        fig1.text(
            0.92,
            0.940,
            "Status: Completed",
            fontsize=8.5,
            color=colors["green"],
            fontweight="bold",
            ha="right",
        )
        fig1.add_artist(
            mlines.Line2D([0.08, 0.92], [0.926, 0.926], color=colors["accent_blue"], lw=1.8)
        )

        gs1 = gridspec.GridSpec(
            2,
            1,
            height_ratios=[1.6, 6.0],
            left=0.08,
            right=0.92,
            top=0.908,
            bottom=0.075,
            hspace=0.24,
        )
        ax_card = fig1.add_subplot(gs1[0])
        ax_card.axis("off")
        _render_pdf_summary_card(ax_card, report, stats_dict, colors)

        gs_diag = gridspec.GridSpecFromSubplotSpec(
            2, 2, subplot_spec=gs1[1], hspace=0.34, wspace=0.28
        )
        _render_pdf_diagnostics(fig1, gs_diag, data_arr, report, scipy_dist, colors)

        fig1.text(
            0.08,
            0.026,
            "PySATL",
            fontsize=8,
            color=colors["gray"],
        )
        fig1.text(0.92, 0.026, "Page 1", fontsize=8, color=colors["gray"], ha="right")
        pdf.savefig(fig1)
        plt.close(fig1)

        stage_rows = [
            ["Stage 1", family, f"{score:.2%}", "—"]
            for family, score in report.stage1_scores.items()
        ]
        for family, scores in report.stage2_scores.items():
            for name, conditional in scores.items():
                joint = report.stage1_scores[family] * conditional
                stage_rows.append([family, name, f"{conditional:.2%}", f"{joint:.2%}"])
        ranks = report.final_ranks or report.model_ranks
        parameter_rows = [
            [
                name,
                ", ".join(
                    f"{k}={_format_raw_statistic(v)}"
                    for k, v in report.candidate_parameters.get(
                        name, (report.parameters or {}) if name == winner else {}
                    ).items()
                )
                or "—",
            ]
            for name, _ in sorted(ranks.items(), key=lambda item: -item[1])[:3]
        ]
        _pdf_table_pages(
            pdf,
            "How the hierarchy scored this sample",
            ["Stage / family", "Candidate", "Forest score", "Overall score"],
            stage_rows,
            "Stage 1: family scores. Stage 2: conditional scores within each family.\n"
            "Overall score = family score × conditional score.\n"
            "Scores refer to the original sample; bootstrap votes are shown separately.",
            colors,
            candidate_rows=parameter_rows,
        )
        if mode == "summary":
            logger.info("Summary PDF report saved to '%s'", output_path)
            return output_path
        if report.bootstrap_errors:
            import textwrap

            _pdf_table_pages(
                pdf,
                "Bootstrap evaluation failures",
                ["Failure"],
                [
                    [line]
                    for message in report.bootstrap_errors
                    for line in textwrap.wrap(message, 100)
                ],
                "Failed resample evaluations are excluded from top-1 shares.\n"
                "Individual criterion warnings within valid resamples remain in runtime logs.",
                colors,
            )
        feature_rows = [
            ["Stage 1", name, _format_raw_statistic(value)]
            for name, value in report.stage1_features.items()
        ]
        for family, features in report.stage2_features.items():
            feature_rows.extend(
                [family, name, _format_raw_statistic(value)] for name, value in features.items()
            )
        _pdf_table_pages(
            pdf,
            "Selected features actually supplied to the forests",
            ["Stage / family", "Feature", "Raw input value"],
            feature_rows,
            "Selected feature lists come from the loaded model, not a fixed template.\n"
            "—: value not obtained. Missingness can influence Random Forest decisions.\n"
            "Scores describe voting; this trace is not a causal feature-attribution analysis.",
            colors,
        )
    logger.info(f"Publication-quality PDF report saved to '{output_path}'")
    return output_path
