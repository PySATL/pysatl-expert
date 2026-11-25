"""PDF layout for already-collected report content."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextToPath

from pysatl_expert.reporting._report_content import _ReportContent, _Section


def _wrap_pdf_line(
    line: str, font_size: float, font: FontProperties, max_width: float
) -> list[str]:
    """Preserve explicit line breaks and wrap each physical paragraph by width."""
    return [
        piece
        for paragraph in line.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        for piece in _wrap_pdf_paragraph(paragraph, font_size, font, max_width)
    ]


def _wrap_pdf_paragraph(
    line: str, font_size: float, font: FontProperties, max_width: float
) -> list[str]:
    """Wrap by rendered font width so long Russian/Latin values stay on the page."""
    text_path = TextToPath()
    sized_font = font.copy()
    sized_font.set_size(font_size)

    def width(text: str) -> float:
        return text_path.get_text_width_height_descent(text, sized_font, ismath=False)[0]

    wrapped: list[str] = []
    current = ""
    for word in line.split(" "):
        candidate = f"{current} {word}" if current else word
        if width(candidate) <= max_width:
            current = candidate
            continue
        if current:
            wrapped.append(current)
            current = ""
        part = ""
        for char in word:
            if part and width(part + char) > max_width:
                wrapped.append(part)
                part = char
            else:
                part += char
        current = part
    if current or not wrapped:
        wrapped.append(current)
    return wrapped


_TABLE_FONT_SIZE = 10
_TABLE_LINE_HEIGHT = 13
_TABLE_VERTICAL_PADDING = 3
_TABLE_CELL_PAD = 0.04
_TABLE_BOTTOM = 0.025
_TABLE_GAP = 0.018


def _wrap_table_row(row, font: FontProperties, ax, *, header: bool = False):
    width_points = ax.get_window_extent().width * 72 / ax.figure.dpi
    available_width = width_points / len(row) * (1 - 2 * _TABLE_CELL_PAD) - 2
    cell_font = font.copy()
    cell_font.set_size(_TABLE_FONT_SIZE)
    if header:
        cell_font.set_weight("bold")
    return tuple(
        "\n".join(
            piece
            for line in cell.split("\n")
            for piece in _wrap_pdf_line(line, _TABLE_FONT_SIZE, cell_font, available_width)
        )
        for cell in row
    )


def _table_row_height(row, ax) -> float:
    lines = max(len(cell.split("\n")) for cell in row)
    height_points = ax.get_window_extent().height * 72 / ax.figure.dpi
    return (_TABLE_LINE_HEIGHT * lines + 2 * _TABLE_VERTICAL_PADDING) / height_points


def _draw_table_chunk(ax, section: _Section, rows, y: float, font: FontProperties) -> float:
    table_rows = [
        _wrap_table_row(section.columns, font, ax, header=True),
        *(_wrap_table_row(row, font, ax) for row in rows),
    ]
    row_heights = [_table_row_height(row, ax) for row in table_rows]
    height = sum(row_heights)
    table = ax.table(
        cellText=table_rows,
        cellLoc="left",
        loc="upper left",
        bbox=(0, y - height, 1, height),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(_TABLE_FONT_SIZE)
    numeric_columns = {
        "Value",
        "Estimate",
        "Statistic",
        "Score",
        "p-value",
        "Critical values",
        "alpha",
    }
    for (row, column), cell in table.get_celld().items():
        cell.set_height(row_heights[row])
        cell.PAD = _TABLE_CELL_PAD
        cell.set_edgecolor("#D8E0E8")
        cell.set_linewidth(0.35)
        cell_font = font.copy()
        cell_font.set_size(_TABLE_FONT_SIZE)
        if row == 0:
            cell.set_facecolor("#E8EEF4")
            cell_font.set_weight("bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F7F9FB")
        cell.set_text_props(fontproperties=cell_font, color="#17324D" if row == 0 else "#263746")
        if row > 0 and section.columns[column] in numeric_columns:
            cell.get_text().set_ha("right")
    return y - height - _TABLE_GAP


def _table_chunk_size(section: _Section, rows, available_y: float, font: FontProperties, ax) -> int:
    header = _wrap_table_row(section.columns, font, ax, header=True)
    used_height = _table_row_height(header, ax) + _TABLE_GAP
    count = 0
    for row in rows:
        row_height = _table_row_height(_wrap_table_row(row, font, ax), ax)
        if used_height + row_height > available_y - _TABLE_BOTTOM:
            break
        used_height += row_height
        count += 1
    return count


def _table_first_chunk_height(section: _Section, font: FontProperties, ax) -> float:
    header = _wrap_table_row(section.columns, font, ax, header=True)
    first_row = _wrap_table_row(section.rows[0], font, ax)
    height = _table_row_height(header, ax) + _table_row_height(first_row, ax) + _TABLE_GAP
    if height > 0.885 - _TABLE_BOTTOM:
        first_line = tuple(cell.split("\n")[0] for cell in first_row)
        return _table_row_height(header, ax) + _table_row_height(first_line, ax) + _TABLE_GAP
    return height


def _split_table_row(section: _Section, row, available_y: float, font: FontProperties, ax):
    header = _wrap_table_row(section.columns, font, ax, header=True)
    height_points = ax.get_window_extent().height * 72 / ax.figure.dpi
    available = available_y - _TABLE_BOTTOM - _TABLE_GAP - _table_row_height(header, ax)
    line_count = int((available * height_points - 2 * _TABLE_VERTICAL_PADDING) / _TABLE_LINE_HEIGHT)
    if line_count < 1:
        return None
    wrapped = [cell.split("\n") for cell in _wrap_table_row(row, font, ax)]
    first = tuple("\n".join(lines[:line_count]) for lines in wrapped)
    rest = tuple("\n".join(lines[line_count:]) for lines in wrapped)
    return first, rest


def _wrapped_text_height(lines: tuple[str, ...], font: FontProperties, size: float) -> float:
    return sum(len(_wrap_pdf_line(line, size, font, 470)) for line in lines) * size / 420


def _draw_wrapped_lines(ax_getter, lines, y, font, size, color, ensure_space) -> float:
    for line in lines:
        for piece in _wrap_pdf_line(line, size, font, 470):
            height = size / 420
            y = ensure_space(height, y)
            axis = ax_getter()
            axis.text(
                0.015,
                y,
                piece,
                color=color,
                fontsize=size,
                transform=axis.transAxes,
                va="top",
                fontproperties=font,
            )
            y -= height
    return y


def _draw_pdf_section(section, get_axis, y, font, navy, ensure_space, new_page):
    reserve = 0.054 + _wrapped_text_height(section.lines, font, 10.5)
    if section.columns and section.rows:
        reserve += _table_first_chunk_height(section, font, get_axis())
    y = ensure_space(reserve, y) - 0.012
    ax = get_axis()
    ax.text(
        0,
        y,
        section.title,
        color=navy,
        fontsize=12,
        fontweight="bold",
        transform=ax.transAxes,
        va="top",
        fontproperties=font,
    )
    y -= 0.03
    ax.plot([0, 1], [y + 0.01, y + 0.01], color="#D5DEE7", linewidth=0.6, transform=ax.transAxes)
    y = _draw_wrapped_lines(get_axis, section.lines, y - 0.012, font, 10.5, "#263746", ensure_space)
    ax = get_axis()
    remaining = list(section.rows) if section.columns else []
    while remaining:
        chunk_size = _table_chunk_size(section, remaining, y, font, ax)
        if chunk_size:
            chunk, remaining = remaining[:chunk_size], remaining[chunk_size:]
        else:
            split = _split_table_row(section, remaining[0], y, font, ax)
            if split is None:
                chunk = []
            else:
                first, rest = split
                chunk = [first]
                remaining = [rest, *remaining[1:]] if any(rest) else remaining[1:]
        if chunk:
            y = _draw_table_chunk(ax, section, chunk, y, font)
        if remaining:
            ax = new_page()
            ax.text(
                0,
                0.92,
                f"{section.title} (continued)",
                color=navy,
                fontsize=12,
                fontweight="bold",
                transform=ax.transAxes,
                va="top",
                fontproperties=font,
            )
            y = 0.885
    return y, ax


def _draw_chart_heading(figure: Figure, section: _Section, font: FontProperties) -> None:
    figure.text(
        0.08,
        0.94,
        section.title,
        fontsize=12,
        fontweight="bold",
        color="#17324D",
        va="top",
        fontproperties=font,
    )
    y = 0.90
    width_points, height_points = figure.get_size_inches() * 72
    for line in section.lines:
        for piece in _wrap_pdf_line(line, 10, font, width_points * 0.84):
            figure.text(
                0.08,
                y,
                piece,
                fontsize=10,
                color="#586879",
                va="top",
                fontproperties=font,
            )
            y -= 14 / height_points
    figure.tight_layout(rect=(0, 0.09, 1, y - 0.025))


def _draw_pdf_pages(
    output: Path,
    content: _ReportContent,
    font: FontProperties,
    chart_figure: Figure | None = None,
) -> None:
    navy = "#17324D"
    muted = "#586879"
    with PdfPages(output) as pdf:
        page_number = 0
        fig: Any = None
        ax: Any = None
        y = 0.89

        def save_text_page():
            nonlocal fig, ax
            if fig is None:
                return
            ax.text(
                0,
                -0.045,
                "PySATL Expert  |  Statistical report",
                transform=ax.transAxes,
                fontsize=8,
                color=muted,
                va="bottom",
                fontproperties=font,
            )
            ax.text(
                1,
                -0.045,
                f"Page {page_number}",
                transform=ax.transAxes,
                fontsize=8,
                color=muted,
                va="bottom",
                ha="right",
                fontproperties=font,
            )
            pdf.savefig(fig)
            fig.clear()
            fig = None
            ax = None

        def new_page():
            nonlocal fig, ax, y, page_number
            save_text_page()
            page_number += 1
            fig = Figure(figsize=(8.27, 11.69), dpi=140)
            ax = fig.add_axes((0.085, 0.085, 0.83, 0.84))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
            ax.plot([0, 1], [0.965, 0.965], color=navy, linewidth=2, transform=ax.transAxes)
            ax.text(
                0,
                0.985,
                "PySATL EXPERT  /  RESULTS",
                color=navy,
                fontsize=8,
                fontweight="bold",
                transform=ax.transAxes,
                va="top",
                fontproperties=font,
            )
            y = 0.92
            return ax

        def save_chart_page(section: _Section):
            nonlocal page_number
            if chart_figure is None:
                return
            save_text_page()
            page_number += 1
            _draw_chart_heading(chart_figure, section, font)
            chart_figure.text(
                0.08,
                0.025,
                "PySATL Expert  |  Statistical report",
                fontsize=8,
                color=muted,
                va="bottom",
                fontproperties=font,
            )
            chart_figure.text(
                0.92,
                0.025,
                f"Page {page_number}",
                fontsize=8,
                color=muted,
                va="bottom",
                ha="right",
                fontproperties=font,
            )
            pdf.savefig(chart_figure)

        def ensure_space(height, current_y):
            nonlocal y
            y = current_y
            if y - height < 0.025:
                new_page()
            return y

        def get_axis():
            return ax

        ax = new_page()
        ax.text(
            0,
            y,
            content.title,
            color=navy,
            fontsize=19,
            fontweight="bold",
            transform=ax.transAxes,
            va="top",
            fontproperties=font,
        )
        y -= 0.055
        ax.plot([0, 1], [y, y], color="#D5DEE7", linewidth=0.8, transform=ax.transAxes)
        y -= 0.025
        ax.text(
            0,
            y,
            "Status",
            color=navy,
            fontsize=11,
            fontweight="bold",
            transform=ax.transAxes,
            va="top",
            fontproperties=font,
        )
        y -= 0.026
        y = _draw_wrapped_lines(lambda: ax, content.status_lines, y, font, 10, muted, ensure_space)
        for section in content.sections:
            if section.name == "plots" and chart_figure is not None:
                save_chart_page(section)
                continue
            if fig is None:
                ax = new_page()
            y, ax = _draw_pdf_section(section, get_axis, y, font, navy, ensure_space, new_page)
        save_text_page()
