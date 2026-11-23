"""Internal immutable records shared by report renderers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class _Section:
    name: str
    title: str
    lines: tuple[str, ...]
    columns: tuple[str, ...] = ()
    rows: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True)
class _ReportContent:
    title: str
    status_lines: tuple[str, ...]
    sections: tuple[_Section, ...]
