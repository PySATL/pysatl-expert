"""Shared formatting for text and graphical reports."""

import numpy as np


def format_raw_statistic(value) -> str:
    """Format raw values without interpreting them as hypothesis decisions."""
    if value is None:
        return "—"
    number = float(value)
    return f"{number:.6g}" if np.isfinite(number) else "—"
