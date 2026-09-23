import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from pysatl_expert.app import build_pipeline
from pysatl_expert.models.report import Report
from pysatl_expert.reporting.pdf import generate_pdf_report


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the user-facing inference CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("sample.csv"))
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to a trusted joblib model with an adjacent manifest",
    )
    parser.add_argument("--output", type=Path, default=Path("output/report.pdf"))
    parser.add_argument("--delimiter", default=",")
    parser.add_argument("--skip-header-rows", type=_non_negative_int, default=0)
    parser.add_argument("--bootstraps", type=_non_negative_int, default=0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--pdf-mode", choices=("summary", "full"), default="summary")
    return parser


def load_sample(
    input_path: Path,
    *,
    delimiter: str = ",",
    skip_header_rows: int = 0,
) -> np.ndarray:
    """Read exactly one numeric column from a user-provided file."""
    values = np.loadtxt(input_path, delimiter=delimiter, skiprows=skip_header_rows, ndmin=2)
    if values.shape[1] != 1:
        raise ValueError("Input file must contain exactly one column of observations")
    return values[:, 0]


def main(argv: list[str] | None = None) -> Report:
    """Analyze the sample and save a PDF; return the same result for Python use."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    if args.output.exists():
        raise FileExistsError(f"Report already exists; choose another --output path: {args.output}")
    sample = load_sample(
        args.input,
        delimiter=args.delimiter,
        skip_header_rows=args.skip_header_rows,
    )
    pipeline = build_pipeline(model_path=args.model)
    identify_options = {
        "n_bootstraps": args.bootstraps,
        "random_state": args.random_state,
    }
    report = pipeline.identify_best(sample, **identify_options)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    generate_pdf_report(sample, report, output_path=args.output, mode=args.pdf_mode)
    return report


def entrypoint() -> int:
    """Run the installed console command without exposing the Report to ``sys.exit``."""
    main()
    return 0
