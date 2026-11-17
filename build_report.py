"""Recalculate the Servo experiment and assemble the final PDF."""

from pathlib import Path

from threadpoolctl import threadpool_limits

import run_analysis
from pdf_report import assemble_pdf, register_fonts


def main() -> None:
    """Build from raw data. No previously generated file is required."""
    register_fonts()
    with threadpool_limits(limits=1):
        run_analysis.main()

    root = Path(__file__).resolve().parent
    assemble_pdf(root)
    print(f"Отчёт собран: {root / 'report.pdf'}", flush=True)


if __name__ == "__main__":
    main()
