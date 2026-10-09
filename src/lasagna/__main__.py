"""Allow running as: python -m lasagna [command]

Usage::

    python -m lasagna init
    python -m lasagna upload report.pdf
    python -m lasagna search "quarterly revenue"
    python -m lasagna serve
    python -m lasagna --help
"""

from lasagna.cli import app

app()
