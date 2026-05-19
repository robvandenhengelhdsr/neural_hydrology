#!/usr/bin/env python3
"""
Databricks Job entrypoint: build ensemble time series NetCDFs for all HDSR polders.

Runs preprocessing for every SHAPE_ID in the attributes CSV over a 365-day
historical window (plus HARMONIE ensemble forecast merge).

Databricks (Jobs → Python script):
  Path: /Workspace/Shared/neural_hydrology/jobs/run_create_timeseries_all_polders.py

Local:
  python jobs/run_create_timeseries_all_polders.py
  # or after pip install -e .:
  python -m neural_hydrology.preprocessing.create_timeseries_files --days 365
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Repo layout: <root>/jobs/this_file.py, <root>/src/neural_hydrology/
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from neural_hydrology.preprocessing.create_timeseries_files import (  # noqa: E402
    MissingDataConfig,
    create_timeseries_files,
)

LOGGER = logging.getLogger(__name__)

DAYS = 365


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(levelname)s:%(name)s:%(message)s",
    )

    days = int(os.environ.get("PREPROCESSING_DAYS", str(DAYS)))
    LOGGER.info(
        "Starting preprocessing for all polders (days=%s, repo_root=%s)",
        days,
        _REPO_ROOT,
    )

    create_timeseries_files(
        days=days,
        missing_cfg=MissingDataConfig(),
        basin_ids=None,
    )


if __name__ == "__main__":
    main()
