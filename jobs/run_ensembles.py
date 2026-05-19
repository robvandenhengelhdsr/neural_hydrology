#!/usr/bin/env python3
"""
Databricks Job entrypoint: 30-member ensemble inference for all HDSR polders.

Runs ``neural_hydrology.inference.run_model`` with paths from ``.env``
(``BEST_MODEL_DIR`` / ``RUNS_DIR``, ``DATA_ENS_DIR``, ``INFERENCE_RUNS_DIR``).

Databricks (Jobs → Python script):
  Path: /Workspace/Shared/neural_hydrology/jobs/run_ensembles.py

Local (from repo-root, after ``pip install -e .``):
  python jobs/run_ensembles.py
  # equivalent:
  python -m neural_hydrology.inference.run_model \\
    --run_dir runs/<run_id> --data_dir data_ens --out_dir inference_runs --n_ensembles 30
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_DATABRICKS_ENV_CANDIDATES = (
    "/Workspace/Shared/neural_hydrology_fork/.env",
    "/Workspace/Shared/neural_hydrology/.env",
)
for _env_path in _DATABRICKS_ENV_CANDIDATES:
    if Path(_env_path).is_file():
        load_dotenv(_env_path, override=False)
        break

from neural_hydrology.inference.run_model import main as run_inference
from neural_hydrology.paths import get_env, get_path, get_project_root, load_env

LOGGER = logging.getLogger(__name__)
DEFAULT_N_ENSEMBLES = 30


def _resolve_run_dir() -> Path:
    """Trained run directory (must contain config.yml)."""
    env = load_env()
    for key in ("BEST_MODEL_DIR", "RUN_DIR"):
        raw = env.get(key)
        if raw and str(raw).strip():
            path = Path(str(raw)).expanduser()
            if not path.is_absolute():
                path = get_project_root() / path
            return path.resolve()

    runs_dir = get_path("RUNS_DIR")
    if (runs_dir / "config.yml").is_file():
        return runs_dir

    candidates = sorted(
        (
            d
            for d in runs_dir.iterdir()
            if d.is_dir() and (d / "config.yml").is_file()
        ),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise RuntimeError(
            f"No trained run found under {runs_dir}. "
            "Set BEST_MODEL_DIR in .env to the run folder (contains config.yml)."
        )
    latest = candidates[-1]
    LOGGER.warning(
        "BEST_MODEL_DIR not set; using newest run under RUNS_DIR: %s",
        latest,
    )
    return latest.resolve()


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(levelname)s:%(name)s:%(message)s",
    )

    run_dir = _resolve_run_dir()
    data_dir = get_path("DATA_ENS_DIR")
    out_dir = get_path("INFERENCE_RUNS_DIR")
    n_ensembles = int(get_env("N_ENSEMBLES", str(DEFAULT_N_ENSEMBLES)) or DEFAULT_N_ENSEMBLES)

    LOGGER.info(
        "Starting ensemble inference (n_ensembles=%s, run_dir=%s, data_dir=%s, out_dir=%s, root=%s)",
        n_ensembles,
        run_dir,
        data_dir,
        out_dir,
        get_project_root(),
    )

    sys.argv = [
        "run_model",
        "--run_dir",
        str(run_dir),
        "--data_dir",
        str(data_dir),
        "--out_dir",
        str(out_dir),
        "--n_ensembles",
        str(n_ensembles),
    ]
    run_inference()


if __name__ == "__main__":
    main()
