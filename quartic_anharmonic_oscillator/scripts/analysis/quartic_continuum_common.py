#!/usr/bin/env python3
# Shared statistical and file-format helpers for the beta=5 quartic-oscillator
# continuum analysis.  These routines define the retained (lambda,Nt) grid,
# blocking convention, and weighted linear fits in eta^2.
"""Shared helpers for the beta=5 continuum analysis."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np


RAW_DIR = Path("data/raw/production/beta5_continuum_v2")
PROCESSED = Path("data/processed/production")
PRODUCTION_BLOCK_SIZE_ROWS = 2000
DEFAULT_ETA_MIN = 0.025
DEFAULT_ETA_MAX = 0.2
EXCLUDED_ETA_VALUES = (0.125,)
EXCLUDED_NT_VALUES = (40,)
LAMBDAS_V2 = (0.0125, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 0.75, 1.00)
FIT_NT_VALUES = (25, 28, 32, 36, 50, 64, 80, 100, 128, 160, 200, 256)
ZOOM_LAMBDAS = (0.025, 0.25, 1.00)
ZOOM_EXTRA_NT_VALUES = (20, 320)
MEAS_COLUMNS = (
    "measurement_index", "sweep", "beta", "eta", "Nt", "lambda",
    "accept_rate_metro", "accept_rate_over", "y_mean", "y2_mean", "y4_mean",
    "dy2_mean", "V_mean", "K_virial", "E_virial",
)


def is_excluded_eta_nt(eta: float, nt: int) -> bool:
    """Return true for eta/Nt values excluded from final processed fits."""
    return any(np.isclose(float(eta), value) for value in EXCLUDED_ETA_VALUES) or int(nt) in EXCLUDED_NT_VALUES


def is_required_measurement(lam: float, nt: int) -> bool:
    """Return true for raw measurement files used by final analysis or plots."""
    if int(nt) in FIT_NT_VALUES:
        return True
    if int(nt) in ZOOM_EXTRA_NT_VALUES and any(np.isclose(float(lam), value) for value in ZOOM_LAMBDAS):
        return True
    return False


def fmt(value: float) -> str:
    value = float(value)
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0.0 else "-inf"
    return f"{value:.12g}"


def raw_measurement_files(raw_dir: Path = RAW_DIR) -> list[Path]:
    """Return the final production measurement files in deterministic order."""
    files = sorted(raw_dir.glob("anharmonic_beta5_v2_lambda*_nt*_c2p4_measurements.dat"))
    if not files:
        raise FileNotFoundError(f"no final measurement files found in {raw_dir}")
    return files


def parse_measurement_metadata(path: Path) -> dict[str, Any]:
    """Read simulation metadata from the first header line of a raw file."""
    with path.open("r", encoding="ascii") as handle:
        first = handle.readline().strip()
    if not first.startswith("#"):
        raise ValueError(f"{path}: missing metadata header")
    tokens = first[1:].split()
    if len(tokens) % 2 != 0:
        raise ValueError(f"{path}: malformed metadata header")
    metadata: dict[str, Any] = {}
    for key, value in zip(tokens[0::2], tokens[1::2]):
        metadata[key] = value
    for key in ("beta", "eta", "lambda", "delta"):
        metadata[key] = float(metadata[key])
    for key in ("Nt", "seed", "n_over", "n_therm", "n_sweeps", "meas_stride"):
        metadata[key] = int(metadata[key])
    metadata["raw_file"] = str(path)
    metadata["block_size_rows"] = PRODUCTION_BLOCK_SIZE_ROWS
    return metadata


def read_named(path: Path) -> np.ndarray:
    header = ""
    with path.open("r", encoding="ascii") as handle:
        for line in handle:
            if line.startswith("#"):
                header = line[1:].strip()
            elif line.strip():
                break
    data = np.genfromtxt(path, comments="#", names=header.split(), dtype=None, encoding=None)
    return np.atleast_1d(data)


def write_table(path: Path, columns: tuple[str, ...], rows: list[dict[str, Any]], int_columns: set[str] | None = None) -> None:
    int_columns = int_columns or set()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        handle.write("# " + " ".join(columns) + "\n")
        for row in rows:
            values = []
            for col in columns:
                value = row[col]
                if isinstance(value, str):
                    values.append(value)
                elif col in int_columns:
                    values.append(str(int(value)))
                else:
                    values.append(fmt(float(value)))
            handle.write(" ".join(values) + "\n")


def load_measurement_columns(path: Path) -> np.ndarray:
    data = np.loadtxt(path, comments="#")
    data = np.atleast_2d(data)
    if data.shape[1] < len(MEAS_COLUMNS):
        raise ValueError(f"{path}: expected at least {len(MEAS_COLUMNS)} columns, got {data.shape[1]}")
    return data[:, : len(MEAS_COLUMNS)]


# Average contiguous measurements so the error includes Markov-chain correlations.
def blocked_mean_error(values: np.ndarray, block_size: int) -> tuple[float, float, int, int]:
    values = np.asarray(values, dtype=float)
    n = int(values.size)
    if n == 0:
        return math.nan, math.nan, 0, 0
    block = min(max(1, int(block_size)), n)
    n_blocks = n // block
    if n_blocks < 2:
        return float(np.mean(values)), math.nan, block, n_blocks
    trimmed = values[: n_blocks * block]
    blocks = trimmed.reshape(n_blocks, block).mean(axis=1)
    return float(np.mean(blocks)), float(np.std(blocks, ddof=1) / math.sqrt(n_blocks)), block, n_blocks


# Fit y=A+B*x with inverse-variance weights; x=eta^2 in continuum applications.
def weighted_linear_fit(x: np.ndarray, y: np.ndarray, err: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    err = np.asarray(err, dtype=float)
    valid = np.isfinite(err) & (err > 0.0)
    if not np.all(valid):
        fallback = float(np.std(y, ddof=1)) if y.size > 1 else 1.0
        err = np.full_like(y, max(fallback, 1e-12))
    xmat = np.column_stack((np.ones_like(x), x))
    w = 1.0 / (err * err)
    xtw = xmat.T * w
    cov = np.linalg.inv(xtw @ xmat)
    coeff = cov @ (xtw @ y)
    resid = y - xmat @ coeff
    chi2 = float(np.sum((resid / err) ** 2))
    dof = int(y.size - 2)
    return {
        "A": float(coeff[0]),
        "A_error": math.sqrt(float(cov[0, 0])),
        "B": float(coeff[1]),
        "B_error": math.sqrt(float(cov[1, 1])),
        "chi2": chi2,
        "dof": float(dof),
        "chi2_red": chi2 / dof if dof > 0 else math.nan,
    }


def exact_map(refs: np.ndarray, obs: str) -> dict[float, float]:
    col = f"{obs}_exact"
    return {float(row["lambda"]): float(row[col]) for row in refs}
