#!/usr/bin/env python3
# Extrapolate potential and kinetic estimators at beta=5 and test the continuum
# virial relation K=<y V'(y)>/2.  Shared block resampling preserves correlations
# when estimating K_ren-K_vir for each lambda.
"""Fit final potential, virial, and renormalized kinetic observables."""

from __future__ import annotations

import numpy as np

from quartic_continuum_common import (
    DEFAULT_ETA_MAX, DEFAULT_ETA_MIN, LAMBDAS_V2, MEAS_COLUMNS, PROCESSED,
    load_measurement_columns, read_named, weighted_linear_fit, write_table,
)


POINTS = PROCESSED / "anharmonic_beta5_continuum_v2_points.dat"
DERIVED_OUT = PROCESSED / "anharmonic_beta5_derived_continuum_v2.dat"
OUT = PROCESSED / "anharmonic_beta5_virial_continuum_v2.dat"
N_BOOT = 500
BOOT_SEED = 12345
OBS_MAP = {
    "V": ("V_eta", "V_eta_err"),
    "K_vir": ("K_vir_eta", "K_vir_eta_err"),
    "K_ren": ("K_ren_eta", "K_ren_eta_err"),
}
DERIVED_COLUMNS = (
    "lambda", "obs", "eta_min", "eta_max", "n_points", "A", "A_err",
    "B", "B_err", "chi2", "dof", "chi2_red", "selection_status",
)
COLUMNS = (
    "lambda", "K_ren_cont", "K_ren_cont_err", "K_vir_cont", "K_vir_cont_err",
    "K_ren_minus_K_vir", "K_ren_minus_K_vir_err", "z_Kren_minus_Kvir",
    "delta_error_method",
)


def select_fit_window(points: np.ndarray, lam: float, obs: str) -> dict[str, float | str]:
    """Select a contiguous eta window without using reference values."""
    value_col, err_col = OBS_MAP[obs]
    sub = points[np.isclose(points["lambda"], lam)]
    sub = sub[(sub["eta"] >= DEFAULT_ETA_MIN) & (sub["eta"] <= DEFAULT_ETA_MAX)]
    sub = np.sort(sub, order="eta")
    candidates = []
    for start in range(len(sub)):
        for stop in range(start + 5, len(sub) + 1):
            win = sub[start:stop]
            fit = weighted_linear_fit(win["eta2"], win[value_col], win[err_col])
            candidates.append((win, fit))
    if not candidates:
        raise RuntimeError(f"lambda={lam:g}, obs={obs}: no fit window with at least 5 points")
    target = [(win, fit) for win, fit in candidates if 0.5 <= float(fit["chi2_red"]) < 1.0]
    if target:
        win, fit = min(target, key=lambda wf: (-len(wf[0]), abs(float(wf[1]["chi2_red"]) - 0.8), float(wf[0]["eta"][0])))
        status = "target"
    else:
        win, fit = min(candidates, key=lambda wf: (abs(float(wf[1]["chi2_red"]) - 0.8), -len(wf[0]), float(wf[0]["eta"][0])))
        status = "fallback"
    return {
        "lambda": lam,
        "obs": obs,
        "eta_min": float(win["eta"][0]),
        "eta_max": float(win["eta"][-1]),
        "n_points": int(len(win)),
        "A": float(fit["A"]),
        "A_err": float(fit["A_error"]),
        "B": float(fit["B"]),
        "B_err": float(fit["B_error"]),
        "chi2": float(fit["chi2"]),
        "dof": int(fit["dof"]),
        "chi2_red": float(fit["chi2_red"]),
        "selection_status": status,
    }


def fit_fixed_window(points: np.ndarray, lam: float, obs: str, eta_min: float, eta_max: float) -> dict[str, float]:
    """Repeat a continuum fit on a fixed eta window."""
    value_col, err_col = OBS_MAP[obs]
    sub = points[np.isclose(points["lambda"], lam)]
    sub = sub[(sub["eta"] >= eta_min - 1e-14) & (sub["eta"] <= eta_max + 1e-14)]
    sub = np.sort(sub, order="eta")
    return weighted_linear_fit(sub["eta2"], sub[value_col], sub[err_col])


def read_block_series(raw_file: str, block_size: int, eta: float, lam: float) -> dict[str, np.ndarray]:
    """Return blocked kinetic estimator series for one raw measurement file."""
    data = load_measurement_columns(raw_file)
    col = {name: i for i, name in enumerate(MEAS_COLUMNS)}
    y2 = data[:, col["y2_mean"]]
    y4 = data[:, col["y4_mean"]]
    dy2 = data[:, col["dy2_mean"]]
    samples = {
        "K_vir": 0.5 * y2 + 2.0 * lam * y4,
        "K_ren": -dy2 / (2.0 * eta * eta) + 1.0 / (2.0 * eta),
    }
    out = {}
    for obs, values in samples.items():
        block = min(max(1, int(block_size)), int(values.size))
        n_blocks = int(values.size) // block
        trimmed = values[: n_blocks * block]
        out[obs] = trimmed.reshape(n_blocks, block).mean(axis=1)
    return out


def bootstrap_delta(points: np.ndarray, derived: dict[tuple[float, str], dict[str, float | str]]) -> dict[float, float]:
    """Estimate K_ren - K_vir errors with shared per-eta block resampling."""
    rng = np.random.default_rng(BOOT_SEED)
    errors: dict[float, float] = {}
    for lam in LAMBDAS_V2:
        k_ren_fit = derived[(lam, "K_ren")]
        k_vir_fit = derived[(lam, "K_vir")]
        eta_windows = {
            "K_ren": (float(k_ren_fit["eta_min"]), float(k_ren_fit["eta_max"])),
            "K_vir": (float(k_vir_fit["eta_min"]), float(k_vir_fit["eta_max"])),
        }
        needed_min = min(v[0] for v in eta_windows.values())
        needed_max = max(v[1] for v in eta_windows.values())
        sub = points[np.isclose(points["lambda"], lam)]
        sub = sub[(sub["eta"] >= needed_min - 1e-14) & (sub["eta"] <= needed_max + 1e-14)]
        sub = np.sort(sub, order="eta")
        block_cache = {
            float(row["eta"]): read_block_series(str(row["raw_file"]), int(row["block_size_rows"]), float(row["eta"]), lam)
            for row in sub
        }
        deltas = []
        for _ in range(N_BOOT):
            boot_rows = []
            for row in sub:
                eta = float(row["eta"])
                boot_row = {name: row[name] for name in points.dtype.names}
                for obs in ("K_ren", "K_vir"):
                    blocks = block_cache[eta][obs]
                    indices = rng.integers(0, len(blocks), size=len(blocks))
                    boot_values = blocks[indices]
                    boot_row[OBS_MAP[obs][0]] = float(np.mean(boot_values))
                    boot_row[OBS_MAP[obs][1]] = float(np.std(boot_values, ddof=1) / np.sqrt(len(boot_values)))
                boot_rows.append(tuple(boot_row[name] for name in points.dtype.names))
            boot_points = np.array(boot_rows, dtype=points.dtype)
            try:
                k_ren = fit_fixed_window(boot_points, lam, "K_ren", *eta_windows["K_ren"])["A"]
                k_vir = fit_fixed_window(boot_points, lam, "K_vir", *eta_windows["K_vir"])["A"]
            except (np.linalg.LinAlgError, ValueError):
                continue
            deltas.append(float(k_ren - k_vir))
        errors[lam] = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else np.nan
    return errors


def main() -> None:
    points = read_named(POINTS)
    derived_rows = [select_fit_window(points, lam, obs) for lam in LAMBDAS_V2 for obs in ("V", "K_vir", "K_ren")]
    write_table(DERIVED_OUT, DERIVED_COLUMNS, derived_rows, int_columns={"n_points", "dof"})
    derived = {(float(row["lambda"]), str(row["obs"])): row for row in derived_rows}
    delta_boot_err = bootstrap_delta(points, derived)
    rows = []
    for lam in LAMBDAS_V2:
        k_ren_row = derived[(lam, "K_ren")]
        k_vir_row = derived[(lam, "K_vir")]
        k_ren = float(k_ren_row["A"])
        k_vir = float(k_vir_row["A"])
        diff = k_ren - k_vir
        diff_err = delta_boot_err[lam]
        rows.append({
            "lambda": lam,
            "K_ren_cont": k_ren,
            "K_ren_cont_err": float(k_ren_row["A_err"]),
            "K_vir_cont": k_vir,
            "K_vir_cont_err": float(k_vir_row["A_err"]),
            "K_ren_minus_K_vir": diff,
            "K_ren_minus_K_vir_err": diff_err,
            "z_Kren_minus_Kvir": diff / diff_err if diff_err > 0.0 else np.nan,
            "delta_error_method": f"bootstrap_blocks_n{N_BOOT}_seed{BOOT_SEED}",
        })
    write_table(OUT, COLUMNS, rows)
    print(f"wrote {DERIVED_OUT}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
