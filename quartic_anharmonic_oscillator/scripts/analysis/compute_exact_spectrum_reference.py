#!/usr/bin/env python3
# Diagonalize H=p^2/2+y^2/2+lambda*y^4 in a truncated harmonic basis to obtain
# reference energies and gaps for the Euclidean-correlator spectrum analysis.
"""Exact-diagonalization reference for the quartic anharmonic oscillator."""

import argparse
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lambdas", nargs="+", type=float, required=True)
    parser.add_argument("--basis-small", type=int, default=120)
    parser.add_argument("--basis-large", type=int, default=160)
    parser.add_argument("--output", type=Path, default=Path(
        "data/processed/spectrum/anharmonic_spectrum_exact_reference.dat"))
    parser.add_argument("--summary", type=Path, default=Path(
        "data/processed/spectrum/anharmonic_spectrum_exact_reference_summary.md"))
    return parser.parse_args()


# Return the five lowest dimensionless energies at coupling lam and basis cutoff.
def energies(lam, basis_size):
    raising = np.zeros((basis_size, basis_size))
    n = np.arange(1, basis_size)
    raising[n, n - 1] = np.sqrt(n)
    y = (raising + raising.T) / np.sqrt(2.0)
    y2 = y @ y
    hamiltonian = np.diag(np.arange(basis_size) + 0.5) + lam * (y2 @ y2)
    return np.linalg.eigvalsh(hamiltonian)[:5]


def main():
    args = parse_args()
    if args.basis_small < 5 or args.basis_large <= args.basis_small:
        raise SystemExit("require basis-large > basis-small >= 5")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for lam in args.lambdas:
        if lam < 0.0:
            raise SystemExit("lambda must be non-negative")
        small = energies(lam, args.basis_small)
        large = energies(lam, args.basis_large)
        gaps = large[1:] - large[0]
        convergence = np.max(np.abs(large - small))
        rows.append((lam, large, gaps, convergence))
    with args.output.open("w", encoding="utf-8") as stream:
        stream.write("# lambda E0 E1 E2 E3 E4 Delta1 Delta2 Delta3 Delta4 "
                     "basis_small basis_large max_abs_E_diff\n")
        for lam, energy, gaps, convergence in rows:
            values = [lam, *energy, *gaps]
            stream.write(" ".join(f"{value:.17g}" for value in values))
            stream.write(f" {args.basis_small:d} {args.basis_large:d} {convergence:.17g}\n")
    with args.summary.open("w", encoding="utf-8") as stream:
        stream.write("# Exact spectrum reference\n\n")
        stream.write("Hamiltonian: `H = p^2/2 + y^2/2 + lambda*y^4`, diagonalized in "
                     "the harmonic-oscillator basis.\n\n")
        stream.write("| lambda | E0 | Delta1 | Delta2 | Delta3 | Delta4 | max |E(160)-E(120)| |\n")
        stream.write("|---:|---:|---:|---:|---:|---:|---:|\n")
        for lam, energy, gaps, convergence in rows:
            stream.write(f"| {lam:g} | {energy[0]:.10g} | {gaps[0]:.10g} | "
                         f"{gaps[1]:.10g} | {gaps[2]:.10g} | {gaps[3]:.10g} | "
                         f"{convergence:.3e} |\n")
    print(f"wrote {args.output}")
    print(f"wrote {args.summary}")


if __name__ == "__main__":
    main()
