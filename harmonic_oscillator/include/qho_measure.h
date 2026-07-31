/*
 * Project: Quantum Harmonic Oscillator PIMC
 * File: include/qho_measure.h
 * Purpose: Public interface for thermodynamic measurements.
 *
 * This module defines the observables measured on one Euclidean path of the
 * quantum harmonic oscillator. The measurements are local in the Markov chain:
 * each call evaluates path averages on the current lattice configuration.
 *
 * The lattice spacing eta is stored in the run parameters and enters the
 * kinetic-energy estimator.
 */

#ifndef QHO_MEASURE_H
#define QHO_MEASURE_H

#include "qho_lattice.h"
#include "qho_params.h"

/* -------------------------------------------------------------------------
 * Section: measured observables
 * ------------------------------------------------------------------------- */

/*
 * Thermodynamic measurements evaluated on one Euclidean path.
 *
 * y_mean:
 *     Path average of the coordinate,
 *
 *         y_mean = (1 / Nt) sum_j y_j.
 *
 *     For the parity-symmetric harmonic oscillator its ensemble average should
 *     vanish within statistical uncertainty.
 *
 * y2_mean:
 *     Path average of the squared coordinate,
 *
 *         y2_mean = (1 / Nt) sum_j y_j^2.
 *
 *     Its ensemble average estimates the thermal expectation value <y^2>.
 *
 * dy2_mean:
 *     Path average of nearest-neighbor squared differences,
 *
 *         dy2_mean = (1 / Nt) sum_j (y_{j+1} - y_j)^2,
 *
 *     with periodic boundary conditions. This quantity enters the kinetic
 *     energy estimator and is sensitive to the Euclidean lattice spacing.
 *
 * potential:
 *     Potential-energy estimator for the harmonic oscillator,
 *
 *         V = 0.5 * y2_mean.
 *
 * kinetic_ren:
 *     Renormalized kinetic-energy estimator. The naive kinetic contribution
 *     contains an additive lattice divergence; the renormalized estimator
 *     subtracts the corresponding free short-distance term.
 *
 * energy_ren:
 *     Renormalized total-energy estimator,
 *
 *         H_ren = kinetic_ren + potential.
 */
typedef struct {
    double y_mean;
    double y2_mean;
    double dy2_mean;
    double potential;
    double kinetic_ren;
    double energy_ren;
} qho_measurements_t;

/* -------------------------------------------------------------------------
 * Section: measurement interface
 * ------------------------------------------------------------------------- */

/*
 * Measure thermodynamic observables on the current lattice configuration.
 *
 * The input lattice is not modified. The run parameters provide beta, Nt, and
 * therefore the Euclidean lattice spacing eta needed by the energy estimator.
 */
qho_measurements_t qho_measure_basic(
    const qho_lattice_t *lattice,
    const qho_params_t *params
);

/*
 * Print a compact human-readable summary of a measurement record.
 *
 * This routine is intended for short checks and interactive runs, not for
 * production data storage. Production measurements are written through the
 * binary or text-output routines.
 */
void qho_measure_print(const qho_measurements_t *measurements);

#endif
