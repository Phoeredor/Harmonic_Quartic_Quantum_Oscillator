/*
 * Project: Quantum Harmonic Oscillator PIMC
 * File: include/qho_correlator.h
 * Purpose: Public interface for Euclidean two-point correlator measurements.
 *
 * The routines declared here accumulate the translationally averaged
 * coordinate correlator
 *
 *     C_y(tau) = < y(t) y(t + tau) >
 *
 * on a periodic Euclidean-time lattice. The measured correlator is used
 * downstream to extract the first harmonic-oscillator gap from finite-beta
 * cosh fits.
 */

#ifndef QHO_CORRELATOR_H
#define QHO_CORRELATOR_H

#include "qho_lattice.h"
#include "qho_params.h"

/*
 * Accumulator for the coordinate two-point function.
 *
 * nt:
 *     Number of Euclidean time slices in the lattice.
 *
 * max_lag:
 *     Largest Euclidean-time separation, in lattice units, accumulated in
 *     sum_yy. The array stores lags tau = 0, ..., max_lag.
 *
 * sum_yy:
 *     Running sums of the translationally averaged products y_j y_{j+tau}.
 *     Periodic boundary conditions are used when j + tau wraps around nt.
 *
 * sum_y:
 *     Running sum of the path average of y. This is useful as a consistency
 *     check of the parity-symmetric harmonic oscillator, where <y> should
 *     vanish within statistical uncertainty.
 *
 * n_measurements:
 *     Number of saved configurations accumulated into the correlator.
 */
typedef struct {
    int nt;
    int max_lag;
    double *sum_yy;
    double sum_y;
    unsigned long long n_measurements;
} qho_y_correlator_t;

/*
 * Allocate and initialize a coordinate-correlator accumulator.
 *
 * Returns 0 on success and a nonzero value if the input parameters are invalid
 * or if memory allocation fails.
 */
int qho_y_correlator_init(qho_y_correlator_t *corr, int nt, int max_lag);

/*
 * Release the memory owned by the correlator accumulator and reset its fields.
 */
void qho_y_correlator_free(qho_y_correlator_t *corr);

/*
 * Accumulate one saved lattice configuration into the coordinate correlator.
 *
 * The contribution is averaged over all Euclidean time origins before being
 * added to the running sums.
 */
void qho_y_correlator_accumulate(qho_y_correlator_t *corr, const qho_lattice_t *lat);

/*
 * Write the normalized correlator and run metadata to a text file.
 *
 * The output is intended for the Python analysis scripts that perform
 * effective-mass estimates and finite-beta cosh fits.
 *
 * Returns 0 on success and a nonzero value on I/O or consistency errors.
 */
int qho_y_correlator_write(
    const qho_y_correlator_t *corr,
    const char *path,
    const qho_params_t *params
);

#endif
