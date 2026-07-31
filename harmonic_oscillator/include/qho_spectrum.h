/*
 * Project: Quantum Harmonic Oscillator PIMC
 * File: include/qho_spectrum.h
 * Purpose: Public interface for spectrum-observable measurements.
 *
 * This module accumulates imaginary-time correlation functions used to extract
 * low-lying energy gaps of the quantum harmonic oscillator.
 *
 * The measured operators are chosen according to their parity and excitation
 * content:
 *
 *     y                         -> first odd excitation, Delta E_1 = 1
 *     y^2                       -> second even excitation, Delta E_2 = 2
 *     A(y) = y^3 - (3/2) y       -> third odd excitation, Delta E_3 = 3
 *
 * Correlators are translationally averaged over Euclidean-time origins on a
 * periodic lattice. The Python analysis scripts later perform connected
 * subtractions where needed and fit the finite-beta cosh behavior.
 */

#ifndef QHO_SPECTRUM_H
#define QHO_SPECTRUM_H

#include "qho_lattice.h"
#include "qho_params.h"

/* -------------------------------------------------------------------------
 * Section: ensemble correlator accumulator
 * ------------------------------------------------------------------------- */

/*
 * Accumulator for spectrum correlators averaged over the full Markov chain.
 *
 * nt:
 *     Number of Euclidean time slices.
 *
 * max_lag:
 *     Largest Euclidean-time separation, in lattice units, accumulated for
 *     each correlator. Arrays store lags tau = 0, ..., max_lag.
 *
 * sum_y_y:
 *     Running sum of the translationally averaged correlator
 *
 *         < y(t) y(t + tau) >.
 *
 *     This channel couples to odd states and is used to estimate the first
 *     harmonic-oscillator gap.
 *
 * sum_y2_y2:
 *     Running sum of the translationally averaged correlator
 *
 *         < y(t)^2 y(t + tau)^2 >.
 *
 *     Its connected part is used to estimate the second gap.
 *
 * sum_a_a:
 *     Running sum of the translationally averaged correlator
 *
 *         < A(t) A(t + tau) >,
 *
 *     where A(y) = y^3 - (3/2) y. This improved odd operator suppresses the
 *     overlap with the first excited state and is used for the third gap.
 *
 * sum_y, sum_y2, sum_a:
 *     Running sums of the path averages of the three operators. They are used
 *     to monitor one-point functions and to form connected correlators in the
 *     downstream analysis.
 *
 * n_measurements:
 *     Number of saved lattice configurations accumulated.
 */
typedef struct {
    int nt;
    int max_lag;
    double *sum_y_y;
    double *sum_y2_y2;
    double *sum_a_a;
    double sum_y;
    double sum_y2;
    double sum_a;
    unsigned long long n_measurements;
} qho_spectrum_correlator_t;

/* -------------------------------------------------------------------------
 * Section: block correlator accumulator
 * ------------------------------------------------------------------------- */

/*
 * Block accumulator for spectrum correlators.
 *
 * The Markov chain is divided into blocks of block_size_saved saved
 * configurations. Each completed block stores operator means and raw
 * correlators. The Python analysis scripts use these block data for jackknife
 * estimates, connected subtractions, effective gaps, and continuum fits.
 *
 * nt:
 *     Number of Euclidean time slices.
 *
 * max_lag:
 *     Largest Euclidean-time separation stored for each block correlator.
 *
 * block_size_saved:
 *     Number of saved configurations per completed block.
 *
 * current_measurements:
 *     Number of saved configurations already accumulated in the current block.
 *
 * n_blocks:
 *     Number of completed blocks stored.
 *
 * block_capacity:
 *     Number of blocks currently allocated in memory.
 *
 * block_measurements:
 *     Number of saved configurations in each completed block.
 *
 * block_mean_y, block_mean_y2, block_mean_y3, block_mean_a:
 *     Block averages of y, y^2, y^3, and A(y). These are needed to monitor
 *     operator means and to form connected correlators where appropriate.
 *
 * block_raw_y_y:
 *     Flattened block array for the raw y-y correlator.
 *
 * block_raw_y2_y2:
 *     Flattened block array for the raw y^2-y^2 correlator.
 *
 * block_raw_y3_y3:
 *     Auxiliary raw y^3-y^3 correlator. It is kept to compare the unimproved
 *     cubic channel with the improved A(y)-A(y) channel.
 *
 * block_raw_a_a:
 *     Flattened block array for the raw A-A correlator.
 *
 * current_sum_y, current_sum_y2, current_sum_y3, current_sum_a:
 *     Running operator sums for the current block.
 *
 * current_sum_y_y, current_sum_y2_y2, current_sum_y3_y3, current_sum_a_a:
 *     Running raw correlator sums for the current block.
 *
 * Flattened correlator arrays use the index b*(max_lag + 1) + tau, where b is
 * the block index and tau is the Euclidean-time lag in lattice units.
 */
typedef struct {
    int nt;
    int max_lag;
    int block_size_saved;
    unsigned long long current_measurements;
    unsigned long long n_blocks;
    unsigned long long block_capacity;
    unsigned long long *block_measurements;
    double *block_mean_y;
    double *block_mean_y2;
    double *block_mean_y3;
    double *block_mean_a;
    double *block_raw_y_y;
    double *block_raw_y2_y2;
    double *block_raw_y3_y3;
    double *block_raw_a_a;
    double current_sum_y;
    double current_sum_y2;
    double current_sum_y3;
    double current_sum_a;
    double *current_sum_y_y;
    double *current_sum_y2_y2;
    double *current_sum_y3_y3;
    double *current_sum_a_a;
} qho_spectrum_block_accumulator_t;

/* -------------------------------------------------------------------------
 * Section: ensemble correlator interface
 * ------------------------------------------------------------------------- */

/*
 * Allocate and initialize a spectrum-correlator accumulator.
 *
 * Returns 0 on success and a nonzero value if the input parameters are invalid
 * or if memory allocation fails.
 */
int qho_spectrum_correlator_init(qho_spectrum_correlator_t *corr, int nt, int max_lag);

/*
 * Release memory owned by a spectrum-correlator accumulator and reset it.
 */
void qho_spectrum_correlator_free(qho_spectrum_correlator_t *corr);

/*
 * Accumulate one saved lattice configuration into the spectrum correlators.
 *
 * The contribution is averaged over all Euclidean-time origins before being
 * added to the running sums. Periodic boundary conditions are used for
 * time-origin wrapping.
 */
void qho_spectrum_correlator_accumulate(qho_spectrum_correlator_t *corr, const qho_lattice_t *lat);

/*
 * Write normalized spectrum correlators and run metadata to a text file.
 *
 * Returns 0 on success and a nonzero value on I/O or consistency errors.
 */
int qho_spectrum_correlator_write(
    const qho_spectrum_correlator_t *corr,
    const char *path,
    const qho_params_t *params
);

/* -------------------------------------------------------------------------
 * Section: block correlator interface
 * ------------------------------------------------------------------------- */

/*
 * Allocate and initialize a block spectrum-correlator accumulator.
 *
 * Returns 0 on success and a nonzero value if the input parameters are invalid
 * or if memory allocation fails.
 */
int qho_spectrum_block_accumulator_init(
    qho_spectrum_block_accumulator_t *blocks,
    int nt,
    int max_lag,
    int block_size_saved
);

/*
 * Release memory owned by a block spectrum-correlator accumulator and reset it.
 */
void qho_spectrum_block_accumulator_free(qho_spectrum_block_accumulator_t *blocks);

/*
 * Accumulate one saved lattice configuration into the current spectrum block.
 *
 * When current_measurements reaches block_size_saved, the current block is
 * finalized and stored.
 *
 * Returns 0 on success and a nonzero value on allocation or consistency errors.
 */
int qho_spectrum_block_accumulator_accumulate(
    qho_spectrum_block_accumulator_t *blocks,
    const qho_lattice_t *lat
);

/*
 * Write all completed block spectrum data to a text file.
 *
 * The output is intended for jackknife uncertainty estimates and continuum
 * extrapolations of the extracted energy gaps.
 *
 * Returns 0 on success and a nonzero value on I/O or consistency errors.
 */
int qho_spectrum_block_accumulator_write(
    qho_spectrum_block_accumulator_t *blocks,
    const char *path,
    const qho_params_t *params
);

#endif
