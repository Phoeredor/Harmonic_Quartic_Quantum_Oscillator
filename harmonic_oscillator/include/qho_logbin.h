/*
 * Project: Quantum Harmonic Oscillator PIMC
 * File: include/qho_logbin.h
 * Purpose: Public interface for logarithmic binning of Monte Carlo time series.
 *
 * This module accumulates measurements in logarithmically growing sweep
 * intervals. The resulting binned time series is useful for checking the
 * approach to stationarity from different initial paths and for visualizing
 * slow relaxation modes.
 *
 * The binning is performed online: measurements are added to the current bin,
 * and the bin is written once its sweep range is complete.
 */

#ifndef QHO_LOGBIN_H
#define QHO_LOGBIN_H

#include <stdio.h>

#include "qho_measure.h"
#include "qho_params.h"

/* -------------------------------------------------------------------------
 * Section: logarithmic-bin accumulator
 * ------------------------------------------------------------------------- */

/*
 * Accumulator for one logarithmic Monte Carlo bin.
 *
 * start, end:
 *     Inclusive lower edge and exclusive upper edge of the sweep interval
 *     represented by the current bin.
 *
 * count:
 *     Number of saved measurements accumulated in the current bin.
 *
 * sum_y, sumsq_y:
 *     Sum and sum of squares of the path average of y. These are used to write
 *     the bin mean and its within-bin fluctuation.
 *
 * sum_y2, sumsq_y2:
 *     Sum and sum of squares of the path average of y^2.
 *
 * sum_dy2, sumsq_dy2:
 *     Sum and sum of squares of the path average of
 *     (y_{j+1} - y_j)^2. This quantity enters kinetic-energy estimators and is
 *     sensitive to short-distance lattice fluctuations.
 */
typedef struct {
    long start;
    long end;
    long count;
    double sum_y;
    double sumsq_y;
    double sum_y2;
    double sumsq_y2;
    double sum_dy2;
    double sumsq_dy2;
} qho_logbin_accumulator_t;

/* -------------------------------------------------------------------------
 * Section: logarithmic-binning interface
 * ------------------------------------------------------------------------- */

/*
 * Initialize the logarithmic-bin accumulator.
 *
 * The parameter base controls the growth of the sweep intervals. A larger
 * base produces wider bins at large Monte Carlo times.
 */
void qho_logbin_init(qho_logbin_accumulator_t *acc, double base);

/*
 * Add one saved measurement to the logarithmic-bin stream.
 *
 * If the current measurement completes a bin, the completed bin is written to
 * the output stream and the accumulator is advanced to the next logarithmic
 * interval.
 *
 * sweep:
 *     Monte Carlo sweep index of the saved measurement.
 *
 * m:
 *     Measured observables for the saved configuration.
 *
 * base:
 *     Growth factor used to define the logarithmic bin boundaries.
 *
 * out:
 *     Already opened text stream receiving the binned output.
 *
 * init_name:
 *     Label identifying the initial condition of the Markov chain, for example
 *     a cold or random initial path. The label is written to the output table.
 *
 * Returns 0 on success and a nonzero value on I/O or consistency errors.
 */
int qho_logbin_accumulate(
    qho_logbin_accumulator_t *acc,
    long sweep,
    const qho_measurements_t *m,
    double base,
    FILE *out,
    const char *init_name
);

/*
 * Write the current partially filled bin, if it contains measurements.
 *
 * This is called at the end of a run so that the final incomplete bin is not
 * discarded.
 *
 * Returns 0 on success and a nonzero value on I/O or consistency errors.
 */
int qho_logbin_flush(
    qho_logbin_accumulator_t *acc,
    double base,
    FILE *out,
    const char *init_name
);

/*
 * Write metadata and column names for the logarithmic-bin output file.
 *
 * The header records the simulation parameters needed to interpret the binned
 * Monte Carlo time series.
 *
 * Returns 0 on success and a nonzero value on I/O error.
 */
int qho_logbin_write_header(FILE *out, const qho_params_t *params);

#endif
