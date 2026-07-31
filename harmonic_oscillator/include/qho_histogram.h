/*
 * Project: Quantum Harmonic Oscillator PIMC
 * File: include/qho_histogram.h
 * Purpose: Public interface for position-histogram accumulation.
 *
 * This module accumulates histograms of the Euclidean coordinate y sampled
 * along Markov-chain configurations of the quantum harmonic oscillator.
 *
 * For each saved lattice configuration, all Euclidean-time sites contribute to
 * the histogram. The resulting density estimates the thermal position
 * probability distribution P(y). Block accumulators are provided to estimate
 * statistical uncertainties and to study derived moments such as <y^2>.
 */

#ifndef QHO_HISTOGRAM_H
#define QHO_HISTOGRAM_H

#include "qho_lattice.h"
#include "qho_params.h"

/* -------------------------------------------------------------------------
 * Section: global histogram accumulator
 * ------------------------------------------------------------------------- */

/*
 * Accumulator for a single position histogram.
 *
 * bins:
 *     Number of equally spaced bins in the interval [y_min, y_max).
 *
 * y_min, y_max:
 *     Lower and upper edges of the histogram range.
 *
 * bin_width:
 *     Width of each bin, equal to (y_max - y_min) / bins.
 *
 * counts:
 *     Array of length bins. counts[k] stores the number of samples whose
 *     coordinate falls inside bin k.
 *
 * total_samples:
 *     Total number of coordinate samples accumulated inside the histogram
 *     range. For each saved lattice configuration, up to nt samples are added.
 *
 * underflow, overflow:
 *     Number of samples falling below y_min or at/above y_max. These samples
 *     are tracked but are not included in counts.
 */
typedef struct {
    int bins;
    double y_min;
    double y_max;
    double bin_width;
    unsigned long long *counts;
    unsigned long long total_samples;
    unsigned long long underflow;
    unsigned long long overflow;
} qho_histogram_t;

/* -------------------------------------------------------------------------
 * Section: block histogram accumulator
 * ------------------------------------------------------------------------- */

/*
 * Block accumulator for histogram and moment estimates.
 *
 * The Markov chain is divided into blocks of block_size_saved saved
 * configurations. Each block stores both a histogram and coordinate moments.
 * This layout is used by the Python analysis scripts to estimate statistical
 * errors with blocking or jackknife procedures.
 *
 * nt:
 *     Number of Euclidean time slices per saved configuration.
 *
 * bins, y_min, y_max, bin_width:
 *     Histogram definition shared by all blocks.
 *
 * block_size_saved:
 *     Number of saved configurations per block.
 *
 * current_measurements:
 *     Number of saved configurations already accumulated in the current block.
 *
 * current_underflow, current_overflow:
 *     Out-of-range sample counters for the current block.
 *
 * current_counts:
 *     Histogram counts for the current block. The array has length bins.
 *
 * current_sum_y, current_sum_y2, current_sum_y4:
 *     Running sums of y, y^2, and y^4 over all coordinate samples in the
 *     current block. These moments are useful for checking the sampled
 *     distribution against exact harmonic-oscillator expectations.
 *
 * n_blocks:
 *     Number of completed blocks stored in the accumulator.
 *
 * block_capacity:
 *     Number of blocks currently allocated in memory.
 *
 * block_measurements:
 *     Number of saved configurations in each completed block.
 *
 * block_underflow, block_overflow:
 *     Out-of-range sample counters for each completed block.
 *
 * block_counts:
 *     Flattened array storing the histogram counts of all completed blocks.
 *     The count of bin k in block b is stored at block_counts[b*bins + k].
 *
 * block_mean_y, block_mean_y2, block_mean_y4:
 *     Block averages of y, y^2, and y^4.
 */
typedef struct {
    int nt;
    int bins;
    double y_min;
    double y_max;
    double bin_width;
    int block_size_saved;
    unsigned long long current_measurements;
    unsigned long long current_underflow;
    unsigned long long current_overflow;
    unsigned long long *current_counts;
    double current_sum_y;
    double current_sum_y2;
    double current_sum_y4;
    unsigned long long n_blocks;
    unsigned long long block_capacity;
    unsigned long long *block_measurements;
    unsigned long long *block_underflow;
    unsigned long long *block_overflow;
    unsigned long long *block_counts;
    double *block_mean_y;
    double *block_mean_y2;
    double *block_mean_y4;
} qho_histogram_block_accumulator_t;

/* -------------------------------------------------------------------------
 * Section: single-histogram interface
 * ------------------------------------------------------------------------- */

/*
 * Allocate and initialize a histogram accumulator.
 *
 * Returns 0 on success and a nonzero value if the range is invalid, the number
 * of bins is not positive, or memory allocation fails.
 */
int qho_histogram_init(qho_histogram_t *hist, int bins, double y_min, double y_max);

/*
 * Release memory owned by a histogram accumulator and reset its fields.
 */
void qho_histogram_free(qho_histogram_t *hist);

/*
 * Accumulate one coordinate value into the histogram.
 *
 * Values below y_min increase the underflow counter. Values greater than or
 * equal to y_max increase the overflow counter.
 */
void qho_histogram_accumulate_value(qho_histogram_t *hist, double y);

/*
 * Accumulate all Euclidean-time coordinates of one saved lattice configuration.
 *
 * Each site y_j contributes one sample to the position histogram.
 */
void qho_histogram_accumulate_lattice(qho_histogram_t *hist, const qho_lattice_t *lat);

/*
 * Write the normalized position density to a text file.
 *
 * The output density is normalized so that the integral over the histogram
 * range is one, up to underflow and overflow losses. Run parameters are written
 * as metadata for reproducibility.
 *
 * Returns 0 on success and a nonzero value on I/O or consistency errors.
 */
int qho_histogram_write_density(
    const qho_histogram_t *hist,
    const char *path,
    const qho_params_t *params
);

/* -------------------------------------------------------------------------
 * Section: block-histogram interface
 * ------------------------------------------------------------------------- */

/*
 * Allocate and initialize a block histogram accumulator.
 *
 * Returns 0 on success and a nonzero value if the histogram parameters are
 * invalid or memory allocation fails.
 */
int qho_histogram_block_init(
    qho_histogram_block_accumulator_t *blocks,
    int nt,
    int bins,
    double y_min,
    double y_max,
    int block_size_saved
);

/*
 * Release memory owned by a block histogram accumulator and reset its fields.
 */
void qho_histogram_block_free(qho_histogram_block_accumulator_t *blocks);

/*
 * Accumulate one saved lattice configuration into the current block.
 *
 * When the current block reaches block_size_saved saved configurations, its
 * histogram and moment averages are stored as one completed block.
 *
 * Returns 0 on success and a nonzero value on allocation or consistency errors.
 */
int qho_histogram_block_accumulate(
    qho_histogram_block_accumulator_t *blocks,
    const qho_lattice_t *lat
);

/*
 * Write all completed block histograms and block moments to a text file.
 *
 * The output is intended for downstream uncertainty analysis of P(y), <y^2>,
 * and higher moments.
 *
 * Returns 0 on success and a nonzero value on I/O or consistency errors.
 */
int qho_histogram_block_write(
    qho_histogram_block_accumulator_t *blocks,
    const char *path,
    const qho_params_t *params
);

#endif
