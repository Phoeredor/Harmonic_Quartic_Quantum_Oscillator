/*
 * Project: Quantum Harmonic Oscillator PIMC
 * File: include/qho_params.h
 * Purpose: Public interface for simulation parameters and command-line parsing.
 *
 * This module defines the complete set of run parameters used by the QHO PIMC
 * executable. The parameter structure contains the physical lattice setup,
 * Monte Carlo lengths, random-number streams, update choices, and optional
 * output channels for histograms, correlators, spectrum measurements, path
 * snapshots, and logarithmic-bin histories.
 */

#ifndef QHO_PARAMS_H
#define QHO_PARAMS_H

#include <stdint.h>
#include <stdio.h>

/* -------------------------------------------------------------------------
 * Section: fixed path-size convention
 * ------------------------------------------------------------------------- */

/*
 * Maximum length, including the terminating null byte, reserved for output
 * paths stored inside qho_params_t.
 */
#define QHO_PATH_MAX 4096

/* -------------------------------------------------------------------------
 * Section: initialization and algorithm choices
 * ------------------------------------------------------------------------- */

/*
 * Initial Euclidean path used at the beginning of the Markov chain.
 *
 * QHO_INIT_ZERO:
 *     Cold start with y_j = 0 for all Euclidean-time slices.
 *
 * QHO_INIT_RANDOM:
 *     Random start using the implementation-defined default random range.
 *
 * QHO_INIT_UNIFORM:
 *     Uniform start used for controlled thermalization comparisons.
 */
typedef enum {
    QHO_INIT_ZERO = 0,
    QHO_INIT_RANDOM = 1,
    QHO_INIT_UNIFORM = 2
} qho_init_t;

/*
 * Local Markov-chain update applied to the Euclidean path.
 *
 * QHO_UPDATE_METRO:
 *     Local Metropolis proposal controlled by the proposal scale delta.
 *
 * QHO_UPDATE_HEATBATH:
 *     Exact local Gaussian heatbath update for the quadratic action.
 *
 * QHO_UPDATE_HB_OVER:
 *     Heatbath update followed by deterministic overrelaxation sweeps.
 */
typedef enum {
    QHO_UPDATE_METRO = 0,
    QHO_UPDATE_HEATBATH = 1,
    QHO_UPDATE_HB_OVER = 2
} qho_update_mode_t;

/*
 * Main time-series output format.
 *
 * QHO_OUTPUT_DAT:
 *     Human-readable text output.
 *
 * QHO_OUTPUT_BIN:
 *     Fixed-size binary output used for larger production runs.
 *
 * QHO_OUTPUT_NONE:
 *     Disable the main measurement time-series output while allowing optional
 *     outputs such as histograms, correlators, or path snapshots.
 */
typedef enum {
    QHO_OUTPUT_DAT = 0,
    QHO_OUTPUT_BIN = 1,
    QHO_OUTPUT_NONE = 2
} qho_output_format_t;

/* -------------------------------------------------------------------------
 * Section: complete simulation parameter set
 * ------------------------------------------------------------------------- */

/*
 * Complete parameter set for one QHO PIMC run.
 *
 * Physical lattice:
 *
 * nt:
 *     Number of Euclidean time slices.
 *
 * beta:
 *     Inverse temperature in units where hbar omega = 1.
 *
 * eta:
 *     Euclidean lattice spacing. Normally eta = beta / nt. If eta is supplied
 *     explicitly, the parser adjusts the remaining lattice parameters according
 *     to the command-line convention implemented in qho_params_parse_args.
 *
 * Monte Carlo lengths:
 *
 * n_therm:
 *     Number of initial sweeps discarded before measurements start.
 *
 * n_sweeps:
 *     Number of post-thermalization sweeps.
 *
 * meas_stride:
 *     Number of sweeps between two saved measurements.
 *
 * Random-number stream:
 *
 * seed, stream:
 *     PCG32 seed and stream selector. Independent runs should use distinct
 *     seed-stream pairs.
 *
 * Update algorithm:
 *
 * delta:
 *     Metropolis proposal scale. It is relevant for QHO_UPDATE_METRO and kept
 *     in the parameter set also when rejection-free updates are selected.
 *
 * init:
 *     Initial path prescription.
 *
 * update_mode:
 *     Local update algorithm used by the Markov chain.
 *
 * n_overrelax:
 *     Number of overrelaxation sweeps applied after each heatbath sweep in
 *     QHO_UPDATE_HB_OVER mode.
 *
 * Main output:
 *
 * out_path:
 *     Path for the main measurement time series.
 *
 * output_format:
 *     Format of the main time-series output.
 *
 * Histogram output:
 *
 * hist_enabled:
 *     Enable the accumulated position histogram.
 *
 * hist_out:
 *     Output path for the accumulated position density.
 *
 * hist_bins:
 *     Number of histogram bins.
 *
 * hist_min, hist_max:
 *     Lower and upper edges of the histogram range.
 *
 * hist_bin_width:
 *     Histogram bin width. It is derived from the range and number of bins
 *     unless explicitly set by the user.
 *
 * hist_bin_width_explicit:
 *     Nonzero if the bin width was supplied on the command line.
 *
 * Block-histogram output:
 *
 * hist_block_enabled:
 *     Enable block-resolved position histograms and moment estimates.
 *
 * hist_block_out:
 *     Output path for the block-histogram table.
 *
 * hist_block_size_saved:
 *     Number of saved configurations per histogram block.
 *
 * Coordinate correlator output:
 *
 * corr_enabled:
 *     Enable measurement of the coordinate two-point correlator C_y(tau).
 *
 * corr_out:
 *     Output path for the coordinate-correlator table.
 *
 * corr_max_lag:
 *     Largest correlator separation, in lattice units, stored in corr_out.
 *
 * corr_max_lag_explicit:
 *     Nonzero if corr_max_lag was supplied explicitly by the user.
 *
 * Spectrum output:
 *
 * spectrum_enabled:
 *     Enable the spectrum-observable correlator output.
 *
 * spectrum_out:
 *     Output path for spectrum correlators.
 *
 * spectrum_max_lag:
 *     Largest spectrum-correlator separation, in lattice units.
 *
 * spectrum_max_lag_explicit:
 *     Nonzero if spectrum_max_lag was supplied explicitly by the user.
 *
 * spectrum_block_enabled:
 *     Enable block-resolved spectrum observables.
 *
 * spectrum_block_out:
 *     Output path for block-resolved spectrum data.
 *
 * spectrum_block_size_saved:
 *     Number of saved configurations per spectrum block.
 *
 * Path snapshot output:
 *
 * path_enabled:
 *     Enable writing representative Euclidean paths.
 *
 * path_out:
 *     Output path for path snapshots.
 *
 * Lattice-spacing parsing:
 *
 * eta_explicit:
 *     Nonzero if eta was supplied explicitly on the command line.
 *
 * Logarithmic-bin output:
 *
 * therm_logbin_enabled:
 *     Enable logarithmic-bin output for thermodynamic histories.
 *
 * therm_logbin_out:
 *     Output path for logarithmic-bin thermodynamic histories.
 *
 * logbin_base:
 *     Growth base for logarithmic bin boundaries.
 */
typedef struct {
    int nt;
    double beta;
    double eta;
    long n_therm;
    long n_sweeps;
    long meas_stride;
    uint64_t seed;
    uint64_t stream;
    double delta;
    qho_init_t init;
    qho_update_mode_t update_mode;
    int n_overrelax;
    char out_path[QHO_PATH_MAX];
    int hist_enabled;
    char hist_out[QHO_PATH_MAX];
    int hist_bins;
    double hist_min;
    double hist_max;
    double hist_bin_width;
    int hist_bin_width_explicit;
    int hist_block_enabled;
    char hist_block_out[QHO_PATH_MAX];
    int hist_block_size_saved;
    int corr_enabled;
    char corr_out[QHO_PATH_MAX];
    int corr_max_lag;
    int corr_max_lag_explicit;
    int spectrum_enabled;
    char spectrum_out[QHO_PATH_MAX];
    int spectrum_max_lag;
    int spectrum_max_lag_explicit;
    int spectrum_block_enabled;
    char spectrum_block_out[QHO_PATH_MAX];
    int spectrum_block_size_saved;
    int path_enabled;
    char path_out[QHO_PATH_MAX];
    int eta_explicit;
    qho_output_format_t output_format;
    int therm_logbin_enabled;
    char therm_logbin_out[QHO_PATH_MAX];
    double logbin_base;
} qho_params_t;

/* -------------------------------------------------------------------------
 * Section: parameter construction and command-line parsing
 * ------------------------------------------------------------------------- */

/*
 * Return a parameter structure initialized with the executable defaults.
 */
qho_params_t qho_params_default(void);

/*
 * Parse command-line arguments and update the parameter structure.
 *
 * The function also checks consistency relations among parameters, such as the
 * relation between beta, nt, and eta, output options, and enabled measurement
 * channels.
 *
 * Returns 0 on success and a nonzero value if parsing or consistency checks
 * fail.
 */
int qho_params_parse_args(qho_params_t *params, int argc, char **argv);

/*
 * Print the resolved parameter set to the selected stream.
 *
 * This is useful for recording the exact run configuration in logs and output
 * metadata.
 */
void qho_params_print(const qho_params_t *params, FILE *stream);

/*
 * Print command-line usage information.
 */
void qho_params_print_usage(FILE *stream, const char *program_name);

/* -------------------------------------------------------------------------
 * Section: enum-to-string helpers
 * ------------------------------------------------------------------------- */

/*
 * Return a stable text label for an initial-path choice.
 */
const char *qho_init_name(qho_init_t init);

/*
 * Return a stable text label for an update algorithm.
 */
const char *qho_update_mode_name(qho_update_mode_t update_mode);

/*
 * Return a stable text label for a main output format.
 */
const char *qho_output_format_name(qho_output_format_t output_format);

#endif
