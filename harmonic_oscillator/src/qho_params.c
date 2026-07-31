/*
 * Project: Quantum Harmonic Oscillator PIMC
 * File: src/qho_params.c
 * Purpose: Command-line parameter parsing and consistency-checking routines.
 * It establishes beta = Nt eta, Markov-chain lengths, update choices, and the
 * observable channels used by the Euclidean path-integral calculation.
 */

/* -------------------------------------------------------------------------
 * Section: includes and implementation
 * ------------------------------------------------------------------------- */

#include "qho_params.h"

#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static int parse_int_value(const char *text, int *value)
{
    char *end = NULL;
    errno = 0;
    const long parsed = strtol(text, &end, 10);

    if (errno != 0 || end == text || *end != '\0' || parsed < INT_MIN || parsed > INT_MAX) {
        return -1;
    }

    *value = (int)parsed;
    return 0;
}

static int parse_long_value(const char *text, long *value)
{
    char *end = NULL;
    errno = 0;
    const long parsed = strtol(text, &end, 10);

    if (errno != 0 || end == text || *end != '\0') {
        return -1;
    }

    *value = parsed;
    return 0;
}

static int parse_u64_value(const char *text, uint64_t *value)
{
    char *end = NULL;
    errno = 0;
    const unsigned long long parsed = strtoull(text, &end, 10);

    if (errno != 0 || end == text || *end != '\0') {
        return -1;
    }

    *value = (uint64_t)parsed;
    return 0;
}

static int parse_double_value(const char *text, double *value)
{
    char *end = NULL;
    errno = 0;
    const double parsed = strtod(text, &end);

    if (errno != 0 || end == text || *end != '\0' || !isfinite(parsed)) {
        return -1;
    }

    *value = parsed;
    return 0;
}

static int copy_path(char *dest, size_t dest_size, const char *path)
{
    const size_t len = strlen(path);

    if (len == 0 || len >= dest_size) {
        return -1;
    }

    memcpy(dest, path, len + 1);
    return 0;
}

static int copy_output_path(qho_params_t *params, const char *path)
{
    return copy_path(params->out_path, sizeof(params->out_path), path);
}

static int copy_histogram_path(qho_params_t *params, const char *path)
{
    return copy_path(params->hist_out, sizeof(params->hist_out), path);
}

static int copy_histogram_block_path(qho_params_t *params, const char *path)
{
    return copy_path(params->hist_block_out, sizeof(params->hist_block_out), path);
}

static int copy_correlator_path(qho_params_t *params, const char *path)
{
    return copy_path(params->corr_out, sizeof(params->corr_out), path);
}

static int copy_spectrum_path(qho_params_t *params, const char *path)
{
    return copy_path(params->spectrum_out, sizeof(params->spectrum_out), path);
}

static int copy_spectrum_block_path(qho_params_t *params, const char *path)
{
    return copy_path(params->spectrum_block_out, sizeof(params->spectrum_block_out), path);
}

static int copy_path_snapshot_path(qho_params_t *params, const char *path)
{
    return copy_path(params->path_out, sizeof(params->path_out), path);
}

static int copy_therm_logbin_path(qho_params_t *params, const char *path)
{
    return copy_path(params->therm_logbin_out, sizeof(params->therm_logbin_out), path);
}

static int parse_init_value(const char *text, qho_init_t *init)
{
    if (strcmp(text, "zero") == 0) {
        *init = QHO_INIT_ZERO;
        return 0;
    }

    if (strcmp(text, "random") == 0) {
        *init = QHO_INIT_RANDOM;
        return 0;
    }

    if (strcmp(text, "uniform") == 0) {
        *init = QHO_INIT_UNIFORM;
        return 0;
    }

    return -1;
}

static int parse_update_value(const char *text, qho_update_mode_t *update_mode)
{
    if (strcmp(text, "metro") == 0) {
        *update_mode = QHO_UPDATE_METRO;
        return 0;
    }

    if (strcmp(text, "heatbath") == 0) {
        *update_mode = QHO_UPDATE_HEATBATH;
        return 0;
    }

    if (strcmp(text, "hb-over") == 0) {
        *update_mode = QHO_UPDATE_HB_OVER;
        return 0;
    }

    return -1;
}

static int parse_output_format_value(const char *text, qho_output_format_t *output_format)
{
    if (strcmp(text, "dat") == 0 || strcmp(text, "ascii") == 0) {
        *output_format = QHO_OUTPUT_DAT;
        return 0;
    }
    if (strcmp(text, "bin") == 0 || strcmp(text, "binary") == 0) {
        *output_format = QHO_OUTPUT_BIN;
        return 0;
    }
    if (strcmp(text, "none") == 0) {
        *output_format = QHO_OUTPUT_NONE;
        return 0;
    }
    return -1;
}

/* Require a consistent periodic lattice and physically admissible run parameters. */
static int check_params(const qho_params_t *params)
{
    if (params->nt <= 1) {
        fprintf(stderr, "error: --nt must be greater than 1\n");
        return -1;
    }
    if (params->beta <= 0.0) {
        fprintf(stderr, "error: --beta must be positive\n");
        return -1;
    }
    if (params->eta <= 0.0) {
        fprintf(stderr, "error: --eta must be positive\n");
        return -1;
    }
    if (params->n_therm < 0L) {
        fprintf(stderr, "error: --therm must be non-negative\n");
        return -1;
    }
    if (params->n_sweeps <= 0L) {
        fprintf(stderr, "error: --sweeps must be positive\n");
        return -1;
    }
    if (params->meas_stride <= 0L) {
        fprintf(stderr, "error: --stride must be positive\n");
        return -1;
    }
    if (params->delta <= 0.0) {
        fprintf(stderr, "error: --delta must be positive\n");
        return -1;
    }
    if (params->n_overrelax < 0) {
        fprintf(stderr, "error: --n-over must be non-negative\n");
        return -1;
    }
    if (params->out_path[0] == '\0') {
        fprintf(stderr, "error: --out must not be empty\n");
        return -1;
    }
    if (params->hist_bins <= 0) {
        fprintf(stderr, "error: --hist-bins must be positive\n");
        return -1;
    }
    if (!(params->hist_min < params->hist_max)) {
        fprintf(stderr, "error: --hist-min must be smaller than --hist-max\n");
        return -1;
    }
    if (params->hist_enabled && params->hist_out[0] == '\0') {
        fprintf(stderr, "error: --hist-out must not be empty\n");
        return -1;
    }
    if (params->hist_block_enabled && params->hist_block_out[0] == '\0') {
        fprintf(stderr, "error: --hist-block-out must not be empty\n");
        return -1;
    }
    if (params->hist_block_enabled && params->hist_block_size_saved <= 0) {
        fprintf(stderr, "error: --hist-block-size-saved must be positive\n");
        return -1;
    }
    if (params->corr_enabled && params->corr_out[0] == '\0') {
        fprintf(stderr, "error: --corr-out must not be empty\n");
        return -1;
    }
    if (params->corr_enabled && params->corr_max_lag <= 0) {
        fprintf(stderr, "error: --corr-max-lag must be positive\n");
        return -1;
    }
    if (params->corr_enabled && params->corr_max_lag >= params->nt) {
        fprintf(stderr, "error: --corr-max-lag must be smaller than --nt\n");
        return -1;
    }
    if (params->spectrum_enabled && params->spectrum_out[0] == '\0') {
        fprintf(stderr, "error: --spectrum-out must not be empty\n");
        return -1;
    }
    if (params->spectrum_block_enabled && params->spectrum_block_out[0] == '\0') {
        fprintf(stderr, "error: --spectrum-block-out must not be empty\n");
        return -1;
    }
    if (params->spectrum_block_enabled && params->spectrum_block_size_saved <= 0) {
        fprintf(stderr, "error: --spectrum-block-size-saved must be positive\n");
        return -1;
    }
    if ((params->spectrum_enabled || params->spectrum_block_enabled) && params->spectrum_max_lag <= 0) {
        fprintf(stderr, "error: --spectrum-max-lag must be positive\n");
        return -1;
    }
    if ((params->spectrum_enabled || params->spectrum_block_enabled) && params->spectrum_max_lag >= params->nt) {
        fprintf(stderr, "error: --spectrum-max-lag must be smaller than --nt\n");
        return -1;
    }
    if (params->path_enabled && params->path_out[0] == '\0') {
        fprintf(stderr, "error: --path-out must not be empty\n");
        return -1;
    }
    if (params->therm_logbin_enabled && params->therm_logbin_out[0] == '\0') {
        fprintf(stderr, "error: --therm-logbin-out must not be empty\n");
        return -1;
    }
    if (params->logbin_base <= 1.0 || !isfinite(params->logbin_base)) {
        fprintf(stderr, "error: --logbin-base must be finite and greater than 1\n");
        return -1;
    }

    return 0;
}

const char *qho_init_name(qho_init_t init)
{
    if (init == QHO_INIT_RANDOM) {
        return "random";
    }
    if (init == QHO_INIT_UNIFORM) {
        return "uniform";
    }
    return "zero";
}

const char *qho_update_mode_name(qho_update_mode_t update_mode)
{
    if (update_mode == QHO_UPDATE_HEATBATH) {
        return "heatbath";
    }
    if (update_mode == QHO_UPDATE_HB_OVER) {
        return "hb-over";
    }
    return "metro";
}

const char *qho_output_format_name(qho_output_format_t output_format)
{
    if (output_format == QHO_OUTPUT_BIN) {
        return "bin";
    }
    if (output_format == QHO_OUTPUT_NONE) {
        return "none";
    }
    return "dat";
}

qho_params_t qho_params_default(void)
{
    qho_params_t params;

    params.nt = 64;
    params.beta = 4.0;
    params.eta = params.beta / (double)params.nt;
    params.n_therm = 1000L;
    params.n_sweeps = 10000L;
    params.meas_stride = 10L;
    params.seed = UINT64_C(123456789);
    params.stream = UINT64_C(54);
    params.delta = 1.0;
    params.init = QHO_INIT_ZERO;
    params.update_mode = QHO_UPDATE_METRO;
    params.n_overrelax = 5;
    params.hist_enabled = 0;
    params.hist_out[0] = '\0';
    params.hist_bins = 120;
    params.hist_min = -4.0;
    params.hist_max = 4.0;
    params.hist_bin_width = 0.0;
    params.hist_bin_width_explicit = 0;
    params.hist_block_enabled = 0;
    params.hist_block_out[0] = '\0';
    params.hist_block_size_saved = 100;
    params.corr_enabled = 0;
    params.corr_out[0] = '\0';
    params.corr_max_lag = 0;
    params.corr_max_lag_explicit = 0;
    params.spectrum_enabled = 0;
    params.spectrum_out[0] = '\0';
    params.spectrum_max_lag = 0;
    params.spectrum_max_lag_explicit = 0;
    params.spectrum_block_enabled = 0;
    params.spectrum_block_out[0] = '\0';
    params.spectrum_block_size_saved = 100;
    params.path_enabled = 0;
    params.path_out[0] = '\0';
    params.eta_explicit = 0;
    params.output_format = QHO_OUTPUT_DAT;
    params.therm_logbin_enabled = 0;
    params.therm_logbin_out[0] = '\0';
    params.logbin_base = 1.25;
    (void)copy_output_path(&params, "data/raw/qho_run.dat");

    return params;
}

int qho_params_parse_args(qho_params_t *params, int argc, char **argv)
{
    int i;
    const char *format_env = getenv("QHO_OUTPUT_FORMAT");
    const char *logbin_env = getenv("QHO_THERM_LOGBIN");
    const char *logbin_base_env = getenv("QHO_LOGBIN_BASE");

    if (format_env != NULL && format_env[0] != '\0') {
        if (parse_output_format_value(format_env, &params->output_format) != 0) {
            fprintf(stderr, "error: QHO_OUTPUT_FORMAT must be dat, bin, or none\n");
            return -1;
        }
    }
    if (logbin_env != NULL && strcmp(logbin_env, "1") == 0) {
        params->therm_logbin_enabled = 1;
    }
    if (logbin_base_env != NULL && logbin_base_env[0] != '\0') {
        if (parse_double_value(logbin_base_env, &params->logbin_base) != 0) {
            fprintf(stderr, "error: invalid QHO_LOGBIN_BASE value: %s\n", logbin_base_env);
            return -1;
        }
    }

    for (i = 1; i < argc; ++i) {
        const char *flag = argv[i];

        if (strcmp(flag, "--help") == 0 || strcmp(flag, "-h") == 0) {
            return 1;
        }

        if (i + 1 >= argc) {
            fprintf(stderr, "error: missing value after %s\n", flag);
            return -1;
        }

        const char *value = argv[++i];

        if (strcmp(flag, "--nt") == 0) {
            if (parse_int_value(value, &params->nt) != 0) {
                fprintf(stderr, "error: invalid --nt value: %s\n", value);
                return -1;
            }
        } else if (strcmp(flag, "--beta") == 0) {
            if (parse_double_value(value, &params->beta) != 0) {
                fprintf(stderr, "error: invalid --beta value: %s\n", value);
                return -1;
            }
        } else if (strcmp(flag, "--eta") == 0) {
            if (parse_double_value(value, &params->eta) != 0) {
                fprintf(stderr, "error: invalid --eta value: %s\n", value);
                return -1;
            }
            params->eta_explicit = 1;
        } else if (strcmp(flag, "--therm") == 0) {
            if (parse_long_value(value, &params->n_therm) != 0) {
                fprintf(stderr, "error: invalid --therm value: %s\n", value);
                return -1;
            }
        } else if (strcmp(flag, "--sweeps") == 0) {
            if (parse_long_value(value, &params->n_sweeps) != 0) {
                fprintf(stderr, "error: invalid --sweeps value: %s\n", value);
                return -1;
            }
        } else if (strcmp(flag, "--stride") == 0) {
            if (parse_long_value(value, &params->meas_stride) != 0) {
                fprintf(stderr, "error: invalid --stride value: %s\n", value);
                return -1;
            }
        } else if (strcmp(flag, "--seed") == 0) {
            if (parse_u64_value(value, &params->seed) != 0) {
                fprintf(stderr, "error: invalid --seed value: %s\n", value);
                return -1;
            }
        } else if (strcmp(flag, "--stream") == 0) {
            if (parse_u64_value(value, &params->stream) != 0) {
                fprintf(stderr, "error: invalid --stream value: %s\n", value);
                return -1;
            }
        } else if (strcmp(flag, "--delta") == 0) {
            if (parse_double_value(value, &params->delta) != 0) {
                fprintf(stderr, "error: invalid --delta value: %s\n", value);
                return -1;
            }
        } else if (strcmp(flag, "--init") == 0) {
            if (parse_init_value(value, &params->init) != 0) {
                fprintf(stderr, "error: --init must be 'zero', 'random', or 'uniform'\n");
                return -1;
            }
        } else if (strcmp(flag, "--update") == 0) {
            if (parse_update_value(value, &params->update_mode) != 0) {
                fprintf(stderr, "error: --update must be 'metro', 'heatbath', or 'hb-over'\n");
                return -1;
            }
        } else if (strcmp(flag, "--n-over") == 0) {
            if (parse_int_value(value, &params->n_overrelax) != 0) {
                fprintf(stderr, "error: invalid --n-over value: %s\n", value);
                return -1;
            }
        } else if (strcmp(flag, "--out") == 0) {
            if (copy_output_path(params, value) != 0) {
                fprintf(stderr, "error: invalid --out path\n");
                return -1;
            }
        } else if (strcmp(flag, "--format") == 0) {
            if (parse_output_format_value(value, &params->output_format) != 0) {
                fprintf(stderr, "error: --format must be dat, bin, or none\n");
                return -1;
            }
        } else if (strcmp(flag, "--hist-out") == 0) {
            if (copy_histogram_path(params, value) != 0) {
                fprintf(stderr, "error: invalid --hist-out path\n");
                return -1;
            }
            params->hist_enabled = 1;
        } else if (strcmp(flag, "--hist-bins") == 0) {
            if (parse_int_value(value, &params->hist_bins) != 0) {
                fprintf(stderr, "error: invalid --hist-bins value: %s\n", value);
                return -1;
            }
        } else if (strcmp(flag, "--hist-min") == 0) {
            if (parse_double_value(value, &params->hist_min) != 0) {
                fprintf(stderr, "error: invalid --hist-min value: %s\n", value);
                return -1;
            }
        } else if (strcmp(flag, "--hist-max") == 0) {
            if (parse_double_value(value, &params->hist_max) != 0) {
                fprintf(stderr, "error: invalid --hist-max value: %s\n", value);
                return -1;
            }
        } else if (strcmp(flag, "--hist-block-out") == 0) {
            if (copy_histogram_block_path(params, value) != 0) {
                fprintf(stderr, "error: invalid --hist-block-out path\n");
                return -1;
            }
            params->hist_block_enabled = 1;
        } else if (strcmp(flag, "--hist-bin-width") == 0) {
            if (parse_double_value(value, &params->hist_bin_width) != 0 || params->hist_bin_width <= 0.0) {
                fprintf(stderr, "error: invalid --hist-bin-width value: %s\n", value);
                return -1;
            }
            params->hist_bin_width_explicit = 1;
        } else if (strcmp(flag, "--hist-block-size-saved") == 0) {
            if (parse_int_value(value, &params->hist_block_size_saved) != 0) {
                fprintf(stderr, "error: invalid --hist-block-size-saved value: %s\n", value);
                return -1;
            }
        } else if (strcmp(flag, "--corr-out") == 0) {
            if (copy_correlator_path(params, value) != 0) {
                fprintf(stderr, "error: invalid --corr-out path\n");
                return -1;
            }
            params->corr_enabled = 1;
        } else if (strcmp(flag, "--corr-max-lag") == 0) {
            if (parse_int_value(value, &params->corr_max_lag) != 0) {
                fprintf(stderr, "error: invalid --corr-max-lag value: %s\n", value);
                return -1;
            }
            params->corr_max_lag_explicit = 1;
        } else if (strcmp(flag, "--spectrum-out") == 0) {
            if (copy_spectrum_path(params, value) != 0) {
                fprintf(stderr, "error: invalid --spectrum-out path\n");
                return -1;
            }
            params->spectrum_enabled = 1;
        } else if (strcmp(flag, "--spectrum-block-out") == 0) {
            if (copy_spectrum_block_path(params, value) != 0) {
                fprintf(stderr, "error: invalid --spectrum-block-out path\n");
                return -1;
            }
            params->spectrum_block_enabled = 1;
        } else if (strcmp(flag, "--spectrum-block-size-saved") == 0) {
            if (parse_int_value(value, &params->spectrum_block_size_saved) != 0) {
                fprintf(stderr, "error: invalid --spectrum-block-size-saved value: %s\n", value);
                return -1;
            }
        } else if (strcmp(flag, "--spectrum-max-lag") == 0) {
            if (parse_int_value(value, &params->spectrum_max_lag) != 0) {
                fprintf(stderr, "error: invalid --spectrum-max-lag value: %s\n", value);
                return -1;
            }
            params->spectrum_max_lag_explicit = 1;
        } else if (strcmp(flag, "--path-out") == 0) {
            if (copy_path_snapshot_path(params, value) != 0) {
                fprintf(stderr, "error: invalid --path-out path\n");
                return -1;
            }
            params->path_enabled = 1;
        } else if (strcmp(flag, "--therm-logbin-out") == 0) {
            if (copy_therm_logbin_path(params, value) != 0) {
                fprintf(stderr, "error: invalid --therm-logbin-out path\n");
                return -1;
            }
            params->therm_logbin_enabled = 1;
        } else if (strcmp(flag, "--logbin-base") == 0) {
            if (parse_double_value(value, &params->logbin_base) != 0) {
                fprintf(stderr, "error: invalid --logbin-base value: %s\n", value);
                return -1;
            }
        } else {
            fprintf(stderr, "error: unknown option: %s\n", flag);
            return -1;
        }
    }

    if (!params->eta_explicit) {
        params->eta = params->beta / (double)params->nt;
    }
    if (params->hist_bin_width_explicit) {
        const double bins_real = (params->hist_max - params->hist_min) / params->hist_bin_width;
        const long bins_round = lround(bins_real);
        if (bins_round <= 0L || bins_round > INT_MAX
            || fabs(bins_real - (double)bins_round) > 1.0e-10 * fmax(1.0, fabs(bins_real))) {
            fprintf(stderr, "error: --hist-bin-width must divide [hist-min,hist-max] into an integer number of bins\n");
            return -1;
        }
        params->hist_bins = (int)bins_round;
    }

    if (params->corr_enabled && !params->corr_max_lag_explicit) {
        params->corr_max_lag = params->nt / 2;
    }
    if ((params->spectrum_enabled || params->spectrum_block_enabled) && !params->spectrum_max_lag_explicit) {
        params->spectrum_max_lag = params->nt / 2;
    }

    return check_params(params);
}

void qho_params_print(const qho_params_t *params, FILE *stream)
{
    fprintf(stream, "Parameters:\n");
    fprintf(stream, "  nt          = %d\n", params->nt);
    fprintf(stream, "  beta        = %.17g\n", params->beta);
    fprintf(stream, "  eta         = %.17g\n", params->eta);
    fprintf(stream, "  n_therm     = %ld\n", params->n_therm);
    fprintf(stream, "  n_sweeps    = %ld\n", params->n_sweeps);
    fprintf(stream, "  meas_stride = %ld\n", params->meas_stride);
    fprintf(stream, "  seed        = %" PRIu64 "\n", params->seed);
    fprintf(stream, "  stream      = %" PRIu64 "\n", params->stream);
    fprintf(stream, "  delta       = %.17g\n", params->delta);
    fprintf(stream, "  init        = %s\n", qho_init_name(params->init));
    fprintf(stream, "  update      = %s\n", qho_update_mode_name(params->update_mode));
    fprintf(stream, "  n_over      = %d\n", params->n_overrelax);
    fprintf(stream, "  out         = %s\n", params->out_path);
    fprintf(stream, "  format      = %s\n", qho_output_format_name(params->output_format));
    if (params->hist_enabled) {
        fprintf(stream, "  hist_out    = %s\n", params->hist_out);
        fprintf(stream, "  hist_bins   = %d\n", params->hist_bins);
        fprintf(stream, "  hist_min    = %.17g\n", params->hist_min);
        fprintf(stream, "  hist_max    = %.17g\n", params->hist_max);
        fprintf(stream, "  hist_bin_width = %.17g\n", (params->hist_max - params->hist_min) / (double)params->hist_bins);
    }
    if (params->hist_block_enabled) {
        fprintf(stream, "  hist_block_out = %s\n", params->hist_block_out);
        fprintf(stream, "  hist_block_size_saved = %d\n", params->hist_block_size_saved);
    }
    if (params->corr_enabled) {
        fprintf(stream, "  corr_out    = %s\n", params->corr_out);
        fprintf(stream, "  corr_max_lag = %d\n", params->corr_max_lag);
    }
    if (params->spectrum_enabled) {
        fprintf(stream, "  spectrum_out = %s\n", params->spectrum_out);
        fprintf(stream, "  spectrum_max_lag = %d\n", params->spectrum_max_lag);
    }
    if (params->spectrum_block_enabled) {
        fprintf(stream, "  spectrum_block_out = %s\n", params->spectrum_block_out);
        fprintf(stream, "  spectrum_block_size_saved = %d\n", params->spectrum_block_size_saved);
        fprintf(stream, "  spectrum_max_lag = %d\n", params->spectrum_max_lag);
    }
    if (params->path_enabled) {
        fprintf(stream, "  path_out    = %s\n", params->path_out);
    }
    if (params->therm_logbin_enabled) {
        fprintf(stream, "  therm_logbin_out = %s\n", params->therm_logbin_out);
        fprintf(stream, "  logbin_base = %.17g\n", params->logbin_base);
    }
}

void qho_params_print_usage(FILE *stream, const char *program_name)
{
    fprintf(stream, "Usage: %s [options]\n", program_name);
    fprintf(stream, "\nOptions:\n");
    fprintf(stream, "  --nt INT                  Number of Euclidean time slices\n");
    fprintf(stream, "  --beta DOUBLE             Inverse temperature\n");
    fprintf(stream, "  --eta DOUBLE              Dimensionless lattice spacing parameter\n");
    fprintf(stream, "  --therm LONG              Thermalization sweeps\n");
    fprintf(stream, "  --sweeps LONG             Production sweeps\n");
    fprintf(stream, "  --stride LONG             Measurement stride in sweeps\n");
    fprintf(stream, "  --seed UINT64             PCG32 initial state\n");
    fprintf(stream, "  --stream UINT64           PCG32 stream sequence\n");
    fprintf(stream, "  --delta DOUBLE            Local Metropolis trial move size\n");
    fprintf(stream, "  --init zero|random|uniform Initial path; random is uniform[-1,1)\n");
    fprintf(stream, "  --update MODE             metro, heatbath, or hb-over\n");
    fprintf(stream, "  --n-over INT              Overrelaxation sweeps after each heatbath sweep\n");
    fprintf(stream, "  --out PATH                Output data file\n");
    fprintf(stream, "  --format dat|bin|none     Measurement output format, default dat\n");
    fprintf(stream, "  --hist-out PATH           Optional position histogram output file\n");
    fprintf(stream, "  --hist-bins INT           Histogram bin count, default 120\n");
    fprintf(stream, "  --hist-min DOUBLE         Histogram lower edge, default -4\n");
    fprintf(stream, "  --hist-max DOUBLE         Histogram upper edge, default 4\n");
    fprintf(stream, "  --hist-bin-width DOUBLE   Alternative width; must divide the histogram range\n");
    fprintf(stream, "  --hist-block-out PATH     Optional block-level position histogram output\n");
    fprintf(stream, "  --hist-block-size-saved INT  Block size in saved measurements, default 100\n");
    fprintf(stream, "  --corr-out PATH           Optional coordinate correlator output file\n");
    fprintf(stream, "  --corr-max-lag INT        Maximum correlator lag, default nt/2\n");
    fprintf(stream, "  --spectrum-out PATH       Optional multi-operator spectrum correlator output file\n");
    fprintf(stream, "  --spectrum-block-out PATH Optional block-level spectrum correlator output file\n");
    fprintf(stream, "  --spectrum-block-size-saved INT  Block size in saved measurements, default 100\n");
    fprintf(stream, "  --spectrum-max-lag INT    Maximum spectrum correlator lag, default nt/2\n");
    fprintf(stream, "  --path-out PATH           Optional final path snapshot output file\n");
    fprintf(stream, "  --therm-logbin-out PATH   Optional online thermalization log-bin output\n");
    fprintf(stream, "  --logbin-base DOUBLE      Log-bin growth base, default 1.25\n");
    fprintf(stream, "  --help                    Show this help message\n");
}
