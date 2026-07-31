/*
 * Parse the physical lattice specification and sampling controls for the
 * quartic oscillator.  beta, Nt, and lambda fix eta=beta/Nt and the interacting
 * Euclidean action used throughout the calculation.
 */

#include "aho_params.h"

#include <errno.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static int parse_int(const char *s, int *out)
{
    char *end = NULL;
    long value;
    errno = 0;
    value = strtol(s, &end, 10);
    if (errno || end == s || *end || value < 0 || value > 2147483647L) {
        return 0;
    }
    *out = (int)value;
    return 1;
}

static int parse_u64(const char *s, uint64_t *out)
{
    char *end = NULL;
    unsigned long long value;
    errno = 0;
    value = strtoull(s, &end, 10);
    if (errno || end == s || *end) {
        return 0;
    }
    *out = (uint64_t)value;
    return 1;
}

static int parse_double(const char *s, double *out)
{
    char *end = NULL;
    double value;
    errno = 0;
    value = strtod(s, &end);
    if (errno || end == s || *end || !isfinite(value)) {
        return 0;
    }
    *out = value;
    return 1;
}

aho_params_t aho_params_default(void)
{
    aho_params_t params;
    params.beta = 5.0;
    params.nt = 128;
    params.lambda = 0.0;
    params.n_therm = 20000;
    params.n_sweeps = 50000;
    params.meas_stride = 10;
    params.seed = 12345;
    params.delta = 1.0;
    params.update = AHO_UPDATE_METRO_OVER;
    params.n_over = 5;
    params.init = AHO_INIT_GAUSSIAN;
    params.hist_ymin = -4.0;
    params.hist_ymax = 4.0;
    params.hist_bins = 160;
    params.out = NULL;
    params.hist_out = NULL;
    params.path_out = NULL;
    params.corr_out = NULL;
    params.corr_max_dt = 0;
    params.corr_block_size = 100;
    return params;
}

int aho_params_parse_args(aho_params_t *params, int argc, char **argv)
{
    int i;
    *params = aho_params_default();
    for (i = 1; i < argc; i += 2) {
        if (i + 1 >= argc) {
            return 0;
        }
        if (strcmp(argv[i], "--beta") == 0) {
            if (!parse_double(argv[i + 1], &params->beta)) return 0;
        } else if (strcmp(argv[i], "--nt") == 0) {
            if (!parse_int(argv[i + 1], &params->nt)) return 0;
        } else if (strcmp(argv[i], "--lambda") == 0) {
            if (!parse_double(argv[i + 1], &params->lambda)) return 0;
        } else if (strcmp(argv[i], "--n-therm") == 0) {
            if (!parse_int(argv[i + 1], &params->n_therm)) return 0;
        } else if (strcmp(argv[i], "--n-sweeps") == 0) {
            if (!parse_int(argv[i + 1], &params->n_sweeps)) return 0;
        } else if (strcmp(argv[i], "--meas-stride") == 0) {
            if (!parse_int(argv[i + 1], &params->meas_stride)) return 0;
        } else if (strcmp(argv[i], "--seed") == 0) {
            if (!parse_u64(argv[i + 1], &params->seed)) return 0;
        } else if (strcmp(argv[i], "--delta") == 0) {
            if (!parse_double(argv[i + 1], &params->delta)) return 0;
        } else if (strcmp(argv[i], "--update") == 0) {
            if (strcmp(argv[i + 1], "metro") == 0) {
                params->update = AHO_UPDATE_METRO;
            } else if (strcmp(argv[i + 1], "metro-over") == 0) {
                params->update = AHO_UPDATE_METRO_OVER;
            } else {
                return 0;
            }
        } else if (strcmp(argv[i], "--n-over") == 0) {
            if (!parse_int(argv[i + 1], &params->n_over)) return 0;
        } else if (strcmp(argv[i], "--init") == 0) {
            if (strcmp(argv[i + 1], "zero") == 0) {
                params->init = AHO_INIT_ZERO;
            } else if (strcmp(argv[i + 1], "gaussian") == 0) {
                params->init = AHO_INIT_GAUSSIAN;
            } else if (strcmp(argv[i + 1], "uniform") == 0) {
                params->init = AHO_INIT_UNIFORM;
            } else {
                return 0;
            }
        } else if (strcmp(argv[i], "--hist-ymin") == 0) {
            if (!parse_double(argv[i + 1], &params->hist_ymin)) return 0;
        } else if (strcmp(argv[i], "--hist-ymax") == 0) {
            if (!parse_double(argv[i + 1], &params->hist_ymax)) return 0;
        } else if (strcmp(argv[i], "--hist-bins") == 0) {
            if (!parse_int(argv[i + 1], &params->hist_bins)) return 0;
        } else if (strcmp(argv[i], "--out") == 0) {
            params->out = argv[i + 1];
        } else if (strcmp(argv[i], "--hist-out") == 0) {
            params->hist_out = argv[i + 1];
        } else if (strcmp(argv[i], "--path-out") == 0) {
            params->path_out = argv[i + 1];
        } else if (strcmp(argv[i], "--corr-out") == 0) {
            params->corr_out = argv[i + 1];
        } else if (strcmp(argv[i], "--corr-max-dt") == 0) {
            if (!parse_int(argv[i + 1], &params->corr_max_dt)) return 0;
        } else if (strcmp(argv[i], "--corr-block-size") == 0) {
            if (!parse_int(argv[i + 1], &params->corr_block_size)) return 0;
        } else {
            return 0;
        }
    }
    return aho_params_validate(params);
}

/* Require a stable quartic potential and a well-defined periodic lattice ensemble. */
int aho_params_validate(const aho_params_t *params)
{
    return params->beta > 0.0 && params->nt > 1 && params->lambda >= 0.0 &&
           params->n_therm >= 0 && params->n_sweeps >= 0 && params->meas_stride > 0 &&
           params->delta > 0.0 && params->n_over >= 0 && params->hist_bins > 0 &&
           params->hist_ymax > params->hist_ymin && params->out &&
           (!params->corr_out || params->corr_block_size > 0);
}

/* Euclidean-time lattice spacing at fixed inverse temperature beta. */
double aho_params_eta(const aho_params_t *params)
{
    return params->beta / (double)params->nt;
}

void aho_params_print_usage(FILE *stream, const char *program_name)
{
    fprintf(stream,
            "Usage: %s --beta B --nt Nt --lambda L --n-therm N --n-sweeps N "
            "--meas-stride N --seed S --delta D --update metro|metro-over "
            "--n-over N --init zero|gaussian|uniform --hist-ymin A --hist-ymax B "
            "--hist-bins N --out FILE [--hist-out FILE] [--path-out FILE] "
            "[--corr-out FILE --corr-max-dt N --corr-block-size N]\n",
            program_name);
}

void aho_params_print_header(FILE *stream, const aho_params_t *params)
{
    fprintf(stream,
            "# beta %.17g eta %.17g Nt %d lambda %.17g seed %llu delta %.17g update %s "
            "n_over %d init %s n_therm %d n_sweeps %d meas_stride %d\n",
            params->beta, aho_params_eta(params), params->nt, params->lambda,
            (unsigned long long)params->seed, params->delta, aho_update_name(params->update),
            params->n_over, aho_init_name(params->init), params->n_therm,
            params->n_sweeps, params->meas_stride);
}

const char *aho_init_name(aho_init_t init)
{
    if (init == AHO_INIT_ZERO) return "zero";
    if (init == AHO_INIT_GAUSSIAN) return "gaussian";
    if (init == AHO_INIT_UNIFORM) return "uniform";
    return "unknown";
}

const char *aho_update_name(aho_update_t update)
{
    if (update == AHO_UPDATE_METRO) return "metro";
    if (update == AHO_UPDATE_METRO_OVER) return "metro-over";
    return "unknown";
}
