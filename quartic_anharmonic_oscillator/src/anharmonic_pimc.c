/*
 * Command-line driver for the quartic-oscillator Euclidean path integral.  It
 * thermalizes periodic paths, samples V=y^2/2+lambda*y^4, and records moments,
 * virial estimators, position densities, representative paths, and correlators.
 */

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "aho_histogram.h"
#include "aho_correlator.h"
#include "aho_lattice.h"
#include "aho_measure.h"
#include "aho_params.h"
#include "aho_path_output.h"
#include "aho_update.h"
#include "pcg32.h"

static int ensure_parent_directories(const char *path)
{
    char *buffer;
    char *cursor;
    size_t length;

    if (!path || *path == '\0' || strchr(path, '/') == NULL) {
        return 1;
    }

    length = strlen(path);
    buffer = malloc(length + 1U);
    if (!buffer) {
        fprintf(stderr, "[ERROR] allocation failed while preparing output path %s\n", path);
        return 0;
    }
    memcpy(buffer, path, length + 1U);

    for (cursor = buffer + 1; *cursor != '\0'; ++cursor) {
        if (*cursor != '/') {
            continue;
        }
        *cursor = '\0';
        if (mkdir(buffer, 0777) != 0 && errno != EEXIST) {
            fprintf(stderr, "[ERROR] cannot create output directory %s: %s\n",
                    buffer, strerror(errno));
            free(buffer);
            return 0;
        }
        *cursor = '/';
    }

    free(buffer);
    return 1;
}

static int ensure_output_directories(const aho_params_t *params)
{
    return ensure_parent_directories(params->out) &&
           ensure_parent_directories(params->hist_out) &&
           ensure_parent_directories(params->path_out) &&
           ensure_parent_directories(params->corr_out);
}

static void write_measurement_header(FILE *stream)
{
    fprintf(stream,
            "# measurement_index sweep beta eta Nt lambda accept_rate_metro accept_rate_over "
            "y_mean y2_mean y4_mean dy2_mean V_mean K_virial E_virial\n");
}

static void write_measurement(FILE *stream, int measurement_index, int sweep,
                              const aho_params_t *params, double accept_rate_metro,
                              double accept_rate_over, const aho_observables_t *obs)
{
    fprintf(stream,
            "%d %d %.17g %.17g %d %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n",
            measurement_index, sweep, params->beta, aho_params_eta(params), params->nt,
            params->lambda, accept_rate_metro, accept_rate_over, obs->y_mean,
            obs->y2_mean, obs->y4_mean, obs->dy2_mean, obs->v_mean,
            obs->k_virial, obs->e_virial);
}

/* Run one ensemble specified by beta, Nt, lambda, and the sampling parameters. */
int main(int argc, char **argv)
{
    aho_params_t params;
    pcg32_rng_t rng;
    aho_lattice_t lattice;
    aho_histogram_t histogram;
    aho_path_output_t path_output;
    aho_correlator_t correlator;
    FILE *measurements = NULL;
    int sweep;
    int measurement_index = 0;
    long long metro_attempts = 0;
    long long metro_accepts = 0;
    long long over_attempts = 0;
    long long over_accepts = 0;
    int exit_code = EXIT_FAILURE;

    lattice.nt = 0;
    lattice.y = NULL;
    histogram.counts = NULL;
    path_output.stream = NULL;
    correlator.stream = NULL;
    correlator.corr_sums = NULL;

    if (!aho_params_parse_args(&params, argc, argv)) {
        fprintf(stderr, "[ERROR] invalid command-line arguments\n");
        aho_params_print_usage(stderr, argv[0]);
        return EXIT_FAILURE;
    }
    if (!ensure_output_directories(&params)) {
        return EXIT_FAILURE;
    }

    pcg32_seed(&rng, params.seed);
    if (!aho_lattice_alloc(&lattice, params.nt)) {
        fprintf(stderr, "[ERROR] allocation failed for lattice\n");
        return EXIT_FAILURE;
    }
    if (params.hist_out && !aho_histogram_alloc(&histogram, params.hist_bins, params.hist_ymin, params.hist_ymax)) {
        fprintf(stderr, "[ERROR] allocation failed for histogram\n");
        aho_lattice_free(&lattice);
        return EXIT_FAILURE;
    }
    aho_lattice_init(&lattice, params.init, &rng);

    measurements = fopen(params.out, "w");
    if (!measurements) {
        fprintf(stderr, "[ERROR] cannot open %s: %s\n", params.out, strerror(errno));
        goto cleanup;
    }
    if (!aho_path_output_open(&path_output, params.path_out, 8)) {
        goto cleanup;
    }
    if (!aho_correlator_open(&correlator, &params)) {
        goto cleanup;
    }

    aho_params_print_header(measurements, &params);
    write_measurement_header(measurements);
    if (ferror(measurements)) {
        fprintf(stderr, "[ERROR] cannot write measurement header to %s: %s\n", params.out, strerror(errno));
        goto cleanup;
    }

    /* Negative sweep labels denote equilibration and are never measured. */
    for (sweep = -params.n_therm; sweep < params.n_sweeps; ++sweep) {
        aho_update_counts_t counts = aho_update_sweep(&lattice, &params, &rng);
        metro_attempts += counts.metro_attempts;
        metro_accepts += counts.metro_accepts;
        over_attempts += counts.over_attempts;
        over_accepts += counts.over_accepts;

        /* Save statistically spaced observables only from the equilibrium chain. */
        if (sweep >= 0 && (sweep % params.meas_stride) == 0) {
            aho_observables_t obs = aho_measure_observables(&lattice, params.lambda);
            double accept_rate_metro = metro_attempts > 0 ?
                                       (double)metro_accepts / (double)metro_attempts : 0.0;
            double accept_rate_over = over_attempts > 0 ?
                                      (double)over_accepts / (double)over_attempts : 0.0;
            write_measurement(measurements, measurement_index, sweep, &params,
                              accept_rate_metro, accept_rate_over, &obs);
            if (ferror(measurements)) {
                fprintf(stderr, "[ERROR] cannot write measurement row to %s: %s\n", params.out, strerror(errno));
                goto cleanup;
            }
            if (params.hist_out) {
                aho_histogram_accumulate(&histogram, &lattice);
            }
            if (!aho_path_output_maybe_write(&path_output, &lattice, aho_params_eta(&params))) {
                goto cleanup;
            }
            if (!aho_correlator_accumulate(&correlator, &lattice, &params)) {
                goto cleanup;
            }
            measurement_index++;
        }
    }

    if (!aho_path_output_close(&path_output)) {
        goto cleanup;
    }
    if (!aho_correlator_finish(&correlator, &params)) {
        goto cleanup;
    }
    if (measurements && fclose(measurements) != 0) {
        fprintf(stderr, "[ERROR] cannot close %s: %s\n", params.out, strerror(errno));
        measurements = NULL;
        goto cleanup;
    }
    measurements = NULL;
    /* Normalize the accumulated position distribution after the Markov chain ends. */
    if (params.hist_out && !aho_histogram_write(&histogram, &params)) {
        goto cleanup;
    }
    exit_code = EXIT_SUCCESS;

cleanup:
    if (path_output.stream) {
        (void)aho_path_output_close(&path_output);
    }
    if (measurements) {
        if (fclose(measurements) != 0) {
            fprintf(stderr, "[ERROR] cannot close %s during cleanup: %s\n", params.out, strerror(errno));
            exit_code = EXIT_FAILURE;
        }
    }
    aho_correlator_free(&correlator);
    aho_histogram_free(&histogram);
    aho_lattice_free(&lattice);
    return exit_code;
}
