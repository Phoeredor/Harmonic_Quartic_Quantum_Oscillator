/*
 * Accumulate block-resolved Euclidean correlator matrices for odd operators
 * (y,y^3) and even operators (y^2,y^4).  Downstream jackknife GEVP analysis uses
 * these matrices to isolate excitation gaps of the quartic oscillator.
 */

#include "aho_correlator.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>

/* Four matrix elements are stored for each of the odd and even sectors. */
enum { N_CORRELATORS = 8 };

static size_t corr_index(int dt, int component)
{
    return (size_t)dt * N_CORRELATORS + (size_t)component;
}

static void reset_block(aho_correlator_t *corr)
{
    size_t n = (size_t)(corr->max_dt + 1) * N_CORRELATORS;
    memset(corr->corr_sums, 0, n * sizeof(*corr->corr_sums));
    memset(corr->mean_sums, 0, sizeof(corr->mean_sums));
    corr->n_meas = 0;
}

/* Normalize one contiguous Monte Carlo block without forming connected subtractions. */
static int write_block(aho_correlator_t *corr, const aho_params_t *params)
{
    int dt;
    double norm_meas = 1.0 / (double)corr->n_meas;
    for (dt = 0; dt <= corr->max_dt; ++dt) {
        int k;
        fprintf(corr->stream, "%d %.17g %.17g %d %.17g %d %d %.17g",
                corr->block_index, params->beta, aho_params_eta(params), params->nt,
                params->lambda, corr->n_meas, dt, dt * aho_params_eta(params));
        for (k = 0; k < 4; ++k) {
            fprintf(corr->stream, " %.17g", corr->mean_sums[k] * norm_meas);
        }
        for (k = 0; k < N_CORRELATORS; ++k) {
            fprintf(corr->stream, " %.17g",
                    corr->corr_sums[corr_index(dt, k)] * norm_meas);
        }
        fputc('\n', corr->stream);
    }
    if (ferror(corr->stream)) {
        fprintf(stderr, "[ERROR] cannot write correlator block: %s\n", strerror(errno));
        return 0;
    }
    corr->block_index++;
    reset_block(corr);
    return 1;
}

int aho_correlator_open(aho_correlator_t *corr, const aho_params_t *params)
{
    size_t n;
    memset(corr, 0, sizeof(*corr));
    if (!params->corr_out) return 1;
    corr->max_dt = params->corr_max_dt;
    if (corr->max_dt > params->nt / 2) {
        fprintf(stderr, "[INFO] corr-max-dt truncated from %d to Nt/2 = %d\n",
                corr->max_dt, params->nt / 2);
        corr->max_dt = params->nt / 2;
    }
    corr->block_size = params->corr_block_size;
    n = (size_t)(corr->max_dt + 1) * N_CORRELATORS;
    corr->corr_sums = calloc(n, sizeof(*corr->corr_sums));
    if (!corr->corr_sums) {
        fprintf(stderr, "[ERROR] allocation failed for correlator accumulator\n");
        return 0;
    }
    corr->stream = fopen(params->corr_out, "w");
    if (!corr->stream) {
        fprintf(stderr, "[ERROR] cannot open %s: %s\n", params->corr_out, strerror(errno));
        aho_correlator_free(corr);
        return 0;
    }
    aho_params_print_header(corr->stream, params);
    fprintf(corr->stream,
            "# block_index beta eta Nt lambda n_meas dt tau mean_y mean_y2 mean_y3 mean_y4 "
            "odd_00 odd_01 odd_10 odd_11 even_00 even_01 even_10 even_11\n");
    if (ferror(corr->stream)) {
        fprintf(stderr, "[ERROR] cannot write correlator header to %s: %s\n",
                params->corr_out, strerror(errno));
        aho_correlator_free(corr);
        return 0;
    }
    return 1;
}

/* Average products over all Euclidean-time origins before adding the configuration. */
int aho_correlator_accumulate(aho_correlator_t *corr, const aho_lattice_t *lat,
                              const aho_params_t *params)
{
    int s;
    int dt;
    double means[4] = {0.0, 0.0, 0.0, 0.0};
    (void)params;
    if (!corr->stream) return 1;

    for (s = 0; s < lat->nt; ++s) {
        double y = lat->y[s];
        double y2 = y * y;
        double y3 = y2 * y;
        double y4 = y2 * y2;
        means[0] += y;
        means[1] += y2;
        means[2] += y3;
        means[3] += y4;
    }
    for (s = 0; s < 4; ++s) corr->mean_sums[s] += means[s] / lat->nt;

    for (dt = 0; dt <= corr->max_dt; ++dt) {
        double sums[N_CORRELATORS] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        for (s = 0; s < lat->nt; ++s) {
            int sp = (s + dt) % lat->nt;
            double y = lat->y[s];
            double yp = lat->y[sp];
            double y2 = y * y;
            double yp2 = yp * yp;
            double y3 = y2 * y;
            double yp3 = yp2 * yp;
            double y4 = y2 * y2;
            double yp4 = yp2 * yp2;
            sums[0] += y * yp;
            sums[1] += y * yp3;
            sums[2] += y3 * yp;
            sums[3] += y3 * yp3;
            sums[4] += y2 * yp2;
            sums[5] += y2 * yp4;
            sums[6] += y4 * yp2;
            sums[7] += y4 * yp4;
        }
        for (s = 0; s < N_CORRELATORS; ++s) {
            corr->corr_sums[corr_index(dt, s)] += sums[s] / lat->nt;
        }
    }
    corr->n_meas++;
    if (corr->n_meas == corr->block_size) return write_block(corr, params);
    return 1;
}

/* Retain a substantial final block while avoiding a very small statistical block. */
int aho_correlator_finish(aho_correlator_t *corr, const aho_params_t *params)
{
    int ok = 1;
    if (!corr->stream) return 1;
    if (corr->n_meas > 0) {
        if (2 * corr->n_meas >= corr->block_size) {
            fprintf(stderr, "[INFO] writing partial correlator block with %d/%d measurements\n",
                    corr->n_meas, corr->block_size);
            ok = write_block(corr, params);
        } else {
            fprintf(stderr, "[INFO] discarding partial correlator block with %d/%d measurements\n",
                    corr->n_meas, corr->block_size);
        }
    }
    if (fclose(corr->stream) != 0) {
        fprintf(stderr, "[ERROR] cannot close %s: %s\n", params->corr_out, strerror(errno));
        ok = 0;
    }
    corr->stream = NULL;
    return ok;
}

void aho_correlator_free(aho_correlator_t *corr)
{
    if (corr->stream) fclose(corr->stream);
    free(corr->corr_sums);
    corr->stream = NULL;
    corr->corr_sums = NULL;
}
