/*
 * Project: Quantum Harmonic Oscillator PIMC
 * File: src/qho_correlator.c
 * Purpose: Euclidean correlator accumulation and normalization routines.
 * Translational averages of y_j y_{j+k} are accumulated on periodic paths and
 * converted to connected correlators and nearest-lag effective gaps.
 */

/* -------------------------------------------------------------------------
 * Section: includes and implementation
 * ------------------------------------------------------------------------- */

#include "qho_correlator.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int qho_y_correlator_init(qho_y_correlator_t *corr, int nt, int max_lag)
{
    if (corr == NULL || nt <= 1 || max_lag <= 0 || max_lag >= nt) {
        return -1;
    }

    corr->nt = nt;
    corr->max_lag = max_lag;
    corr->sum_yy = calloc((size_t)max_lag + 1U, sizeof(*corr->sum_yy));
    corr->sum_y = 0.0;
    corr->n_measurements = 0ULL;

    if (corr->sum_yy == NULL) {
        corr->nt = 0;
        corr->max_lag = 0;
        return -1;
    }

    return 0;
}

void qho_y_correlator_free(qho_y_correlator_t *corr)
{
    if (corr == NULL) {
        return;
    }

    free(corr->sum_yy);
    corr->sum_yy = NULL;
    corr->nt = 0;
    corr->max_lag = 0;
    corr->sum_y = 0.0;
    corr->n_measurements = 0ULL;
}

/* Average over every Euclidean-time origin to reduce the correlator variance. */
void qho_y_correlator_accumulate(qho_y_correlator_t *corr, const qho_lattice_t *lat)
{
    int i;
    int lag;
    double y_sum = 0.0;

    if (corr == NULL || corr->sum_yy == NULL || lat == NULL || lat->y == NULL) {
        return;
    }
    if (lat->nt != corr->nt) {
        return;
    }

    for (i = 0; i < corr->nt; ++i) {
        y_sum += lat->y[i];
    }
    corr->sum_y += y_sum / (double)corr->nt;

    for (lag = 0; lag <= corr->max_lag; ++lag) {
        double yy_sum = 0.0;

        for (i = 0; i < corr->nt; ++i) {
            const int j = (i + lag) % corr->nt;
            yy_sum += lat->y[i] * lat->y[j];
        }

        corr->sum_yy[lag] += yy_sum / (double)corr->nt;
    }

    ++corr->n_measurements;
}

/* Normalize the ensemble sums and form C_y(tau) = <yy> - <y>^2. */
int qho_y_correlator_write(
    const qho_y_correlator_t *corr,
    const char *path,
    const qho_params_t *params
)
{
    int lag;
    FILE *out;
    const double inv_measurements = (corr != NULL && corr->n_measurements > 0ULL)
        ? 1.0 / (double)corr->n_measurements
        : 0.0;
    double mean_y;

    if (corr == NULL || corr->sum_yy == NULL || path == NULL || params == NULL) {
        return -1;
    }
    if (corr->n_measurements == 0ULL) {
        return -1;
    }

    out = fopen(path, "w");
    if (out == NULL) {
        return -1;
    }

    mean_y = corr->sum_y * inv_measurements;

    fprintf(out, "# QHO_PIMC y correlator\n");
    fprintf(out, "# beta %.17g\n", params->beta);
    fprintf(out, "# eta %.17g\n", params->eta);
    fprintf(out, "# nt %d\n", params->nt);
    fprintf(out, "# therm %ld\n", params->n_therm);
    fprintf(out, "# sweeps %ld\n", params->n_sweeps);
    fprintf(out, "# stride %ld\n", params->meas_stride);
    fprintf(out, "# seed %llu\n", (unsigned long long)params->seed);
    fprintf(out, "# stream %llu\n", (unsigned long long)params->stream);
    fprintf(out, "# init %s\n", qho_init_name(params->init));
    fprintf(out, "# update %s\n", qho_update_mode_name(params->update_mode));
    fprintf(out, "# n_over %d\n", params->n_overrelax);
    fprintf(out, "# corr_max_lag %d\n", corr->max_lag);
    fprintf(out, "# corr_measurements %llu\n", corr->n_measurements);
    fprintf(out, "# columns lag tau raw_yy connected_y effective_mass\n");

    for (lag = 0; lag <= corr->max_lag; ++lag) {
        const double tau = (double)lag * params->eta;
        const double raw_yy = corr->sum_yy[lag] * inv_measurements;
        const double connected = raw_yy - mean_y * mean_y;
        double effective_mass = NAN;

        /* The log ratio approaches the lowest gap coupled to y at large tau. */
        if (lag < corr->max_lag) {
            const double raw_next = corr->sum_yy[lag + 1] * inv_measurements;
            const double connected_next = raw_next - mean_y * mean_y;

            if (connected > 0.0 && connected_next > 0.0) {
                effective_mass = log(connected / connected_next) / params->eta;
            }
        }

        fprintf(out, "%d %.17g %.17g %.17g %.17g\n",
            lag,
            tau,
            raw_yy,
            connected,
            effective_mass);
    }

    if (fclose(out) != 0) {
        return -1;
    }

    return 0;
}
