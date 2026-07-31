/*
 * Project: Quantum Harmonic Oscillator PIMC
 * File: src/qho_spectrum.c
 * Purpose: Observable construction for imaginary-time spectrum measurements.
 * Raw and connected correlators of y, y^2, y^3, and an improved cubic operator
 * are accumulated in full ensembles and statistical blocks for gap extraction.
 */

/* -------------------------------------------------------------------------
 * Section: includes and implementation
 * ------------------------------------------------------------------------- */

#include "qho_spectrum.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static size_t lag_count(int max_lag)
{
    return (size_t)max_lag + 1U;
}

/* Subtract the component coupling to the first odd state, isolating Delta E_3. */
static double op_a3(double y)
{
    return y * y * y - 1.5 * y;
}

static double op_y3(double y)
{
    return y * y * y;
}

/* Adjacent-lag log ratio for a correlator dominated by one exponential state. */
static double effective_mass(double c0, double c1, double eta)
{
    if (c0 > 0.0 && c1 > 0.0) {
        return log(c0 / c1) / eta;
    }
    return NAN;
}

static int spectrum_params_valid(const qho_params_t *params)
{
    return params != NULL
        && params->nt > 1
        && params->beta > 0.0
        && params->eta > 0.0
        && isfinite(params->beta)
        && isfinite(params->eta);
}

static int spectrum_block_arrays_ready(const qho_spectrum_block_accumulator_t *blocks)
{
    return blocks != NULL
        && blocks->block_measurements != NULL
        && blocks->block_mean_y != NULL
        && blocks->block_mean_y2 != NULL
        && blocks->block_mean_y3 != NULL
        && blocks->block_mean_a != NULL
        && blocks->block_raw_y_y != NULL
        && blocks->block_raw_y2_y2 != NULL
        && blocks->block_raw_y3_y3 != NULL
        && blocks->block_raw_a_a != NULL;
}

static int spectrum_block_current_arrays_ready(const qho_spectrum_block_accumulator_t *blocks)
{
    return blocks != NULL
        && blocks->current_sum_y_y != NULL
        && blocks->current_sum_y2_y2 != NULL
        && blocks->current_sum_y3_y3 != NULL
        && blocks->current_sum_a_a != NULL;
}

static int spectrum_block_values_finite(const qho_spectrum_block_accumulator_t *blocks)
{
    const size_t n_lag = lag_count(blocks->max_lag);
    unsigned long long block_id;
    int lag;

    for (block_id = 0ULL; block_id < blocks->n_blocks; ++block_id) {
        const size_t offset = (size_t)block_id * n_lag;
        if (blocks->block_measurements[block_id] == 0ULL) {
            return 0;
        }
        if (!isfinite(blocks->block_mean_y[block_id])
            || !isfinite(blocks->block_mean_y2[block_id])
            || !isfinite(blocks->block_mean_y3[block_id])
            || !isfinite(blocks->block_mean_a[block_id])) {
            return 0;
        }
        for (lag = 0; lag <= blocks->max_lag; ++lag) {
            const size_t index = offset + (size_t)lag;
            if (!isfinite(blocks->block_raw_y_y[index])
                || !isfinite(blocks->block_raw_y2_y2[index])
                || !isfinite(blocks->block_raw_y3_y3[index])
                || !isfinite(blocks->block_raw_a_a[index])) {
                return 0;
            }
        }
    }

    return 1;
}

int qho_spectrum_correlator_init(qho_spectrum_correlator_t *corr, int nt, int max_lag)
{
    const size_t n = (size_t)max_lag + 1U;

    if (corr == NULL || nt <= 1 || max_lag <= 0 || max_lag >= nt) {
        return -1;
    }

    corr->nt = nt;
    corr->max_lag = max_lag;
    corr->sum_y_y = calloc(n, sizeof(*corr->sum_y_y));
    corr->sum_y2_y2 = calloc(n, sizeof(*corr->sum_y2_y2));
    corr->sum_a_a = calloc(n, sizeof(*corr->sum_a_a));
    corr->sum_y = 0.0;
    corr->sum_y2 = 0.0;
    corr->sum_a = 0.0;
    corr->n_measurements = 0ULL;

    if (corr->sum_y_y == NULL || corr->sum_y2_y2 == NULL || corr->sum_a_a == NULL) {
        qho_spectrum_correlator_free(corr);
        return -1;
    }

    return 0;
}

void qho_spectrum_correlator_free(qho_spectrum_correlator_t *corr)
{
    if (corr == NULL) {
        return;
    }

    free(corr->sum_y_y);
    free(corr->sum_y2_y2);
    free(corr->sum_a_a);
    corr->sum_y_y = NULL;
    corr->sum_y2_y2 = NULL;
    corr->sum_a_a = NULL;
    corr->nt = 0;
    corr->max_lag = 0;
    corr->sum_y = 0.0;
    corr->sum_y2 = 0.0;
    corr->sum_a = 0.0;
    corr->n_measurements = 0ULL;
}

/* Accumulate one- and two-point functions averaged over all time origins. */
void qho_spectrum_correlator_accumulate(qho_spectrum_correlator_t *corr, const qho_lattice_t *lat)
{
    int i;
    int lag;
    double y_sum = 0.0;
    double y2_sum = 0.0;
    double a_sum = 0.0;

    if (corr == NULL || corr->sum_y_y == NULL || corr->sum_y2_y2 == NULL || corr->sum_a_a == NULL) {
        return;
    }
    if (lat == NULL || lat->y == NULL || lat->nt != corr->nt) {
        return;
    }

    for (i = 0; i < corr->nt; ++i) {
        const double y = lat->y[i];
        const double y2 = y * y;
        const double a = op_a3(y);

        y_sum += y;
        y2_sum += y2;
        a_sum += a;
    }

    corr->sum_y += y_sum / (double)corr->nt;
    corr->sum_y2 += y2_sum / (double)corr->nt;
    corr->sum_a += a_sum / (double)corr->nt;

    for (lag = 0; lag <= corr->max_lag; ++lag) {
        double raw_y = 0.0;
        double raw_y2 = 0.0;
        double raw_a = 0.0;

        for (i = 0; i < corr->nt; ++i) {
            const int j = (i + lag) % corr->nt;
            const double yi = lat->y[i];
            const double yj = lat->y[j];
            const double yi2 = yi * yi;
            const double yj2 = yj * yj;
            const double ai = op_a3(yi);
            const double aj = op_a3(yj);

            raw_y += yi * yj;
            raw_y2 += yi2 * yj2;
            raw_a += ai * aj;
        }

        corr->sum_y_y[lag] += raw_y / (double)corr->nt;
        corr->sum_y2_y2[lag] += raw_y2 / (double)corr->nt;
        corr->sum_a_a[lag] += raw_a / (double)corr->nt;
    }

    ++corr->n_measurements;
}

int qho_spectrum_correlator_write(
    const qho_spectrum_correlator_t *corr,
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
    double mean_y2;
    double mean_a;

    if (corr == NULL || corr->sum_y_y == NULL || corr->sum_y2_y2 == NULL || corr->sum_a_a == NULL) {
        return -1;
    }
    if (path == NULL || params == NULL || corr->n_measurements == 0ULL) {
        return -1;
    }

    out = fopen(path, "w");
    if (out == NULL) {
        return -1;
    }

    mean_y = corr->sum_y * inv_measurements;
    mean_y2 = corr->sum_y2 * inv_measurements;
    mean_a = corr->sum_a * inv_measurements;

    fprintf(out, "# QHO_PIMC spectrum correlators\n");
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
    fprintf(out, "# spectrum_max_lag %d\n", corr->max_lag);
    fprintf(out, "# spectrum_measurements %llu\n", corr->n_measurements);
    fprintf(out, "# operators y y2 a3\n");
    fprintf(out, "# a3_definition y^3 - 1.5*y\n");
    fprintf(out, "# columns lag tau raw_y connected_y meff_y raw_y2 connected_y2 meff_y2 raw_a3 connected_a3 meff_a3\n");

    for (lag = 0; lag <= corr->max_lag; ++lag) {
        const double tau = (double)lag * params->eta;
        const double raw_y = corr->sum_y_y[lag] * inv_measurements;
        const double raw_y2 = corr->sum_y2_y2[lag] * inv_measurements;
        const double raw_a = corr->sum_a_a[lag] * inv_measurements;
        const double connected_y = raw_y - mean_y * mean_y;
        const double connected_y2 = raw_y2 - mean_y2 * mean_y2;
        const double connected_a = raw_a - mean_a * mean_a;
        double meff_y = NAN;
        double meff_y2 = NAN;
        double meff_a = NAN;

        if (lag < corr->max_lag) {
            const double raw_y_next = corr->sum_y_y[lag + 1] * inv_measurements;
            const double raw_y2_next = corr->sum_y2_y2[lag + 1] * inv_measurements;
            const double raw_a_next = corr->sum_a_a[lag + 1] * inv_measurements;
            const double connected_y_next = raw_y_next - mean_y * mean_y;
            const double connected_y2_next = raw_y2_next - mean_y2 * mean_y2;
            const double connected_a_next = raw_a_next - mean_a * mean_a;

            meff_y = effective_mass(connected_y, connected_y_next, params->eta);
            meff_y2 = effective_mass(connected_y2, connected_y2_next, params->eta);
            meff_a = effective_mass(connected_a, connected_a_next, params->eta);
        }

        fprintf(out, "%d %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n",
            lag,
            tau,
            raw_y,
            connected_y,
            meff_y,
            raw_y2,
            connected_y2,
            meff_y2,
            raw_a,
            connected_a,
            meff_a);
    }

    if (fclose(out) != 0) {
        return -1;
    }

    return 0;
}

static void reset_current_block(qho_spectrum_block_accumulator_t *blocks)
{
    const size_t n_lag = lag_count(blocks->max_lag);
    size_t i;

    blocks->current_measurements = 0ULL;
    blocks->current_sum_y = 0.0;
    blocks->current_sum_y2 = 0.0;
    blocks->current_sum_y3 = 0.0;
    blocks->current_sum_a = 0.0;
    for (i = 0U; i < n_lag; ++i) {
        blocks->current_sum_y_y[i] = 0.0;
        blocks->current_sum_y2_y2[i] = 0.0;
        blocks->current_sum_y3_y3[i] = 0.0;
        blocks->current_sum_a_a[i] = 0.0;
    }
}

static int ensure_block_capacity(qho_spectrum_block_accumulator_t *blocks)
{
    unsigned long long new_capacity;
    unsigned long long *new_measurements;
    double *new_mean_y;
    double *new_mean_y2;
    double *new_mean_y3;
    double *new_mean_a;
    double *new_raw_y_y;
    double *new_raw_y2_y2;
    double *new_raw_y3_y3;
    double *new_raw_a_a;
    const size_t n_lag = lag_count(blocks->max_lag);

    if (blocks->n_blocks < blocks->block_capacity) {
        return 0;
    }

    new_capacity = (blocks->block_capacity == 0ULL) ? 8ULL : 2ULL * blocks->block_capacity;

    new_measurements = realloc(blocks->block_measurements, (size_t)new_capacity * sizeof(*blocks->block_measurements));
    if (new_measurements == NULL) return -1;
    blocks->block_measurements = new_measurements;

    new_mean_y = realloc(blocks->block_mean_y, (size_t)new_capacity * sizeof(*blocks->block_mean_y));
    if (new_mean_y == NULL) return -1;
    blocks->block_mean_y = new_mean_y;

    new_mean_y2 = realloc(blocks->block_mean_y2, (size_t)new_capacity * sizeof(*blocks->block_mean_y2));
    if (new_mean_y2 == NULL) return -1;
    blocks->block_mean_y2 = new_mean_y2;

    new_mean_y3 = realloc(blocks->block_mean_y3, (size_t)new_capacity * sizeof(*blocks->block_mean_y3));
    if (new_mean_y3 == NULL) return -1;
    blocks->block_mean_y3 = new_mean_y3;

    new_mean_a = realloc(blocks->block_mean_a, (size_t)new_capacity * sizeof(*blocks->block_mean_a));
    if (new_mean_a == NULL) return -1;
    blocks->block_mean_a = new_mean_a;

    new_raw_y_y = realloc(blocks->block_raw_y_y, (size_t)new_capacity * n_lag * sizeof(*blocks->block_raw_y_y));
    if (new_raw_y_y == NULL) return -1;
    blocks->block_raw_y_y = new_raw_y_y;

    new_raw_y2_y2 = realloc(blocks->block_raw_y2_y2, (size_t)new_capacity * n_lag * sizeof(*blocks->block_raw_y2_y2));
    if (new_raw_y2_y2 == NULL) return -1;
    blocks->block_raw_y2_y2 = new_raw_y2_y2;

    new_raw_y3_y3 = realloc(blocks->block_raw_y3_y3, (size_t)new_capacity * n_lag * sizeof(*blocks->block_raw_y3_y3));
    if (new_raw_y3_y3 == NULL) return -1;
    blocks->block_raw_y3_y3 = new_raw_y3_y3;

    new_raw_a_a = realloc(blocks->block_raw_a_a, (size_t)new_capacity * n_lag * sizeof(*blocks->block_raw_a_a));
    if (new_raw_a_a == NULL) return -1;
    blocks->block_raw_a_a = new_raw_a_a;

    blocks->block_capacity = new_capacity;
    return 0;
}

/* Store raw block means so connected subtractions can be jackknifed downstream. */
static int append_current_block(qho_spectrum_block_accumulator_t *blocks)
{
    const size_t n_lag = lag_count(blocks->max_lag);
    const double inv_measurements = (blocks->current_measurements > 0ULL)
        ? 1.0 / (double)blocks->current_measurements
        : 0.0;
    const size_t block_offset = (size_t)blocks->n_blocks * n_lag;
    size_t lag;

    if (blocks->current_measurements == 0ULL) {
        return 0;
    }
    if (ensure_block_capacity(blocks) != 0) {
        return -1;
    }

    blocks->block_measurements[blocks->n_blocks] = blocks->current_measurements;
    blocks->block_mean_y[blocks->n_blocks] = blocks->current_sum_y * inv_measurements;
    blocks->block_mean_y2[blocks->n_blocks] = blocks->current_sum_y2 * inv_measurements;
    blocks->block_mean_y3[blocks->n_blocks] = blocks->current_sum_y3 * inv_measurements;
    blocks->block_mean_a[blocks->n_blocks] = blocks->current_sum_a * inv_measurements;

    for (lag = 0U; lag < n_lag; ++lag) {
        blocks->block_raw_y_y[block_offset + lag] = blocks->current_sum_y_y[lag] * inv_measurements;
        blocks->block_raw_y2_y2[block_offset + lag] = blocks->current_sum_y2_y2[lag] * inv_measurements;
        blocks->block_raw_y3_y3[block_offset + lag] = blocks->current_sum_y3_y3[lag] * inv_measurements;
        blocks->block_raw_a_a[block_offset + lag] = blocks->current_sum_a_a[lag] * inv_measurements;
    }

    ++blocks->n_blocks;
    reset_current_block(blocks);
    return 0;
}

int qho_spectrum_block_accumulator_init(
    qho_spectrum_block_accumulator_t *blocks,
    int nt,
    int max_lag,
    int block_size_saved
)
{
    const size_t n_lag = lag_count(max_lag);

    if (blocks == NULL || nt <= 1 || max_lag <= 0 || max_lag >= nt || block_size_saved <= 0) {
        return -1;
    }

    blocks->nt = nt;
    blocks->max_lag = max_lag;
    blocks->block_size_saved = block_size_saved;
    blocks->current_measurements = 0ULL;
    blocks->n_blocks = 0ULL;
    blocks->block_capacity = 0ULL;
    blocks->block_measurements = NULL;
    blocks->block_mean_y = NULL;
    blocks->block_mean_y2 = NULL;
    blocks->block_mean_y3 = NULL;
    blocks->block_mean_a = NULL;
    blocks->block_raw_y_y = NULL;
    blocks->block_raw_y2_y2 = NULL;
    blocks->block_raw_y3_y3 = NULL;
    blocks->block_raw_a_a = NULL;
    blocks->current_sum_y = 0.0;
    blocks->current_sum_y2 = 0.0;
    blocks->current_sum_y3 = 0.0;
    blocks->current_sum_a = 0.0;
    blocks->current_sum_y_y = calloc(n_lag, sizeof(*blocks->current_sum_y_y));
    blocks->current_sum_y2_y2 = calloc(n_lag, sizeof(*blocks->current_sum_y2_y2));
    blocks->current_sum_y3_y3 = calloc(n_lag, sizeof(*blocks->current_sum_y3_y3));
    blocks->current_sum_a_a = calloc(n_lag, sizeof(*blocks->current_sum_a_a));

    if (blocks->current_sum_y_y == NULL || blocks->current_sum_y2_y2 == NULL || blocks->current_sum_y3_y3 == NULL || blocks->current_sum_a_a == NULL) {
        qho_spectrum_block_accumulator_free(blocks);
        return -1;
    }

    return 0;
}

void qho_spectrum_block_accumulator_free(qho_spectrum_block_accumulator_t *blocks)
{
    if (blocks == NULL) {
        return;
    }

    free(blocks->block_measurements);
    free(blocks->block_mean_y);
    free(blocks->block_mean_y2);
    free(blocks->block_mean_y3);
    free(blocks->block_mean_a);
    free(blocks->block_raw_y_y);
    free(blocks->block_raw_y2_y2);
    free(blocks->block_raw_y3_y3);
    free(blocks->block_raw_a_a);
    free(blocks->current_sum_y_y);
    free(blocks->current_sum_y2_y2);
    free(blocks->current_sum_y3_y3);
    free(blocks->current_sum_a_a);

    blocks->nt = 0;
    blocks->max_lag = 0;
    blocks->block_size_saved = 0;
    blocks->current_measurements = 0ULL;
    blocks->n_blocks = 0ULL;
    blocks->block_capacity = 0ULL;
    blocks->block_measurements = NULL;
    blocks->block_mean_y = NULL;
    blocks->block_mean_y2 = NULL;
    blocks->block_mean_y3 = NULL;
    blocks->block_mean_a = NULL;
    blocks->block_raw_y_y = NULL;
    blocks->block_raw_y2_y2 = NULL;
    blocks->block_raw_y3_y3 = NULL;
    blocks->block_raw_a_a = NULL;
    blocks->current_sum_y = 0.0;
    blocks->current_sum_y2 = 0.0;
    blocks->current_sum_y3 = 0.0;
    blocks->current_sum_a = 0.0;
    blocks->current_sum_y_y = NULL;
    blocks->current_sum_y2_y2 = NULL;
    blocks->current_sum_y3_y3 = NULL;
    blocks->current_sum_a_a = NULL;
}

int qho_spectrum_block_accumulator_accumulate(
    qho_spectrum_block_accumulator_t *blocks,
    const qho_lattice_t *lat
)
{
    int i;
    int lag;
    double y_sum = 0.0;
    double y2_sum = 0.0;
    double y3_sum = 0.0;
    double a_sum = 0.0;

    if (blocks == NULL || blocks->current_sum_y_y == NULL || blocks->current_sum_y2_y2 == NULL || blocks->current_sum_y3_y3 == NULL || blocks->current_sum_a_a == NULL) {
        return -1;
    }
    if (lat == NULL || lat->y == NULL || lat->nt != blocks->nt) {
        return -1;
    }

    for (i = 0; i < blocks->nt; ++i) {
        const double y = lat->y[i];
        const double y2 = y * y;
        const double y3 = op_y3(y);
        const double a = y3 - 1.5 * y;

        y_sum += y;
        y2_sum += y2;
        y3_sum += y3;
        a_sum += a;
    }

    blocks->current_sum_y += y_sum / (double)blocks->nt;
    blocks->current_sum_y2 += y2_sum / (double)blocks->nt;
    blocks->current_sum_y3 += y3_sum / (double)blocks->nt;
    blocks->current_sum_a += a_sum / (double)blocks->nt;

    for (lag = 0; lag <= blocks->max_lag; ++lag) {
        double raw_y = 0.0;
        double raw_y2 = 0.0;
        double raw_y3 = 0.0;
        double raw_a = 0.0;

        for (i = 0; i < blocks->nt; ++i) {
            const int j = (i + lag) % blocks->nt;
            const double yi = lat->y[i];
            const double yj = lat->y[j];
            const double yi2 = yi * yi;
            const double yj2 = yj * yj;
            const double yi3 = op_y3(yi);
            const double yj3 = op_y3(yj);
            const double ai = yi3 - 1.5 * yi;
            const double aj = yj3 - 1.5 * yj;

            raw_y += yi * yj;
            raw_y2 += yi2 * yj2;
            raw_y3 += yi3 * yj3;
            raw_a += ai * aj;
        }

        blocks->current_sum_y_y[lag] += raw_y / (double)blocks->nt;
        blocks->current_sum_y2_y2[lag] += raw_y2 / (double)blocks->nt;
        blocks->current_sum_y3_y3[lag] += raw_y3 / (double)blocks->nt;
        blocks->current_sum_a_a[lag] += raw_a / (double)blocks->nt;
    }

    ++blocks->current_measurements;
    if (blocks->current_measurements >= (unsigned long long)blocks->block_size_saved) {
        return append_current_block(blocks);
    }

    return 0;
}

int qho_spectrum_block_accumulator_write(
    qho_spectrum_block_accumulator_t *blocks,
    const char *path,
    const qho_params_t *params
)
{
    FILE *out;
    unsigned long long block_id;
    int lag;
    size_t n_lag;

    if (blocks == NULL || path == NULL || !spectrum_params_valid(params)) {
        return -1;
    }
    if (blocks->nt <= 1 || blocks->max_lag <= 0 || blocks->max_lag >= blocks->nt || blocks->block_size_saved <= 0) {
        return -1;
    }
    if (blocks->nt != params->nt || !spectrum_block_current_arrays_ready(blocks)) {
        return -1;
    }
    if (append_current_block(blocks) != 0 || blocks->n_blocks == 0ULL) {
        return -1;
    }
    if (blocks->n_blocks > blocks->block_capacity || !spectrum_block_arrays_ready(blocks) || !spectrum_block_values_finite(blocks)) {
        return -1;
    }

    n_lag = lag_count(blocks->max_lag);

    out = fopen(path, "w");
    if (out == NULL) {
        return -1;
    }

    fprintf(out, "# QHO_PIMC spectrum block correlators with y3 check\n");
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
    fprintf(out, "# spectrum_max_lag %d\n", blocks->max_lag);
    fprintf(out, "# spectrum_block_size_saved %d\n", blocks->block_size_saved);
    fprintf(out, "# spectrum_blocks %llu\n", blocks->n_blocks);
    fprintf(out, "# operators y y2 y3 a3\n");
    fprintf(out, "# y3_definition y^3\n");
    fprintf(out, "# a3_definition y^3 - 1.5*y\n");
    fprintf(out, "# connected_reconstruction subtract leave-one-block means in analysis\n");
    fprintf(out, "# columns block_id block_measurements lag tau mean_y mean_y2 mean_y3 mean_a raw_y_y raw_y2_y2 raw_y3_y3 raw_a_a\n");

    for (block_id = 0ULL; block_id < blocks->n_blocks; ++block_id) {
        const size_t offset = (size_t)block_id * n_lag;
        for (lag = 0; lag <= blocks->max_lag; ++lag) {
            const double tau = (double)lag * params->eta;
            fprintf(out, "%llu %llu %d %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n",
                block_id,
                blocks->block_measurements[block_id],
                lag,
                tau,
                blocks->block_mean_y[block_id],
                blocks->block_mean_y2[block_id],
                blocks->block_mean_y3[block_id],
                blocks->block_mean_a[block_id],
                blocks->block_raw_y_y[offset + (size_t)lag],
                blocks->block_raw_y2_y2[offset + (size_t)lag],
                blocks->block_raw_y3_y3[offset + (size_t)lag],
                blocks->block_raw_a_a[offset + (size_t)lag]);
        }
    }

    if (fclose(out) != 0) {
        return -1;
    }

    return 0;
}
