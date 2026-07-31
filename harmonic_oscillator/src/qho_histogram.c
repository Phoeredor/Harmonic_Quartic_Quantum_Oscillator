/*
 * Project: Quantum Harmonic Oscillator PIMC
 * File: src/qho_histogram.c
 * Purpose: Histogram accumulation routines for position-probability estimates.
 * All Euclidean-time sites contribute samples of P_beta(y); block histograms and
 * moments retain the information needed for correlated uncertainty estimates.
 */

/* -------------------------------------------------------------------------
 * Section: includes and implementation
 * ------------------------------------------------------------------------- */

#include "qho_histogram.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define QHO_HISTOGRAM_PI 3.141592653589793238462643383279502884

/* Exact continuum variance <y^2> = coth(beta/2)/2 in oscillator units. */
static double exact_y_variance(double beta)
{
    if (beta > 700.0) {
        return 0.5;
    }
    return 0.5 + 1.0 / expm1(beta);
}

/* Thermal position density of the continuum harmonic oscillator. */
static double exact_density(double y, double beta)
{
    const double var = exact_y_variance(beta);
    const double norm = 1.0 / sqrt(2.0 * QHO_HISTOGRAM_PI * var);
    return norm * exp(-0.5 * y * y / var);
}

int qho_histogram_init(qho_histogram_t *hist, int bins, double y_min, double y_max)
{
    if (hist == NULL || bins <= 0 || !(y_min < y_max)) {
        return -1;
    }

    hist->bins = bins;
    hist->y_min = y_min;
    hist->y_max = y_max;
    hist->bin_width = (y_max - y_min) / (double)bins;
    hist->counts = calloc((size_t)bins, sizeof(*hist->counts));
    hist->total_samples = 0ULL;
    hist->underflow = 0ULL;
    hist->overflow = 0ULL;

    if (hist->counts == NULL) {
        hist->bins = 0;
        hist->bin_width = 0.0;
        return -1;
    }

    return 0;
}

void qho_histogram_free(qho_histogram_t *hist)
{
    if (hist == NULL) {
        return;
    }

    free(hist->counts);
    hist->counts = NULL;
    hist->bins = 0;
    hist->total_samples = 0ULL;
    hist->underflow = 0ULL;
    hist->overflow = 0ULL;
}

void qho_histogram_accumulate_value(qho_histogram_t *hist, double y)
{
    int bin;

    if (hist == NULL || hist->counts == NULL) {
        return;
    }

    ++hist->total_samples;

    if (!isfinite(y) || y < hist->y_min) {
        ++hist->underflow;
        return;
    }

    if (y >= hist->y_max) {
        ++hist->overflow;
        return;
    }

    bin = (int)((y - hist->y_min) / hist->bin_width);
    if (bin < 0) {
        bin = 0;
    } else if (bin >= hist->bins) {
        bin = hist->bins - 1;
    }

    ++hist->counts[bin];
}

/* Translational invariance permits every time slice of a saved path to contribute. */
void qho_histogram_accumulate_lattice(qho_histogram_t *hist, const qho_lattice_t *lat)
{
    int i;

    if (hist == NULL || lat == NULL || lat->y == NULL) {
        return;
    }

    for (i = 0; i < lat->nt; ++i) {
        qho_histogram_accumulate_value(hist, lat->y[i]);
    }
}

int qho_histogram_write_density(
    const qho_histogram_t *hist,
    const char *path,
    const qho_params_t *params
)
{
    int i;
    FILE *out;

    if (hist == NULL || hist->counts == NULL || path == NULL || params == NULL) {
        return -1;
    }
    if (hist->total_samples == 0ULL) {
        return -1;
    }

    out = fopen(path, "w");
    if (out == NULL) {
        return -1;
    }

    fprintf(out, "# QHO_PIMC position histogram\n");
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
    fprintf(out, "# hist_bins %d\n", hist->bins);
    fprintf(out, "# hist_min %.17g\n", hist->y_min);
    fprintf(out, "# hist_max %.17g\n", hist->y_max);
    fprintf(out, "# total_samples %llu\n", hist->total_samples);
    fprintf(out, "# underflow %llu\n", hist->underflow);
    fprintf(out, "# overflow %llu\n", hist->overflow);
    fprintf(out, "# columns bin_center density count exact_density\n");

    for (i = 0; i < hist->bins; ++i) {
        const double center = hist->y_min + ((double)i + 0.5) * hist->bin_width;
        const double density = (double)hist->counts[i]
            / ((double)hist->total_samples * hist->bin_width);
        const double exact = exact_density(center, params->beta);

        fprintf(out, "%.17g %.17g %llu %.17g\n",
            center,
            density,
            hist->counts[i],
            exact);
    }

    if (fclose(out) != 0) {
        return -1;
    }

    return 0;
}

static void histogram_block_reset(qho_histogram_block_accumulator_t *blocks)
{
    int bin;
    blocks->current_measurements = 0ULL;
    blocks->current_underflow = 0ULL;
    blocks->current_overflow = 0ULL;
    blocks->current_sum_y = 0.0;
    blocks->current_sum_y2 = 0.0;
    blocks->current_sum_y4 = 0.0;
    for (bin = 0; bin < blocks->bins; ++bin) {
        blocks->current_counts[bin] = 0ULL;
    }
}

static int histogram_block_reserve(qho_histogram_block_accumulator_t *blocks)
{
    unsigned long long capacity;
    void *ptr;

    if (blocks->n_blocks < blocks->block_capacity) return 0;
    capacity = blocks->block_capacity == 0ULL ? 8ULL : 2ULL * blocks->block_capacity;

#define GROW(field, count) do { \
    ptr = realloc(blocks->field, (size_t)(count) * sizeof(*blocks->field)); \
    if (ptr == NULL) return -1; \
    blocks->field = ptr; \
} while (0)
    GROW(block_measurements, capacity);
    GROW(block_underflow, capacity);
    GROW(block_overflow, capacity);
    GROW(block_mean_y, capacity);
    GROW(block_mean_y2, capacity);
    GROW(block_mean_y4, capacity);
    GROW(block_counts, capacity * (unsigned long long)blocks->bins);
#undef GROW

    blocks->block_capacity = capacity;
    return 0;
}

/* Preserve one statistically contiguous block of histogram counts and moments. */
static int histogram_block_append(qho_histogram_block_accumulator_t *blocks)
{
    unsigned long long block_id;
    double inv_samples;
    int bin;

    if (blocks->current_measurements == 0ULL) return 0;
    if (histogram_block_reserve(blocks) != 0) return -1;

    block_id = blocks->n_blocks;
    inv_samples = 1.0 / ((double)blocks->current_measurements * (double)blocks->nt);
    blocks->block_measurements[block_id] = blocks->current_measurements;
    blocks->block_underflow[block_id] = blocks->current_underflow;
    blocks->block_overflow[block_id] = blocks->current_overflow;
    blocks->block_mean_y[block_id] = blocks->current_sum_y * inv_samples;
    blocks->block_mean_y2[block_id] = blocks->current_sum_y2 * inv_samples;
    blocks->block_mean_y4[block_id] = blocks->current_sum_y4 * inv_samples;
    for (bin = 0; bin < blocks->bins; ++bin) {
        blocks->block_counts[(size_t)block_id * (size_t)blocks->bins + (size_t)bin]
            = blocks->current_counts[bin];
    }
    ++blocks->n_blocks;
    histogram_block_reset(blocks);
    return 0;
}

int qho_histogram_block_init(
    qho_histogram_block_accumulator_t *blocks,
    int nt,
    int bins,
    double y_min,
    double y_max,
    int block_size_saved
)
{
    if (blocks == NULL || nt <= 1 || bins <= 0 || !(y_min < y_max) || block_size_saved <= 0) return -1;
    blocks->nt = nt;
    blocks->bins = bins;
    blocks->y_min = y_min;
    blocks->y_max = y_max;
    blocks->bin_width = (y_max - y_min) / (double)bins;
    blocks->block_size_saved = block_size_saved;
    blocks->current_counts = calloc((size_t)bins, sizeof(*blocks->current_counts));
    blocks->block_measurements = NULL;
    blocks->block_underflow = NULL;
    blocks->block_overflow = NULL;
    blocks->block_counts = NULL;
    blocks->block_mean_y = NULL;
    blocks->block_mean_y2 = NULL;
    blocks->block_mean_y4 = NULL;
    blocks->n_blocks = 0ULL;
    blocks->block_capacity = 0ULL;
    if (blocks->current_counts == NULL) return -1;
    histogram_block_reset(blocks);
    return 0;
}

void qho_histogram_block_free(qho_histogram_block_accumulator_t *blocks)
{
    if (blocks == NULL) return;
    free(blocks->current_counts);
    free(blocks->block_measurements);
    free(blocks->block_underflow);
    free(blocks->block_overflow);
    free(blocks->block_counts);
    free(blocks->block_mean_y);
    free(blocks->block_mean_y2);
    free(blocks->block_mean_y4);
    blocks->current_counts = NULL;
    blocks->block_measurements = NULL;
    blocks->block_underflow = NULL;
    blocks->block_overflow = NULL;
    blocks->block_counts = NULL;
    blocks->block_mean_y = NULL;
    blocks->block_mean_y2 = NULL;
    blocks->block_mean_y4 = NULL;
    blocks->n_blocks = 0ULL;
    blocks->block_capacity = 0ULL;
}

int qho_histogram_block_accumulate(
    qho_histogram_block_accumulator_t *blocks,
    const qho_lattice_t *lat
)
{
    int i;
    if (blocks == NULL || blocks->current_counts == NULL || lat == NULL || lat->y == NULL || lat->nt != blocks->nt) return -1;
    for (i = 0; i < blocks->nt; ++i) {
        const double y = lat->y[i];
        const double y2 = y * y;
        int bin;
        if (!isfinite(y)) return -1;
        blocks->current_sum_y += y;
        blocks->current_sum_y2 += y2;
        blocks->current_sum_y4 += y2 * y2;
        if (y < blocks->y_min) {
            ++blocks->current_underflow;
        } else if (y >= blocks->y_max) {
            ++blocks->current_overflow;
        } else {
            bin = (int)((y - blocks->y_min) / blocks->bin_width);
            if (bin < 0 || bin >= blocks->bins) return -1;
            ++blocks->current_counts[bin];
        }
    }
    ++blocks->current_measurements;
    if (blocks->current_measurements == (unsigned long long)blocks->block_size_saved) {
        return histogram_block_append(blocks);
    }
    return 0;
}

int qho_histogram_block_write(
    qho_histogram_block_accumulator_t *blocks,
    const char *path,
    const qho_params_t *params
)
{
    FILE *out;
    unsigned long long block_id;
    int bin;
    unsigned long long total_underflow = 0ULL;
    unsigned long long total_overflow = 0ULL;

    if (blocks == NULL || path == NULL || params == NULL) return -1;
    if (histogram_block_append(blocks) != 0 || blocks->n_blocks == 0ULL) return -1;
    out = fopen(path, "w");
    if (out == NULL) return -1;
    for (block_id = 0ULL; block_id < blocks->n_blocks; ++block_id) {
        total_underflow += blocks->block_underflow[block_id];
        total_overflow += blocks->block_overflow[block_id];
    }
    fprintf(out, "# QHO_PIMC position histogram blocks\n");
    fprintf(out, "# schema_version 1\n");
    fprintf(out, "# beta %.17g\n# nt %d\n# eta %.17g\n", params->beta, params->nt, params->eta);
    fprintf(out, "# therm %ld\n# sweeps %ld\n# stride %ld\n# measure_every %ld\n", params->n_therm, params->n_sweeps, params->meas_stride, params->meas_stride);
    fprintf(out, "# seed %llu\n# stream %llu\n", (unsigned long long)params->seed, (unsigned long long)params->stream);
    fprintf(out, "# init %s\n# update %s\n# n_over %d\n", qho_init_name(params->init), qho_update_mode_name(params->update_mode), params->n_overrelax);
    fprintf(out, "# n_bins %d\n# y_min %.17g\n# y_max %.17g\n# bin_width %.17g\n", blocks->bins, blocks->y_min, blocks->y_max, blocks->bin_width);
    fprintf(out, "# block_measurements_target %d\n# block_sweeps %ld\n# n_blocks %llu\n", blocks->block_size_saved, params->meas_stride * (long)blocks->block_size_saved, blocks->n_blocks);
    fprintf(out, "# underflow_count %llu\n# overflow_count %llu\n# use_all_timeslices true\n", total_underflow, total_overflow);
    fprintf(out, "# columns block_id block_measurements block_site_samples bin_id bin_left bin_right bin_center count probability_mass probability_density underflow_count overflow_count block_mean_y block_mean_y2 block_mean_y4\n");
    for (block_id = 0ULL; block_id < blocks->n_blocks; ++block_id) {
        const unsigned long long site_samples = blocks->block_measurements[block_id] * (unsigned long long)blocks->nt;
        for (bin = 0; bin < blocks->bins; ++bin) {
            const double left = blocks->y_min + (double)bin * blocks->bin_width;
            const double right = left + blocks->bin_width;
            const unsigned long long count = blocks->block_counts[(size_t)block_id * (size_t)blocks->bins + (size_t)bin];
            const double mass = (double)count / (double)site_samples;
            fprintf(out, "%llu %llu %llu %d %.17g %.17g %.17g %llu %.17g %.17g %llu %llu %.17g %.17g %.17g\n",
                block_id, blocks->block_measurements[block_id], site_samples, bin, left, right,
                0.5 * (left + right), count, mass, mass / blocks->bin_width,
                blocks->block_underflow[block_id], blocks->block_overflow[block_id],
                blocks->block_mean_y[block_id], blocks->block_mean_y2[block_id], blocks->block_mean_y4[block_id]);
        }
    }
    return fclose(out) == 0 ? 0 : -1;
}
