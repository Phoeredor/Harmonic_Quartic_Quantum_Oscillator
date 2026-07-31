/*
 * Project: Quantum Harmonic Oscillator PIMC
 * File: src/qho_logbin.c
 * Purpose: Logarithmic binning utilities for Monte Carlo time-series checks.
 * Growing sweep intervals summarize relaxation of <y>, <y^2>, and the nearest-
 * neighbor fluctuation <(Delta y)^2> from the chosen initial path.
 */

/* -------------------------------------------------------------------------
 * Section: includes and implementation
 * ------------------------------------------------------------------------- */

#include "qho_logbin.h"

#include <math.h>
#include <stdio.h>

/* Advance to a wider Monte Carlo-time interval while guaranteeing progress. */
static long next_end(long end, double base)
{
    long candidate = (long)((double)end * base) + 1L;
    if (candidate <= end) {
        candidate = end + 1L;
    }
    return candidate;
}

static int write_observable(FILE *out, const char *init_name, const char *observable,
    const qho_logbin_accumulator_t *acc, double sum, double sumsq)
{
    const double mean = sum / (double)acc->count;
    double err = 0.0;
    const double center = sqrt((double)acc->start * (double)acc->end);
    if (acc->count > 1L) {
        const double var = (sumsq - sum * sum / (double)acc->count) / (double)(acc->count - 1L);
        err = sqrt((var > 0.0 ? var : 0.0) / (double)acc->count);
    }
    if (fprintf(out, "%s %s %ld %ld %.17g %.17g %.17g %ld\n",
            init_name, observable, acc->start, acc->end, center, mean, err, acc->count) < 0) {
        fprintf(stderr, "error: failed while writing log-bin output\n");
        return -1;
    }
    return 0;
}

static int flush_current(qho_logbin_accumulator_t *acc, FILE *out, const char *init_name)
{
    if (acc->count <= 0L) {
        return 0;
    }
    if (write_observable(out, init_name, "y_mean", acc, acc->sum_y, acc->sumsq_y) != 0) return -1;
    if (write_observable(out, init_name, "y2_mean", acc, acc->sum_y2, acc->sumsq_y2) != 0) return -1;
    if (write_observable(out, init_name, "dy2_mean", acc, acc->sum_dy2, acc->sumsq_dy2) != 0) return -1;
    return 0;
}

void qho_logbin_init(qho_logbin_accumulator_t *acc, double base)
{
    (void)base;
    acc->start = 1L;
    acc->end = 2L;
    acc->count = 0L;
    acc->sum_y = 0.0;
    acc->sumsq_y = 0.0;
    acc->sum_y2 = 0.0;
    acc->sumsq_y2 = 0.0;
    acc->sum_dy2 = 0.0;
    acc->sumsq_dy2 = 0.0;
}

int qho_logbin_accumulate(qho_logbin_accumulator_t *acc, long sweep, const qho_measurements_t *m, double base, FILE *out, const char *init_name)
{
    while (sweep > acc->end) {
        if (flush_current(acc, out, init_name) != 0) {
            return -1;
        }
        acc->start = acc->end;
        acc->end = next_end(acc->end, base);
        acc->count = 0L;
        acc->sum_y = acc->sumsq_y = 0.0;
        acc->sum_y2 = acc->sumsq_y2 = 0.0;
        acc->sum_dy2 = acc->sumsq_dy2 = 0.0;
    }
    acc->count += 1L;
    acc->sum_y += m->y_mean;
    acc->sumsq_y += m->y_mean * m->y_mean;
    acc->sum_y2 += m->y2_mean;
    acc->sumsq_y2 += m->y2_mean * m->y2_mean;
    acc->sum_dy2 += m->dy2_mean;
    acc->sumsq_dy2 += m->dy2_mean * m->dy2_mean;
    return 0;
}

int qho_logbin_flush(qho_logbin_accumulator_t *acc, double base, FILE *out, const char *init_name)
{
    (void)base;
    return flush_current(acc, out, init_name);
}

int qho_logbin_write_header(FILE *out, const qho_params_t *params)
{
    if (fprintf(out, "# QHO_PIMC thermalization log bins\n") < 0) return -1;
    if (fprintf(out, "# beta %.17g\n", params->beta) < 0) return -1;
    if (fprintf(out, "# eta %.17g\n", params->eta) < 0) return -1;
    if (fprintf(out, "# nt %d\n", params->nt) < 0) return -1;
    if (fprintf(out, "# sweeps %ld\n", params->n_sweeps) < 0) return -1;
    if (fprintf(out, "# init %s\n", qho_init_name(params->init)) < 0) return -1;
    if (fprintf(out, "# update %s\n", qho_update_mode_name(params->update_mode)) < 0) return -1;
    if (fprintf(out, "# n_over %d\n", params->n_overrelax) < 0) return -1;
    if (fprintf(out, "# logbin_base %.17g\n", params->logbin_base) < 0) return -1;
    if (fprintf(out, "# columns init observable bin_start bin_end bin_center bin_mean bin_err n_points\n") < 0) return -1;
    return 0;
}
