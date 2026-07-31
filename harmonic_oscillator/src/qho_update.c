/*
 * Project: Quantum Harmonic Oscillator PIMC
 * File: src/qho_update.c
 * Purpose: Local Metropolis, heatbath, and overrelaxation update routines.
 * Each kernel preserves the lattice path measure exp(-S_E); their differing
 * autocorrelations determine how efficiently equilibrium paths are sampled.
 */

/* -------------------------------------------------------------------------
 * Section: includes and implementation
 * ------------------------------------------------------------------------- */

#include "qho_update.h"

#include <math.h>

static int random_site(const qho_lattice_t *lat, pcg32_rng_t *pcg)
{
    int site = (int)(pcg32_uniform(pcg) * (double)lat->nt);

    if (site >= lat->nt) {
        site = lat->nt - 1;
    }

    return site;
}

/* Mean of the exact Gaussian distribution for y_i with its neighbors fixed. */
static double conditional_mean(const qho_lattice_t *lat, const qho_params_t *params, int i)
{
    const int im = qho_lattice_prev_index(lat, i);
    const int ip = qho_lattice_next_index(lat, i);
    const double eta = params->eta;
    const double denom = eta + 2.0 / eta;

    return ((lat->y[im] + lat->y[ip]) / eta) / denom;
}

/* Dimensionless discretized Euclidean action with periodic nearest neighbors. */
double qho_action_total(const qho_lattice_t *lat, const qho_params_t *params)
{
    int i;
    double action = 0.0;
    const double eta = params->eta;

    for (i = 0; i < lat->nt; ++i) {
        const int ip = qho_lattice_next_index(lat, i);
        const double yi = lat->y[i];
        const double dy = lat->y[ip] - yi;

        action += 0.5 * eta * yi * yi + 0.5 * dy * dy / eta;
    }

    return action;
}

/* Terms in S_E that depend on y_i; constants cancel in a local action change. */
double qho_local_action(const qho_lattice_t *lat, const qho_params_t *params, int i)
{
    const int im = qho_lattice_prev_index(lat, i);
    const int ip = qho_lattice_next_index(lat, i);
    const double eta = params->eta;
    const double c1 = 1.0 / eta;
    const double c2 = 1.0 / eta + 0.5 * eta;
    const double yi = lat->y[i];

    return c2 * yi * yi - c1 * yi * (lat->y[im] + lat->y[ip]);
}

/* Propose a displacement of width delta and accept it with exp(-Delta S_E). */
int qho_metropolis_site(qho_lattice_t *lat, const qho_params_t *params, pcg32_rng_t *pcg, int i)
{
    const double previous_y = lat->y[i];
    const double previous_action = qho_local_action(lat, params, i);
    const double trial_y = previous_y + params->delta * pcg32_uniform_range(pcg, -1.0, 1.0);
    double delta_s;

    lat->y[i] = trial_y;
    delta_s = qho_local_action(lat, params, i) - previous_action;

    if (delta_s <= 0.0 || log(pcg32_uniform(pcg)) < -delta_s) {
        return 1;
    }

    lat->y[i] = previous_y;
    return 0;
}

double qho_metropolis_sweep(qho_lattice_t *lat, const qho_params_t *params, pcg32_rng_t *pcg)
{
    int step;
    int accepted = 0;

    for (step = 0; step < lat->nt; ++step) {
        accepted += qho_metropolis_site(lat, params, pcg, random_site(lat, pcg));
    }

    return (double)accepted / (double)lat->nt;
}

/* Draw y_i directly from its Gaussian conditional distribution. */
void qho_heatbath_site(qho_lattice_t *lat, const qho_params_t *params, pcg32_rng_t *pcg, int i)
{
    const double eta = params->eta;
    const double denom = eta + 2.0 / eta;
    const double mean = conditional_mean(lat, params, i);
    const double std = 1.0 / sqrt(denom);

    lat->y[i] = mean + std * pcg32_gaussian(pcg);
}

void qho_heatbath_sweep(qho_lattice_t *lat, const qho_params_t *params, pcg32_rng_t *pcg)
{
    int step;

    for (step = 0; step < lat->nt; ++step) {
        qho_heatbath_site(lat, params, pcg, random_site(lat, pcg));
    }
}

/* Reflect y_i about its conditional mean, leaving the local action unchanged. */
void qho_overrelax_site(qho_lattice_t *lat, const qho_params_t *params, int i)
{
    const double mean = conditional_mean(lat, params, i);

    lat->y[i] = 2.0 * mean - lat->y[i];
}

void qho_overrelax_sweep(qho_lattice_t *lat, const qho_params_t *params, pcg32_rng_t *pcg)
{
    int step;

    for (step = 0; step < lat->nt; ++step) {
        qho_overrelax_site(lat, params, random_site(lat, pcg));
    }
}

double qho_update_sweep(qho_lattice_t *lat, const qho_params_t *params, pcg32_rng_t *pcg)
{
    int over;

    if (params->update_mode == QHO_UPDATE_HEATBATH) {
        qho_heatbath_sweep(lat, params, pcg);
        return 1.0;
    }

    if (params->update_mode == QHO_UPDATE_HB_OVER) {
        qho_heatbath_sweep(lat, params, pcg);
        for (over = 0; over < params->n_overrelax; ++over) {
            qho_overrelax_sweep(lat, params, pcg);
        }
        return 1.0;
    }

    return qho_metropolis_sweep(lat, params, pcg);
}
