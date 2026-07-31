/*
 * Sample periodic paths with weight exp(-S_E) for
 * V(y)=y^2/2+lambda*y^4.  Local Metropolis moves explore the interacting path
 * measure, and accepted reflections reduce correlations between configurations.
 */

#include "aho_update.h"

#include <math.h>

/* Dimensionless confining potential; lambda controls the anharmonic strength. */
double aho_potential(double y, double lambda)
{
    double y2 = y * y;
    return 0.5 * y2 + lambda * y2 * y2;
}

/* Action terms depending on y_i with its two periodic neighbors held fixed. */
double aho_local_action_value(double yi, double ym, double yp, double eta, double lambda)
{
    double left = yi - ym;
    double right = yp - yi;
    return (left * left + right * right) / (2.0 * eta) + eta * aho_potential(yi, lambda);
}

double aho_local_action(const aho_lattice_t *lat, const aho_params_t *params, int i)
{
    int im = aho_lattice_prev_index(lat, i);
    int ip = aho_lattice_next_index(lat, i);
    return aho_local_action_value(lat->y[i], lat->y[im], lat->y[ip],
                                  aho_params_eta(params), params->lambda);
}

/* Full discretized Euclidean action with eta=beta/Nt. */
double aho_total_action(const aho_lattice_t *lat, const aho_params_t *params)
{
    int i;
    double action = 0.0;
    double eta = aho_params_eta(params);
    for (i = 0; i < lat->nt; ++i) {
        int ip = aho_lattice_next_index(lat, i);
        double diff = lat->y[ip] - lat->y[i];
        action += diff * diff / (2.0 * eta) + eta * aho_potential(lat->y[i], params->lambda);
    }
    return action;
}

static int accept_delta(double delta_s, pcg32_rng_t *rng)
{
    return delta_s <= 0.0 || pcg32_uniform(rng) < exp(-delta_s);
}

/* Propose a local displacement of scale delta and accept it with exp(-Delta S_E). */
int aho_metropolis_site(aho_lattice_t *lat, const aho_params_t *params, pcg32_rng_t *rng, int i)
{
    int im = aho_lattice_prev_index(lat, i);
    int ip = aho_lattice_next_index(lat, i);
    double eta = aho_params_eta(params);
    double old_y = lat->y[i];
    double trial_y = old_y + params->delta * pcg32_uniform_range(rng, -1.0, 1.0);
    double delta_s = aho_local_action_value(trial_y, lat->y[im], lat->y[ip], eta, params->lambda) -
                     aho_local_action_value(old_y, lat->y[im], lat->y[ip], eta, params->lambda);
    if (accept_delta(delta_s, rng)) {
        lat->y[i] = trial_y;
        return 1;
    }
    return 0;
}

int aho_metropolis_sweep(aho_lattice_t *lat, const aho_params_t *params, pcg32_rng_t *rng)
{
    int i;
    int accepted = 0;
    for (i = 0; i < lat->nt; ++i) {
        accepted += aho_metropolis_site(lat, params, rng, i);
    }
    return accepted;
}

/* Reflect about the harmonic conditional mean, then accept the quartic action change. */
int aho_overrelax_site(aho_lattice_t *lat, const aho_params_t *params, pcg32_rng_t *rng, int i)
{
    int im = aho_lattice_prev_index(lat, i);
    int ip = aho_lattice_next_index(lat, i);
    double eta = aho_params_eta(params);
    double old_y = lat->y[i];
    double mu = (lat->y[im] + lat->y[ip]) / (2.0 + eta * eta);
    double trial_y = 2.0 * mu - old_y;
    double delta_s = aho_local_action_value(trial_y, lat->y[im], lat->y[ip], eta, params->lambda) -
                     aho_local_action_value(old_y, lat->y[im], lat->y[ip], eta, params->lambda);
    if (accept_delta(delta_s, rng)) {
        lat->y[i] = trial_y;
        return 1;
    }
    return 0;
}

int aho_overrelax_sweep(aho_lattice_t *lat, const aho_params_t *params, pcg32_rng_t *rng)
{
    int i;
    int accepted = 0;
    for (i = 0; i < lat->nt; ++i) {
        accepted += aho_overrelax_site(lat, params, rng, i);
    }
    return accepted;
}

aho_update_counts_t aho_update_sweep(aho_lattice_t *lat, const aho_params_t *params, pcg32_rng_t *rng)
{
    int k;
    aho_update_counts_t counts;
    counts.metro_attempts = lat->nt;
    counts.metro_accepts = aho_metropolis_sweep(lat, params, rng);
    counts.over_attempts = 0;
    counts.over_accepts = 0;
    if (params->update == AHO_UPDATE_METRO_OVER) {
        for (k = 0; k < params->n_over; ++k) {
            counts.over_accepts += aho_overrelax_sweep(lat, params, rng);
            counts.over_attempts += lat->nt;
        }
    }
    return counts;
}
