/*
 * Local Markov updates for the discretized Euclidean action
 * S_E=sum_j[(y_{j+1}-y_j)^2/(2 eta)+eta(y_j^2/2+lambda*y_j^4)].
 * Metropolis and accepted overrelaxation moves sample the weight exp(-S_E).
 */

#ifndef AHO_UPDATE_H
#define AHO_UPDATE_H

#include "aho_lattice.h"
#include "aho_params.h"
#include "pcg32.h"

typedef struct {
    int metro_attempts;
    int metro_accepts;
    int over_attempts;
    int over_accepts;
} aho_update_counts_t;

/* Potential and action contributions are dimensionless oscillator quantities. */
double aho_potential(double y, double lambda);
double aho_local_action_value(double yi, double ym, double yp, double eta, double lambda);
double aho_local_action(const aho_lattice_t *lat, const aho_params_t *params, int i);
double aho_total_action(const aho_lattice_t *lat, const aho_params_t *params);
/* Site and sweep updates return acceptance counts for sampling statistics. */
int aho_metropolis_site(aho_lattice_t *lat, const aho_params_t *params, pcg32_rng_t *rng, int i);
int aho_metropolis_sweep(aho_lattice_t *lat, const aho_params_t *params, pcg32_rng_t *rng);
int aho_overrelax_site(aho_lattice_t *lat, const aho_params_t *params, pcg32_rng_t *rng, int i);
int aho_overrelax_sweep(aho_lattice_t *lat, const aho_params_t *params, pcg32_rng_t *rng);
aho_update_counts_t aho_update_sweep(aho_lattice_t *lat, const aho_params_t *params, pcg32_rng_t *rng);

#endif
