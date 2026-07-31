/*
 * Project: Quantum Harmonic Oscillator PIMC
 * File: include/qho_update.h
 * Purpose: Public interface for Markov-chain update kernels.
 *
 * This module declares the local update kernels used to sample periodic
 * Euclidean paths of the quantum harmonic oscillator. The sampled probability
 * distribution is proportional to exp(-S_E), with lattice action
 *
 *     S_E[y] =
 *     sum_j [ (y_{j+1} - y_j)^2 / (2 eta) + eta y_j^2 / 2 ],
 *
 * where eta = beta / Nt and periodic boundary conditions are used.
 *
 * The harmonic action is quadratic, so both a local Metropolis update and an
 * exact Gaussian heatbath update are available. Overrelaxation sweeps can be
 * combined with heatbath sweeps to reduce autocorrelation times.
 */

#ifndef QHO_UPDATE_H
#define QHO_UPDATE_H

#include "pcg32.h"
#include "qho_lattice.h"
#include "qho_params.h"

/* -------------------------------------------------------------------------
 * Section: action evaluation
 * ------------------------------------------------------------------------- */

/*
 * Return the full Euclidean lattice action of the current path.
 *
 * The action includes all nearest-neighbor kinetic terms and all harmonic
 * potential terms. Periodic boundary conditions are used when connecting the
 * last time slice to the first one.
 */
double qho_action_total(const qho_lattice_t *lat, const qho_params_t *params);

/*
 * Return the part of the action that depends on site i.
 *
 * The local action contains the harmonic potential at site i and the two
 * kinetic links connecting i to its nearest neighbors. It is used to evaluate
 * the Metropolis acceptance probability for a single-site proposal.
 */
double qho_local_action(const qho_lattice_t *lat, const qho_params_t *params, int i);

/* -------------------------------------------------------------------------
 * Section: local Metropolis update
 * ------------------------------------------------------------------------- */

/*
 * Perform one local Metropolis proposal at site i.
 *
 * The proposal scale is controlled by params->delta. The random-number stream
 * is advanced by the proposal and accept-reject step.
 *
 * Returns 1 if the proposal is accepted and 0 if it is rejected.
 */
int qho_metropolis_site(
    qho_lattice_t *lat,
    const qho_params_t *params,
    pcg32_rng_t *pcg,
    int i
);

/*
 * Sweep once over the lattice with local Metropolis proposals.
 *
 * Returns the fraction of accepted site updates in the sweep.
 */
double qho_metropolis_sweep(
    qho_lattice_t *lat,
    const qho_params_t *params,
    pcg32_rng_t *pcg
);

/* -------------------------------------------------------------------------
 * Section: exact local heatbath update
 * ------------------------------------------------------------------------- */

/*
 * Replace site i with a draw from its exact conditional Gaussian distribution.
 *
 * For the quadratic harmonic action, the conditional probability of y_i at
 * fixed neighboring sites is Gaussian. This update is rejection-free and
 * advances the supplied random-number stream.
 */
void qho_heatbath_site(
    qho_lattice_t *lat,
    const qho_params_t *params,
    pcg32_rng_t *pcg,
    int i
);

/*
 * Sweep once over the lattice with exact local heatbath updates.
 *
 * The update is rejection-free, so no acceptance rate is returned.
 */
void qho_heatbath_sweep(
    qho_lattice_t *lat,
    const qho_params_t *params,
    pcg32_rng_t *pcg
);

/* -------------------------------------------------------------------------
 * Section: deterministic overrelaxation update
 * ------------------------------------------------------------------------- */

/*
 * Apply one local overrelaxation reflection at site i.
 *
 * The update reflects y_i around the mean of its local conditional Gaussian
 * distribution. For a quadratic action this transformation preserves the local
 * Boltzmann weight and is accepted deterministically.
 */
void qho_overrelax_site(
    qho_lattice_t *lat,
    const qho_params_t *params,
    int i
);

/*
 * Sweep once over the lattice with local overrelaxation reflections.
 *
 * The random-number stream is accepted as an argument for interface symmetry
 * with other sweep routines; the deterministic local reflections themselves do
 * not require random numbers.
 */
void qho_overrelax_sweep(
    qho_lattice_t *lat,
    const qho_params_t *params,
    pcg32_rng_t *pcg
);

/* -------------------------------------------------------------------------
 * Section: selected production update
 * ------------------------------------------------------------------------- */

/*
 * Perform one update sweep according to params->update_mode.
 *
 * For QHO_UPDATE_METRO this calls the Metropolis sweep and returns its
 * acceptance fraction. For rejection-free heatbath-based modes, the returned
 * value is a conventional acceptance indicator for the common output format.
 *
 * The lattice is modified in place and the random-number stream is advanced
 * whenever the selected update requires random numbers.
 */
double qho_update_sweep(
    qho_lattice_t *lat,
    const qho_params_t *params,
    pcg32_rng_t *pcg
);

#endif
