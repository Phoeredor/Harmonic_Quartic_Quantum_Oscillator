/*
 * Project: Quantum Harmonic Oscillator PIMC
 * File: src/qho_lattice.c
 * Purpose: Lattice allocation and Euclidean path utility routines.
 * The coordinate array represents y_j on a periodic lattice of Nt time slices;
 * eta and beta are carried separately by the simulation parameters.
 */

/* -------------------------------------------------------------------------
 * Section: includes and implementation
 * ------------------------------------------------------------------------- */

#include "qho_lattice.h"

#include <stdlib.h>

int qho_periodic_index(int i, int nt)
{
    int wrapped;

    if (nt <= 0) {
        return 0;
    }

    wrapped = i % nt;
    if (wrapped < 0) {
        wrapped += nt;
    }

    return wrapped;
}

int qho_lattice_init(qho_lattice_t *lattice, int nt)
{
    if (lattice == NULL || nt <= 0) {
        return -1;
    }

    lattice->nt = nt;
    lattice->y = calloc((size_t)nt, sizeof(*lattice->y));

    if (lattice->y == NULL) {
        lattice->nt = 0;
        return -1;
    }

    return 0;
}

void qho_lattice_free(qho_lattice_t *lattice)
{
    if (lattice == NULL) {
        return;
    }

    free(lattice->y);
    lattice->y = NULL;
    lattice->nt = 0;
}

void qho_lattice_fill(qho_lattice_t *lattice, double value)
{
    int i;

    if (lattice == NULL || lattice->y == NULL) {
        return;
    }

    for (i = 0; i < lattice->nt; ++i) {
        lattice->y[i] = value;
    }
}

/* A broad non-equilibrium path provides an alternative to the y_j = 0 start. */
void qho_lattice_init_random_uniform(qho_lattice_t *lattice, pcg32_rng_t *pcg)
{
    int i;

    if (lattice == NULL || lattice->y == NULL || pcg == NULL) {
        return;
    }

    for (i = 0; i < lattice->nt; ++i) {
        lattice->y[i] = pcg32_uniform_range(pcg, -1.0, 1.0);
    }
}

int qho_lattice_next_index(const qho_lattice_t *lattice, int i)
{
    return qho_periodic_index(i + 1, lattice->nt);
}

int qho_lattice_prev_index(const qho_lattice_t *lattice, int i)
{
    return qho_periodic_index(i - 1, lattice->nt);
}
