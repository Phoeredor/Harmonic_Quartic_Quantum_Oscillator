/*
 * Manage one periodic lattice path y_j for the Euclidean quartic oscillator.
 * The path has Nt coordinates; its spacing eta=beta/Nt enters through the action
 * and is therefore kept in the parameter structure.
 */

#include "aho_lattice.h"

#include <stdlib.h>

/* Map any integer site to the periodic thermal lattice. */
int aho_periodic_index(int i, int nt)
{
    int r = i % nt;
    return r < 0 ? r + nt : r;
}

int aho_lattice_prev_index(const aho_lattice_t *lat, int i)
{
    return i == 0 ? lat->nt - 1 : i - 1;
}

int aho_lattice_next_index(const aho_lattice_t *lat, int i)
{
    return i + 1 == lat->nt ? 0 : i + 1;
}

int aho_lattice_alloc(aho_lattice_t *lat, int nt)
{
    lat->nt = nt;
    lat->y = calloc((size_t)nt, sizeof(*lat->y));
    return lat->y != NULL;
}

void aho_lattice_free(aho_lattice_t *lat)
{
    free(lat->y);
    lat->y = NULL;
    lat->nt = 0;
}

void aho_lattice_fill(aho_lattice_t *lat, double value)
{
    int i;
    for (i = 0; i < lat->nt; ++i) {
        lat->y[i] = value;
    }
}

/* Prepare zero, Gaussian, or uniform trial paths before thermalization. */
void aho_lattice_init(aho_lattice_t *lat, aho_init_t init, pcg32_rng_t *rng)
{
    int i;
    if (init == AHO_INIT_ZERO) {
        aho_lattice_fill(lat, 0.0);
        return;
    }
    for (i = 0; i < lat->nt; ++i) {
        if (init == AHO_INIT_GAUSSIAN) {
            lat->y[i] = pcg32_gaussian(rng);
        } else {
            lat->y[i] = pcg32_uniform_range(rng, -1.0, 1.0);
        }
    }
}
