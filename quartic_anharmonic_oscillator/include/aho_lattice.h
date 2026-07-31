/*
 * Periodic Euclidean-lattice representation of one anharmonic-oscillator path.
 * The array y[j] has Nt sites; beta and eta=beta/Nt are carried by the run
 * parameters rather than stored with the coordinates.
 */

#ifndef AHO_LATTICE_H
#define AHO_LATTICE_H

#include "aho_params.h"
#include "pcg32.h"

typedef struct {
    int nt;
    double *y;
} aho_lattice_t;

/* Periodic index helpers implement the thermal boundary condition y_Nt=y_0. */
int aho_periodic_index(int i, int nt);
int aho_lattice_prev_index(const aho_lattice_t *lat, int i);
int aho_lattice_next_index(const aho_lattice_t *lat, int i);
/* Allocate and initialize trial paths used before thermalization. */
int aho_lattice_alloc(aho_lattice_t *lat, int nt);
void aho_lattice_free(aho_lattice_t *lat);
void aho_lattice_fill(aho_lattice_t *lat, double value);
void aho_lattice_init(aho_lattice_t *lat, aho_init_t init, pcg32_rng_t *rng);

#endif
