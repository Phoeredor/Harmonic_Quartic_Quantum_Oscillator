/*
 * Project: Quantum Harmonic Oscillator PIMC
 * File: include/qho_lattice.h
 * Purpose: Public interface for Euclidean-time lattice storage.
 *
 * This module defines the lattice representation of one periodic Euclidean
 * path of the quantum harmonic oscillator. A path is stored as the array
 *
 *     y[0], y[1], ..., y[nt - 1],
 *
 * with periodic boundary conditions y[nt] = y[0]. The lattice spacing eta is
 * not stored here; it is part of the run parameters. This file only owns the
 * discrete coordinates and provides small index utilities.
 */

#ifndef QHO_LATTICE_H
#define QHO_LATTICE_H

#include "pcg32.h"

/* -------------------------------------------------------------------------
 * Section: Euclidean path storage
 * ------------------------------------------------------------------------- */

/*
 * Storage for one periodic Euclidean path.
 *
 * nt:
 *     Number of Euclidean time slices.
 *
 * y:
 *     Array of length nt containing the coordinate y_j at each time slice.
 *     The array is allocated by qho_lattice_init and released by
 *     qho_lattice_free.
 */
typedef struct {
    int nt;
    double *y;
} qho_lattice_t;

/* -------------------------------------------------------------------------
 * Section: periodic index utilities
 * ------------------------------------------------------------------------- */

/*
 * Return the periodic representative of index i on a lattice of size nt.
 *
 * The returned value is always in the interval 0, ..., nt - 1. This is used to
 * implement the periodic boundary condition of the thermal path integral.
 */
int qho_periodic_index(int i, int nt);

/*
 * Return the next Euclidean-time index after i, with periodic wrapping.
 */
int qho_lattice_next_index(const qho_lattice_t *lattice, int i);

/*
 * Return the previous Euclidean-time index before i, with periodic wrapping.
 */
int qho_lattice_prev_index(const qho_lattice_t *lattice, int i);

/* -------------------------------------------------------------------------
 * Section: lattice allocation and initialization
 * ------------------------------------------------------------------------- */

/*
 * Allocate storage for a lattice path with nt time slices.
 *
 * Returns 0 on success and a nonzero value if nt is invalid or memory
 * allocation fails.
 */
int qho_lattice_init(qho_lattice_t *lattice, int nt);

/*
 * Release the coordinate array owned by the lattice and reset the structure.
 */
void qho_lattice_free(qho_lattice_t *lattice);

/*
 * Fill all coordinates of the path with the same value.
 *
 * This is used, for example, to initialize a cold path with y_j = 0.
 */
void qho_lattice_fill(qho_lattice_t *lattice, double value);

/*
 * Initialize the path with uniformly distributed coordinates.
 *
 * The random numbers are drawn from the supplied PCG32 stream. The precise
 * interval used for the uniform initialization is defined in the corresponding
 * implementation file.
 */
void qho_lattice_init_random_uniform(qho_lattice_t *lattice, pcg32_rng_t *pcg);

#endif
