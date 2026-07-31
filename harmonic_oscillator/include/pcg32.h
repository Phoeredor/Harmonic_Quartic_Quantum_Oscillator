/*
 * Project: Quantum Harmonic Oscillator PIMC
 * File: include/pcg32.h
 * Purpose: Public interface for the PCG32 random-number generator.
 */

/* -------------------------------------------------------------------------
 * Section: public declarations
 * ------------------------------------------------------------------------- */

#ifndef QHO_PCG32_H
#define QHO_PCG32_H

/*
 * Compact PCG32 pseudo-random number generator wrapper for Monte Carlo
 * simulations.
 *
 * The underlying generator is PCG32: a 32-bit output PCG variant based on a
 * 64-bit linear-congruential state and the XSH-RR output permutation. The
 * algorithm follows Melissa E. O'Neill's public minimal C implementation of
 * PCG. The wrapper API, names, and cached Gaussian variate are project-specific
 * conveniences for QHO_PIMC.
 *
 * Uniform numbers are generated from PCG32 output words. Gaussian numbers are
 * generated from those uniforms with the polar Box-Muller method. This module
 * is not intended for cryptographic use.
 *
 * References:
 * 1. M. E. O'Neill, "PCG: A Family of Simple Fast Space-Efficient
 *    Statistically Good Algorithms for Random Number Generation", 2014.
 * 2. PCG official website and minimal C implementation:
 *    https://www.pcg-random.org/
 * 3. Polar Box-Muller transform for Gaussian random numbers.
 */

#include <stdint.h>

typedef struct {
    uint64_t state;
    uint64_t inc;
    int has_spare_gaussian;
    double spare_gaussian;
} pcg32_rng_t;

/* Initialize one reproducible stream; separate sequences support independent replicas. */
void pcg32_seed(pcg32_rng_t *pcg, uint64_t seed);
void pcg32_seed_sequence(pcg32_rng_t *pcg, uint64_t initstate, uint64_t initseq);
uint32_t pcg32_u32(pcg32_rng_t *pcg);
/* Continuous uniform variates lie in [0, 1) or the mapped interval [a, b). */
double pcg32_uniform(pcg32_rng_t *pcg);
double pcg32_uniform_range(pcg32_rng_t *pcg, double a, double b);
/* Return a standard normal variate for exact Gaussian heatbath sampling. */
double pcg32_gaussian(pcg32_rng_t *pcg);

#endif
