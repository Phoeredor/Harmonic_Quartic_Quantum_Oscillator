/*
 * Random-number interface for quartic-oscillator path sampling.  Uniform
 * variates drive Metropolis decisions, while Gaussian variates initialize
 * nontrivial paths before equilibration.
 */

#ifndef ANHARMONIC_PCG32_H
#define ANHARMONIC_PCG32_H

/*
 * Compact PCG32 pseudo-random number generator wrapper for Monte Carlo
 * simulations. Copied/adapted from the harmonic oscillator project to keep
 * RNG conventions consistent across the PIMC codes.
 *
 * The underlying generator follows Melissa E. O'Neill's public minimal C
 * implementation of PCG32: a 32-bit output PCG variant based on a 64-bit LCG
 * state and XSH-RR output permutation. Gaussian numbers use polar Box-Muller.
 * This module is not intended for cryptographic use.
 */

#include <stdint.h>

typedef struct {
    uint64_t state;
    uint64_t inc;
    int has_spare_gaussian;
    double spare_gaussian;
} pcg32_rng_t;

void pcg32_seed(pcg32_rng_t *pcg, uint64_t seed);
void pcg32_seed_sequence(pcg32_rng_t *pcg, uint64_t initstate, uint64_t initseq);
uint32_t pcg32_u32(pcg32_rng_t *pcg);
/* Continuous uniforms lie in [0,1) or the mapped interval [a,b). */
double pcg32_uniform(pcg32_rng_t *pcg);
double pcg32_uniform_range(pcg32_rng_t *pcg, double a, double b);
/* Return a standard normal variate. */
double pcg32_gaussian(pcg32_rng_t *pcg);

#endif
