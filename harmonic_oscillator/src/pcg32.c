/*
 * Project: Quantum Harmonic Oscillator PIMC
 * File: src/pcg32.c
 * Purpose: PCG32 random-number generator implementation used by the Markov chains.
 * Uniform variates drive site selection and Metropolis decisions, while Gaussian
 * variates sample the exact local conditional distribution in heatbath updates.
 */

/* -------------------------------------------------------------------------
 * Section: includes and implementation
 * ------------------------------------------------------------------------- */

#include "pcg32.h"

/*
 * Compact PCG32 wrapper for QHO_PIMC.
 *
 * PCG32 combines a 64-bit LCG state update with an XSH-RR output permutation
 * to produce 32-bit random words. This implementation is based on Melissa E.
 * O'Neill's public minimal C implementation. The wrapper adds project-specific
 * naming and a cached spare Gaussian from the polar Box-Muller transform.
 *
 * The `initstate` value selects the initial state. The `initseq` value selects
 * the stream; PCG requires an odd LCG increment, hence `(initseq << 1u) | 1u`.
 * Distinct streams are useful for independent Monte Carlo replicas.
 *
 * References:
 * 1. M. E. O'Neill, "PCG: A Family of Simple Fast Space-Efficient
 *    Statistically Good Algorithms for Random Number Generation", 2014.
 * 2. PCG official website and minimal C implementation:
 *    https://www.pcg-random.org/
 * 3. Polar Box-Muller transform for Gaussian random numbers.
 *
 * This generator is not intended for cryptographic use.
 */

#include <math.h>

#define PCG32_MULTIPLIER 6364136223846793005ULL
#define PCG32_DEFAULT_STREAM 54ULL

void pcg32_seed_sequence(pcg32_rng_t *pcg, uint64_t initstate, uint64_t initseq)
{
    pcg->state = 0ULL;
    pcg->inc = (initseq << 1u) | 1u;
    pcg->has_spare_gaussian = 0;
    pcg->spare_gaussian = 0.0;

    (void)pcg32_u32(pcg);
    pcg->state += initstate;
    (void)pcg32_u32(pcg);
}

void pcg32_seed(pcg32_rng_t *pcg, uint64_t seed)
{
    pcg32_seed_sequence(pcg, seed, PCG32_DEFAULT_STREAM);
}

uint32_t pcg32_u32(pcg32_rng_t *pcg)
{
    const uint64_t previous_state = pcg->state;
    const uint32_t xorshifted = (uint32_t)(((previous_state >> 18u) ^ previous_state) >> 27u);
    const uint32_t rot = (uint32_t)(previous_state >> 59u);

    pcg->state = previous_state * PCG32_MULTIPLIER + pcg->inc;

    /* XSH-RR: xor-shift high bits, then rotate by state-dependent amount. */
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

double pcg32_uniform(pcg32_rng_t *pcg)
{
    /* Divide by 2^32, not 2^32-1, so the returned value is in [0, 1). */
    return (double)pcg32_u32(pcg) * (1.0 / 4294967296.0);
}

double pcg32_uniform_range(pcg32_rng_t *pcg, double a, double b)
{
    return a + (b - a) * pcg32_uniform(pcg);
}

double pcg32_gaussian(pcg32_rng_t *pcg)
{
    double u;
    double v;
    double s;
    double scale;

    if (pcg->has_spare_gaussian) {
        pcg->has_spare_gaussian = 0;
        return pcg->spare_gaussian;
    }

    do {
        u = pcg32_uniform_range(pcg, -1.0, 1.0);
        v = pcg32_uniform_range(pcg, -1.0, 1.0);
        s = u * u + v * v;
    } while (s <= 0.0 || s >= 1.0);

    scale = sqrt(-2.0 * log(s) / s);

    /* The polar Box-Muller transform produces two independent Gaussians. */
    pcg->spare_gaussian = v * scale;
    pcg->has_spare_gaussian = 1;
    return u * scale;
}
