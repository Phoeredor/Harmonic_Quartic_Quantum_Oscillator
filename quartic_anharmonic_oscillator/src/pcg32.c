/*
 * PCG32 implementation used by the anharmonic-oscillator Markov chain.  Uniform
 * variates determine local proposals and acceptances; Gaussian variates provide
 * an alternative initial path before equilibration.
 */

#include "pcg32.h"

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
    const uint64_t oldstate = pcg->state;
    const uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
    const uint32_t rot = (uint32_t)(oldstate >> 59u);

    pcg->state = oldstate * PCG32_MULTIPLIER + pcg->inc;

    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

double pcg32_uniform(pcg32_rng_t *pcg)
{
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

    pcg->spare_gaussian = v * scale;
    pcg->has_spare_gaussian = 1;
    return u * scale;
}
