/*
 * Project: Quantum Harmonic Oscillator PIMC
 * File: tools/seed_generator.c
 * Purpose: Generate reproducible PCG32 seed-stream pairs.
 * Distinct deterministic streams support statistically independent Monte Carlo
 * replicas at common physical and lattice parameters.
 */

/* -------------------------------------------------------------------------
 * Section: includes and implementation
 * ------------------------------------------------------------------------- */

#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Reproducible PCG32 seed/stream generator for independent Monte Carlo
 * replicas. This utility does not use OS randomness: a user-provided master
 * seed initializes SplitMix64, which deterministically expands it into
 * seed/stream pairs. Use distinct pairs for independent production replicas.
 */

typedef struct {
    uint64_t state;
} splitmix64_t;

static uint64_t splitmix64_next(splitmix64_t *sm)
{
    uint64_t z;

    sm->state += UINT64_C(0x9e3779b97f4a7c15);
    z = sm->state;
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

static void print_usage(FILE *stream, const char *program)
{
    fprintf(stream, "Usage: %s --n INT --master-seed UINT64\n", program);
}

static int parse_int(const char *text, int *value)
{
    char *end = NULL;
    errno = 0;
    const long parsed = strtol(text, &end, 10);

    if (errno != 0 || end == text || *end != '\0' || parsed <= 0L || parsed > 1000000L) {
        return -1;
    }

    *value = (int)parsed;
    return 0;
}

static int parse_u64(const char *text, uint64_t *value)
{
    char *end = NULL;
    errno = 0;
    const unsigned long long parsed = strtoull(text, &end, 10);

    if (errno != 0 || end == text || *end != '\0') {
        return -1;
    }

    *value = (uint64_t)parsed;
    return 0;
}

int main(int argc, char **argv)
{
    int i;
    int n = 0;
    int have_master_seed = 0;
    uint64_t master_seed = 0ULL;
    splitmix64_t sm;

    for (i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(stdout, argv[0]);
            return EXIT_SUCCESS;
        }

        if (i + 1 >= argc) {
            print_usage(stderr, argv[0]);
            return EXIT_FAILURE;
        }

        if (strcmp(argv[i], "--n") == 0) {
            if (parse_int(argv[++i], &n) != 0) {
                fprintf(stderr, "error: invalid --n value\n");
                return EXIT_FAILURE;
            }
        } else if (strcmp(argv[i], "--master-seed") == 0) {
            if (parse_u64(argv[++i], &master_seed) != 0) {
                fprintf(stderr, "error: invalid --master-seed value\n");
                return EXIT_FAILURE;
            }
            have_master_seed = 1;
        } else {
            fprintf(stderr, "error: unknown option: %s\n", argv[i]);
            print_usage(stderr, argv[0]);
            return EXIT_FAILURE;
        }
    }

    if (n <= 0 || !have_master_seed) {
        print_usage(stderr, argv[0]);
        return EXIT_FAILURE;
    }

    sm.state = master_seed;
    printf("# replica seed stream\n");
    for (i = 0; i < n; ++i) {
        const uint64_t seed = splitmix64_next(&sm);
        const uint64_t stream = splitmix64_next(&sm);
        printf("%d %" PRIu64 " %" PRIu64 "\n", i, seed, stream);
    }

    return EXIT_SUCCESS;
}
