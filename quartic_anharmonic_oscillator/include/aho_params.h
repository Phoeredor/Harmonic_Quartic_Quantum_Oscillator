/*
 * Physical and sampling parameters for the quartic-oscillator PIMC calculation.
 * beta, Nt, and lambda determine the lattice action, with eta=beta/Nt; the
 * remaining fields control Markov updates and optional observable outputs.
 */

#ifndef AHO_PARAMS_H
#define AHO_PARAMS_H

#include <stdint.h>
#include <stdio.h>

typedef enum {
    AHO_INIT_ZERO = 0,
    AHO_INIT_GAUSSIAN = 1,
    AHO_INIT_UNIFORM = 2
} aho_init_t;

typedef enum {
    AHO_UPDATE_METRO = 0,
    AHO_UPDATE_METRO_OVER = 1
} aho_update_t;

typedef struct {
    double beta;
    int nt;
    double lambda;
    int n_therm;
    int n_sweeps;
    int meas_stride;
    uint64_t seed;
    double delta;
    aho_update_t update;
    int n_over;
    aho_init_t init;
    double hist_ymin;
    double hist_ymax;
    int hist_bins;
    const char *out;
    const char *hist_out;
    const char *path_out;
    const char *corr_out;
    int corr_max_dt;
    int corr_block_size;
} aho_params_t;

/* Parse and validate one dimensionless lattice ensemble specification. */
aho_params_t aho_params_default(void);
int aho_params_parse_args(aho_params_t *params, int argc, char **argv);
int aho_params_validate(const aho_params_t *params);
/* Return the Euclidean-time lattice spacing eta=beta/Nt. */
double aho_params_eta(const aho_params_t *params);
void aho_params_print_usage(FILE *stream, const char *program_name);
void aho_params_print_header(FILE *stream, const aho_params_t *params);
const char *aho_init_name(aho_init_t init);
const char *aho_update_name(aho_update_t update);

#endif
