/*
 * Block correlator interface for the quartic oscillator.  Matrices formed from
 * the odd basis (y,y^3) and even basis (y^2,y^4) are measured on periodic paths
 * for jackknife GEVP estimates of excitation gaps.
 */

#ifndef AHO_CORRELATOR_H
#define AHO_CORRELATOR_H

#include <stdio.h>

#include "aho_lattice.h"
#include "aho_params.h"

typedef struct {
    FILE *stream;
    double *corr_sums;
    double mean_sums[4];
    int max_dt;
    int block_size;
    int n_meas;
    int block_index;
} aho_correlator_t;

/* Open, accumulate, and finalize block-resolved Euclidean correlator output. */
int aho_correlator_open(aho_correlator_t *corr, const aho_params_t *params);
int aho_correlator_accumulate(aho_correlator_t *corr, const aho_lattice_t *lat,
                              const aho_params_t *params);
int aho_correlator_finish(aho_correlator_t *corr, const aho_params_t *params);
void aho_correlator_free(aho_correlator_t *corr);

#endif
