/*
 * Position-distribution interface for the quartic oscillator.  Coordinates from
 * all Euclidean-time slices estimate the thermal density P_beta(y), whose shape
 * changes with the quartic coupling lambda.
 */

#ifndef AHO_HISTOGRAM_H
#define AHO_HISTOGRAM_H

#include "aho_lattice.h"
#include "aho_params.h"

typedef struct {
    int bins;
    double ymin;
    double ymax;
    double total;
    double *counts;
} aho_histogram_t;

/* Allocate the y range, accumulate path coordinates, and write a normalized density. */
int aho_histogram_alloc(aho_histogram_t *hist, int bins, double ymin, double ymax);
void aho_histogram_free(aho_histogram_t *hist);
void aho_histogram_accumulate(aho_histogram_t *hist, const aho_lattice_t *lat);
int aho_histogram_write(const aho_histogram_t *hist, const aho_params_t *params);

#endif
