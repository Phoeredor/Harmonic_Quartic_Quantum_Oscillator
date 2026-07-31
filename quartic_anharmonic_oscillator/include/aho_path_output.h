/*
 * Output interface for representative periodic Euclidean paths y(tau).  These
 * configurations illustrate lattice resolution and anharmonic fluctuations;
 * they are not ensemble-averaged observables.
 */

#ifndef AHO_PATH_OUTPUT_H
#define AHO_PATH_OUTPUT_H

#include <stdio.h>

#include "aho_lattice.h"

typedef struct {
    FILE *stream;
    int saved;
    int max_paths;
} aho_path_output_t;

/* Write at most max_paths configurations with tau=j*eta. */
int aho_path_output_open(aho_path_output_t *out, const char *path, int max_paths);
int aho_path_output_close(aho_path_output_t *out);
int aho_path_output_maybe_write(aho_path_output_t *out, const aho_lattice_t *lat, double eta);

#endif
