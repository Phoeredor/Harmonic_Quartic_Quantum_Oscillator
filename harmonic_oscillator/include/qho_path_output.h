/*
 * Project: Quantum Harmonic Oscillator PIMC
 * File: include/qho_path_output.h
 * Purpose: Public interface for writing representative Euclidean paths.
 *
 * This module writes one sampled Euclidean path to a text file. The output is
 * intended for visualization of typical periodic configurations of the quantum
 * harmonic oscillator, not for estimating ensemble averages. Statistical
 * observables are written through the measurement, histogram, and correlator
 * output routines.
 */

#ifndef QHO_PATH_OUTPUT_H
#define QHO_PATH_OUTPUT_H

#include "qho_lattice.h"
#include "qho_params.h"

/* -------------------------------------------------------------------------
 * Section: path-snapshot output
 * ------------------------------------------------------------------------- */

/*
 * Write the current Euclidean path to a text file.
 *
 * The lattice contains the sampled coordinates y_j for j = 0, ..., Nt - 1.
 * The run parameters provide beta, Nt, and eta, so the output can include the
 * corresponding Euclidean-time coordinate tau_j = j eta.
 *
 * lat:
 *     Lattice configuration to write. The function does not modify it.
 *
 * params:
 *     Run parameters associated with the sampled path.
 *
 * path:
 *     Destination file path.
 *
 * Returns 0 on success and a nonzero value on I/O or consistency errors.
 */
int qho_write_path_snapshot(
    const qho_lattice_t *lat,
    const qho_params_t *params,
    const char *path
);

#endif
