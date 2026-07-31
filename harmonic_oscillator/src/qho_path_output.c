/*
 * Project: Quantum Harmonic Oscillator PIMC
 * File: src/qho_path_output.c
 * Purpose: Path-output helpers for representative Euclidean configurations.
 * A sampled y_j is paired with tau_j = j eta to visualize how a periodic path
 * resolves the fixed interval [0, beta) at different lattice spacings.
 */

/* -------------------------------------------------------------------------
 * Section: includes and implementation
 * ------------------------------------------------------------------------- */

#include "qho_path_output.h"

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

/* Write one configuration for visualization, not an ensemble-averaged observable. */
int qho_write_path_snapshot(
    const qho_lattice_t *lat,
    const qho_params_t *params,
    const char *path
)
{
    int j;
    FILE *out;

    if (lat == NULL || params == NULL || path == NULL || path[0] == '\0') {
        return -1;
    }

    out = fopen(path, "w");
    if (out == NULL) {
        fprintf(stderr, "error: cannot open path snapshot file '%s': %s\n", path, strerror(errno));
        return -1;
    }

    fprintf(out, "# QHO_PIMC path snapshot\n");
    fprintf(out, "# beta %.17g\n", params->beta);
    fprintf(out, "# eta %.17g\n", params->eta);
    fprintf(out, "# nt %d\n", params->nt);
    fprintf(out, "# therm %ld\n", params->n_therm);
    fprintf(out, "# sweeps %ld\n", params->n_sweeps);
    fprintf(out, "# stride %ld\n", params->meas_stride);
    fprintf(out, "# seed %" PRIu64 "\n", params->seed);
    fprintf(out, "# stream %" PRIu64 "\n", params->stream);
    fprintf(out, "# init %s\n", qho_init_name(params->init));
    fprintf(out, "# update %s\n", qho_update_mode_name(params->update_mode));
    fprintf(out, "# n_over %d\n", params->n_overrelax);
    fprintf(out, "# columns j tau y\n");

    for (j = 0; j < lat->nt; ++j) {
        const double tau = params->eta * (double)j;
        if (fprintf(out, "%d %.17g %.17g\n", j, tau, lat->y[j]) < 0) {
            fclose(out);
            return -1;
        }
    }

    if (fclose(out) != 0) {
        fprintf(stderr, "error: failed to close path snapshot file '%s': %s\n", path, strerror(errno));
        return -1;
    }

    return 0;
}
