/*
 * Write a small set of representative periodic configurations y(tau).  The
 * output visualizes fluctuations induced by the quartic interaction and the
 * resolution tau_j=j*eta, rather than an ensemble average.
 */

#include "aho_path_output.h"

#include <errno.h>
#include <string.h>

int aho_path_output_open(aho_path_output_t *out, const char *path, int max_paths)
{
    out->stream = NULL;
    out->saved = 0;
    out->max_paths = max_paths;
    if (!path) {
        return 1;
    }
    out->stream = fopen(path, "w");
    if (!out->stream) {
        fprintf(stderr, "[ERROR] cannot open %s: %s\n", path, strerror(errno));
        return 0;
    }
    if (fprintf(out->stream, "# path_index tau y\n") < 0) {
        fprintf(stderr, "[ERROR] cannot write header to %s: %s\n", path, strerror(errno));
        fclose(out->stream);
        out->stream = NULL;
        return 0;
    }
    return 1;
}

int aho_path_output_close(aho_path_output_t *out)
{
    int ok = 1;
    if (out->stream) {
        if (fclose(out->stream) != 0) {
            fprintf(stderr, "[ERROR] cannot close path output: %s\n", strerror(errno));
            ok = 0;
        }
    }
    out->stream = NULL;
    return ok;
}

/* Pair each lattice coordinate with its dimensionless Euclidean time tau=j*eta. */
int aho_path_output_maybe_write(aho_path_output_t *out, const aho_lattice_t *lat, double eta)
{
    int i;
    if (!out->stream || out->saved >= out->max_paths) {
        return 1;
    }
    for (i = 0; i < lat->nt; ++i) {
        if (fprintf(out->stream, "%d %.17g %.17g\n", out->saved, i * eta, lat->y[i]) < 0) {
            fprintf(stderr, "[ERROR] cannot write path output row: %s\n", strerror(errno));
            return 0;
        }
    }
    if (fprintf(out->stream, "\n") < 0) {
        fprintf(stderr, "[ERROR] cannot write path output separator: %s\n", strerror(errno));
        return 0;
    }
    out->saved++;
    return ferror(out->stream) == 0;
}
