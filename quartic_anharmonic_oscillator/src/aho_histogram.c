/*
 * Estimate the thermal position density P_beta(y) from sampled Euclidean paths.
 * Every time slice contributes by translational invariance, and the written
 * density is also symmetrized using the even quartic potential.
 */

#include "aho_histogram.h"

#include <math.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int aho_histogram_alloc(aho_histogram_t *hist, int bins, double ymin, double ymax)
{
    hist->bins = bins;
    hist->ymin = ymin;
    hist->ymax = ymax;
    hist->total = 0.0;
    hist->counts = calloc((size_t)bins, sizeof(*hist->counts));
    return hist->counts != NULL;
}

void aho_histogram_free(aho_histogram_t *hist)
{
    free(hist->counts);
    hist->counts = NULL;
    hist->bins = 0;
    hist->total = 0.0;
}

/* Treat all Nt coordinates of a saved path as samples of the one-point density. */
void aho_histogram_accumulate(aho_histogram_t *hist, const aho_lattice_t *lat)
{
    int i;
    double width = (hist->ymax - hist->ymin) / hist->bins;
    for (i = 0; i < lat->nt; ++i) {
        int bin = (int)floor((lat->y[i] - hist->ymin) / width);
        if (bin >= 0 && bin < hist->bins) {
            hist->counts[bin] += 1.0;
        }
    }
    hist->total += lat->nt;
}

/* Normalize counts to unit probability and average parity-related bins. */
int aho_histogram_write(const aho_histogram_t *hist, const aho_params_t *params)
{
    FILE *stream = fopen(params->hist_out, "w");
    int b;
    double width = (hist->ymax - hist->ymin) / hist->bins;
    double total = hist->total > 0.0 ? hist->total : 1.0;
    if (!stream) {
        fprintf(stderr, "[ERROR] cannot open %s: %s\n", params->hist_out, strerror(errno));
        return 0;
    }
    if (fprintf(stream,
                "# beta %.17g eta %.17g Nt %d lambda %.17g seed %llu bins %d ymin %.17g ymax %.17g\n",
                params->beta, aho_params_eta(params), params->nt, params->lambda,
                (unsigned long long)params->seed, hist->bins, hist->ymin, hist->ymax) < 0 ||
        fprintf(stream, "# bin_center density density_sym\n") < 0) {
        fprintf(stderr, "[ERROR] cannot write histogram header to %s: %s\n", params->hist_out, strerror(errno));
        fclose(stream);
        return 0;
    }
    for (b = 0; b < hist->bins; ++b) {
        double center = hist->ymin + (b + 0.5) * width;
        double density = hist->counts[b] / (total * width);
        double mirror = -center;
        int mb = (int)floor((mirror - hist->ymin) / width);
        double mdensity = (mb >= 0 && mb < hist->bins) ?
                          hist->counts[mb] / (total * width) : density;
        if (fprintf(stream, "%.17g %.17g %.17g\n", center, density, 0.5 * (density + mdensity)) < 0) {
            fprintf(stderr, "[ERROR] cannot write histogram row to %s: %s\n", params->hist_out, strerror(errno));
            fclose(stream);
            return 0;
        }
    }
    if (fclose(stream) != 0) {
        fprintf(stderr, "[ERROR] cannot close %s: %s\n", params->hist_out, strerror(errno));
        return 0;
    }
    return 1;
}
