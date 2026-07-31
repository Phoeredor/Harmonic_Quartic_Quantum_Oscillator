/*
 * Project: Quantum Harmonic Oscillator PIMC
 * File: src/qho_measure.c
 * Purpose: Measurement routines for thermodynamic and path observables.
 * Path averages of y, y^2, and (Delta y)^2 feed the potential, kinetic, and
 * renormalized total-energy estimators in dimensionless oscillator units.
 */

/* -------------------------------------------------------------------------
 * Section: includes and implementation
 * ------------------------------------------------------------------------- */

#include "qho_measure.h"

#include <stdio.h>

/* Evaluate single-configuration estimators, averaging over Euclidean time. */
qho_measurements_t qho_measure_basic(const qho_lattice_t *lattice, const qho_params_t *params)
{
    qho_measurements_t measurements;
    int i;

    measurements.y_mean = 0.0;
    measurements.y2_mean = 0.0;
    measurements.dy2_mean = 0.0;
    measurements.potential = 0.0;
    measurements.kinetic_ren = 0.0;
    measurements.energy_ren = 0.0;

    if (lattice == NULL || lattice->y == NULL || lattice->nt <= 0 || params == NULL) {
        return measurements;
    }

    for (i = 0; i < lattice->nt; ++i) {
        const int next = qho_lattice_next_index(lattice, i);
        const double y = lattice->y[i];
        const double dy = lattice->y[next] - y;

        measurements.y_mean += y;
        measurements.y2_mean += y * y;
        measurements.dy2_mean += dy * dy;
    }

    measurements.y_mean /= (double)lattice->nt;
    measurements.y2_mean /= (double)lattice->nt;
    measurements.dy2_mean /= (double)lattice->nt;

    measurements.potential = 0.5 * measurements.y2_mean;
    /* The divergent 1/(2 eta) term cancels the short-distance path fluctuation. */
    measurements.kinetic_ren = 1.0 / (2.0 * params->eta)
        - measurements.dy2_mean / (2.0 * params->eta * params->eta);
    measurements.energy_ren = measurements.potential + measurements.kinetic_ren;

    return measurements;
}

void qho_measure_print(const qho_measurements_t *measurements)
{
    printf("  <y>              = %.17g\n", measurements->y_mean);
    printf("  <y^2>            = %.17g\n", measurements->y2_mean);
    printf("  <(Delta y)^2>    = %.17g\n", measurements->dy2_mean);
    printf("  V                = %.17g\n", measurements->potential);
    printf("  K_ren            = %.17g\n", measurements->kinetic_ren);
    printf("  H_ren            = %.17g\n", measurements->energy_ren);
}
