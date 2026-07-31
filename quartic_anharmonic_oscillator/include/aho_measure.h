/*
 * Single-path observables for V(y)=y^2/2+lambda*y^4.  Euclidean-time averages
 * of y^2, y^4, and nearest-neighbor fluctuations support thermodynamic,
 * renormalized-kinetic, and virial analyses.
 */

#ifndef AHO_MEASURE_H
#define AHO_MEASURE_H

#include "aho_lattice.h"

typedef struct {
    double y_mean;
    double y2_mean;
    double y4_mean;
    double dy2_mean;
    double v_mean;
    double k_virial;
    double e_virial;
} aho_observables_t;

/* Return dimensionless path averages and the potential/virial estimators. */
aho_observables_t aho_measure_observables(const aho_lattice_t *lat, double lambda);

#endif
