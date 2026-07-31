/*
 * Measure dimensionless one-path observables for V=y^2/2+lambda*y^4.  The
 * moments <y^2> and <y^4> determine the potential and virial estimators, while
 * <(Delta y)^2> supplies the short-distance lattice kinetic estimator later.
 */

#include "aho_measure.h"

/* Average each estimator over all Euclidean-time sites of one configuration. */
aho_observables_t aho_measure_observables(const aho_lattice_t *lat, double lambda)
{
    int i;
    double sy = 0.0;
    double sy2 = 0.0;
    double sy4 = 0.0;
    double sdy2 = 0.0;
    aho_observables_t obs;
    for (i = 0; i < lat->nt; ++i) {
        int ip = aho_lattice_next_index(lat, i);
        double y = lat->y[i];
        double y2 = y * y;
        double diff = lat->y[ip] - y;
        sy += y;
        sy2 += y2;
        sy4 += y2 * y2;
        sdy2 += diff * diff;
    }
    obs.y_mean = sy / lat->nt;
    obs.y2_mean = sy2 / lat->nt;
    obs.y4_mean = sy4 / lat->nt;
    obs.dy2_mean = sdy2 / lat->nt;
    /* For this potential, K_vir=<y V'(y)>/2 and E_vir=<V>+K_vir. */
    obs.v_mean = 0.5 * obs.y2_mean + lambda * obs.y4_mean;
    obs.k_virial = 0.5 * obs.y2_mean + 2.0 * lambda * obs.y4_mean;
    obs.e_virial = obs.y2_mean + 3.0 * lambda * obs.y4_mean;
    return obs;
}
