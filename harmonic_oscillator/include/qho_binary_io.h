/*
 * Project: Quantum Harmonic Oscillator PIMC
 * File: include/qho_binary_io.h
 * Purpose: Binary output format for Monte Carlo measurements.
 *
 * This module defines the binary file layout used by the C executable to store
 * measured time series. The format is intentionally simple and fixed-size:
 *
 *   - one fixed-size header of QHO_BIN_HEADER_SIZE bytes;
 *   - followed by records of QHO_BIN_RECORD_SIZE bytes each.
 *
 * Keeping the layout explicit makes the files easy to read from the Python
 * analysis scripts and avoids ambiguity in long production runs.
 */

#ifndef QHO_BINARY_IO_H
#define QHO_BINARY_IO_H

#include <stdint.h>
#include <stdio.h>

#include "qho_measure.h"
#include "qho_params.h"

/* -------------------------------------------------------------------------
 * Section: binary file format constants
 * ------------------------------------------------------------------------- */

/*
 * Magic string written at the beginning of each binary output file.
 * Readers use it to reject files that are not QHO PIMC measurement files.
 */
#define QHO_BIN_MAGIC "QHOPIMC"

/*
 * Binary format version. Increase this number only if the header or record
 * layout changes in a way that readers must handle explicitly.
 */
#define QHO_BIN_VERSION 1U

/*
 * Fixed byte sizes of the file header and of each measurement record.
 *
 * These values must stay synchronized with both the C writer and the Python
 * binary reader.
 */
#define QHO_BIN_HEADER_SIZE 256U
#define QHO_BIN_RECORD_SIZE 48U

/*
 * Endianness marker written into the header. A reader can compare the stored
 * value with 0x01020304 to detect byte-order mismatches.
 */
#define QHO_BIN_ENDIAN_MARKER UINT32_C(0x01020304)

/* -------------------------------------------------------------------------
 * Section: observable bit masks
 * ------------------------------------------------------------------------- */

/*
 * Bit masks describing which observables are present in each binary record.
 * The default format stores all primary observables used by the public
 * thermodynamic analyses.
 */
#define QHO_BIN_OBS_Y_MEAN UINT32_C(1)
#define QHO_BIN_OBS_Y2_MEAN UINT32_C(2)
#define QHO_BIN_OBS_DY2_MEAN UINT32_C(4)
#define QHO_BIN_OBS_ENERGY_REN UINT32_C(8)
#define QHO_BIN_OBS_ACC_RATE UINT32_C(16)

#define QHO_BIN_OBS_DEFAULT \
    (QHO_BIN_OBS_Y_MEAN | \
     QHO_BIN_OBS_Y2_MEAN | \
     QHO_BIN_OBS_DY2_MEAN | \
     QHO_BIN_OBS_ENERGY_REN | \
     QHO_BIN_OBS_ACC_RATE)

/* -------------------------------------------------------------------------
 * Section: binary measurement record
 * ------------------------------------------------------------------------- */

/*
 * One saved Monte Carlo measurement.
 *
 * sweep:
 *     Monte Carlo sweep index at which the measurement was saved.
 *
 * y_mean:
 *     Path average of y over the Euclidean-time lattice.
 *
 * y2_mean:
 *     Path average of y^2. Its ensemble average estimates <y^2>.
 *
 * dy2_mean:
 *     Path average of (y_{j+1} - y_j)^2. This enters the kinetic-energy
 *     estimator and its lattice-spacing dependence.
 *
 * energy_ren:
 *     Renormalized energy estimator for the harmonic oscillator.
 *
 * acc_rate:
 *     Metropolis acceptance rate. For heatbath-based updates this field is
 *     still written for a uniform record layout.
 */
typedef struct {
    int64_t sweep;
    double y_mean;
    double y2_mean;
    double dy2_mean;
    double energy_ren;
    double acc_rate;
} qho_binary_record_t;

/* -------------------------------------------------------------------------
 * Section: writer interface
 * ------------------------------------------------------------------------- */

/*
 * Return the number of measurement records expected from the run parameters.
 */
unsigned long long qho_expected_records(const qho_params_t *params);

/*
 * Check whether the output path can accommodate the expected number of bytes.
 *
 * Returns 0 on success and a nonzero value if the check fails.
 */
int qho_preflight_output_space(const char *path, unsigned long long expected_bytes);

/*
 * Write the fixed-size binary header to an already opened output stream.
 *
 * Returns 0 on success and a nonzero value on I/O error.
 */
int qho_binary_write_header(FILE *out, const qho_params_t *params);

/*
 * Write one fixed-size measurement record to an already opened output stream.
 *
 * Returns 0 on success and a nonzero value on I/O error.
 */
int qho_binary_write_record(
    FILE *out,
    int64_t sweep,
    const qho_measurements_t *m,
    double acc_rate
);

/*
 * Build the staging path used before replacing the final output file.
 *
 * The caller owns the character buffer and must provide its size. The function
 * returns 0 on success and a nonzero value if the path does not fit.
 */
int qho_write_staging_path(
    const char *path,
    char *staging_path,
    size_t staging_path_size
);

#endif
