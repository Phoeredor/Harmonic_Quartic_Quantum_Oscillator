/*
 * Project: Quantum Harmonic Oscillator PIMC
 * File: src/qho_binary_io.c
 * Purpose: Binary input-output helpers for sampled paths and measured observables.
 * Fixed-size records store sweep-indexed path averages together with beta, eta,
 * Nt, and update metadata required for subsequent statistical analysis.
 */

/* -------------------------------------------------------------------------
 * Section: includes and implementation
 * ------------------------------------------------------------------------- */

#include "qho_binary_io.h"

#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/statvfs.h>

static int write_exact(FILE *out, const void *ptr, size_t size, const char *field)
{
    if (fwrite(ptr, size, 1, out) != 1) {
        fprintf(stderr, "error: failed to write binary field '%s'\n", field);
        return -1;
    }
    return 0;
}

unsigned long long qho_expected_records(const qho_params_t *params)
{
    if (params == NULL || params->meas_stride <= 0L || params->n_sweeps <= 0L) {
        return 0ULL;
    }
    return (unsigned long long)(params->n_sweeps / params->meas_stride);
}

int qho_write_staging_path(const char *path, char *staging_path, size_t staging_path_size)
{
    const int written = snprintf(staging_path, staging_path_size, "%s.staging", path);
    if (written < 0 || (size_t)written >= staging_path_size) {
        fprintf(stderr, "error: output path too long for staging path\n");
        return -1;
    }
    return 0;
}

int qho_preflight_output_space(const char *path, unsigned long long expected_bytes)
{
    char dir[QHO_PATH_MAX];
    char *slash = NULL;
    struct statvfs vfs;
    unsigned long long available;
    const unsigned long long required = expected_bytes + expected_bytes / 5ULL;

    if (path == NULL || path[0] == '\0') {
        return -1;
    }
    if (strlen(path) >= sizeof(dir)) {
        fprintf(stderr, "error: path too long for disk-space preflight\n");
        return -1;
    }
    strcpy(dir, path);
    slash = strrchr(dir, '/');
    if (slash != NULL) {
        if (slash == dir) {
            slash[1] = '\0';
        } else {
            *slash = '\0';
        }
    } else {
        strcpy(dir, ".");
    }
    if (statvfs(dir, &vfs) != 0) {
        fprintf(stderr, "error: statvfs failed for '%s': %s\n", dir, strerror(errno));
        return -1;
    }
    available = (unsigned long long)vfs.f_bavail * (unsigned long long)vfs.f_frsize;
    printf("Output preflight:\n");
    printf("  expected bytes = %llu\n", expected_bytes);
    printf("  required bytes = %llu\n", required);
    printf("  available bytes = %llu\n", available);
    if (required > available) {
        fprintf(stderr, "error: output preflight failed for '%s': need %llu bytes with margin, have %llu\n",
            path, required, available);
        return -1;
    }
    return 0;
}

int qho_binary_write_header(FILE *out, const qho_params_t *params)
{
    unsigned char header[QHO_BIN_HEADER_SIZE];
    size_t offset = 0U;
    const char magic[8] = QHO_BIN_MAGIC;
    const uint32_t version = QHO_BIN_VERSION;
    const uint32_t header_size = QHO_BIN_HEADER_SIZE;
    const uint32_t record_size = QHO_BIN_RECORD_SIZE;
    const uint32_t endian = QHO_BIN_ENDIAN_MARKER;
    const int32_t nt = (int32_t)params->nt;
    const int32_t update = (int32_t)params->update_mode;
    const int32_t init = (int32_t)params->init;
    const int32_t n_over = (int32_t)params->n_overrelax;
    const int64_t n_therm = (int64_t)params->n_therm;
    const int64_t n_sweeps = (int64_t)params->n_sweeps;
    const int64_t stride = (int64_t)params->meas_stride;
    const uint64_t seed = params->seed;
    const uint64_t stream = params->stream;
    const uint32_t obs_mask = QHO_BIN_OBS_DEFAULT;
    const uint64_t n_expected = (uint64_t)qho_expected_records(params);
    const double beta = params->beta;
    const double eta = params->eta;

    memset(header, 0, sizeof(header));
#define PUT_FIELD(value) do { memcpy(header + offset, &(value), sizeof(value)); offset += sizeof(value); } while (0)
    memcpy(header + offset, magic, sizeof(magic)); offset += sizeof(magic);
    PUT_FIELD(version);
    PUT_FIELD(header_size);
    PUT_FIELD(record_size);
    PUT_FIELD(endian);
    PUT_FIELD(nt);
    PUT_FIELD(beta);
    PUT_FIELD(eta);
    PUT_FIELD(n_therm);
    PUT_FIELD(n_sweeps);
    PUT_FIELD(stride);
    PUT_FIELD(seed);
    PUT_FIELD(stream);
    PUT_FIELD(update);
    PUT_FIELD(init);
    PUT_FIELD(n_over);
    PUT_FIELD(obs_mask);
    PUT_FIELD(n_expected);
#undef PUT_FIELD
    if (offset > sizeof(header)) {
        fprintf(stderr, "error: internal binary header overflow\n");
        return -1;
    }
    if (fwrite(header, sizeof(header), 1, out) != 1) {
        fprintf(stderr, "error: failed to write binary header\n");
        return -1;
    }
    return 0;
}

int qho_binary_write_record(FILE *out, int64_t sweep, const qho_measurements_t *m, double acc_rate)
{
    if (write_exact(out, &sweep, sizeof(sweep), "sweep") != 0) return -1;
    if (write_exact(out, &m->y_mean, sizeof(m->y_mean), "y_mean") != 0) return -1;
    if (write_exact(out, &m->y2_mean, sizeof(m->y2_mean), "y2_mean") != 0) return -1;
    if (write_exact(out, &m->dy2_mean, sizeof(m->dy2_mean), "dy2_mean") != 0) return -1;
    if (write_exact(out, &m->energy_ren, sizeof(m->energy_ren), "energy_ren") != 0) return -1;
    if (write_exact(out, &acc_rate, sizeof(acc_rate), "acc_rate") != 0) return -1;
    return 0;
}
