/*
 * Project: Quantum Harmonic Oscillator PIMC
 * File: src/main_qho.c
 * Purpose: Command-line driver for quantum harmonic oscillator PIMC simulations.
 * It samples periodic paths with weight exp(-S_E), measures thermodynamic and
 * correlation observables, and dispatches the requested statistical outputs.
 */

/* -------------------------------------------------------------------------
 * Section: includes and implementation
 * ------------------------------------------------------------------------- */

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "pcg32.h"
#include "qho_binary_io.h"
#include "qho_correlator.h"
#include "qho_histogram.h"
#include "qho_lattice.h"
#include "qho_logbin.h"
#include "qho_measure.h"
#include "qho_params.h"
#include "qho_path_output.h"
#include "qho_spectrum.h"
#include "qho_update.h"

static int ensure_parent_directories(const char *path)
{
    char buffer[QHO_PATH_MAX];
    char *cursor;
    const size_t length = strlen(path);

    if (length == 0U || strchr(path, '/') == NULL) {
        return 0;
    }
    if (length >= sizeof(buffer)) {
        fprintf(stderr, "[ERROR]: output path is too long: '%s'\n", path);
        return -1;
    }

    memcpy(buffer, path, length + 1U);
    for (cursor = buffer + 1; *cursor != '\0'; ++cursor) {
        if (*cursor != '/') {
            continue;
        }
        *cursor = '\0';
        if (mkdir(buffer, 0777) != 0 && errno != EEXIST) {
            fprintf(stderr, "[ERROR]: cannot create output directory '%s': %s\n", buffer, strerror(errno));
            return -1;
        }
        *cursor = '/';
    }
    return 0;
}

static int ensure_output_directories(const qho_params_t *params)
{
    if (params->output_format != QHO_OUTPUT_NONE && ensure_parent_directories(params->out_path) != 0) {
        return -1;
    }
    if (params->therm_logbin_enabled && ensure_parent_directories(params->therm_logbin_out) != 0) {
        return -1;
    }
    if (params->hist_enabled && ensure_parent_directories(params->hist_out) != 0) {
        return -1;
    }
    if (params->hist_block_enabled && ensure_parent_directories(params->hist_block_out) != 0) {
        return -1;
    }
    if (params->corr_enabled && ensure_parent_directories(params->corr_out) != 0) {
        return -1;
    }
    if (params->spectrum_enabled && ensure_parent_directories(params->spectrum_out) != 0) {
        return -1;
    }
    if (params->spectrum_block_enabled && ensure_parent_directories(params->spectrum_block_out) != 0) {
        return -1;
    }
    if (params->path_enabled && ensure_parent_directories(params->path_out) != 0) {
        return -1;
    }
    return 0;
}

static void write_output_header(FILE *out, const qho_params_t *params)
{
    fprintf(out, "# QHO_PIMC run\n");
    fprintf(out, "# action S = sum_i [ eta/2 y_i^2 + 1/(2 eta) (y_{i+1}-y_i)^2 ]\n");
    fprintf(out, "# nt %d\n", params->nt);
    fprintf(out, "# beta %.17g\n", params->beta);
    fprintf(out, "# eta %.17g\n", params->eta);
    fprintf(out, "# therm %ld\n", params->n_therm);
    fprintf(out, "# sweeps %ld\n", params->n_sweeps);
    fprintf(out, "# stride %ld\n", params->meas_stride);
    fprintf(out, "# seed %" PRIu64 "\n", params->seed);
    fprintf(out, "# stream %" PRIu64 "\n", params->stream);
    fprintf(out, "# delta %.17g\n", params->delta);
    fprintf(out, "# init %s\n", qho_init_name(params->init));
    fprintf(out, "# update %s\n", qho_update_mode_name(params->update_mode));
    fprintf(out, "# n_over %d\n", params->n_overrelax);
    if (params->hist_enabled) {
        fprintf(out, "# hist_out %s\n", params->hist_out);
        fprintf(out, "# hist_bins %d\n", params->hist_bins);
        fprintf(out, "# hist_min %.17g\n", params->hist_min);
        fprintf(out, "# hist_max %.17g\n", params->hist_max);
    }
    if (params->hist_block_enabled) {
        fprintf(out, "# hist_block_out %s\n", params->hist_block_out);
        fprintf(out, "# hist_block_size_saved %d\n", params->hist_block_size_saved);
    }
    if (params->corr_enabled) {
        fprintf(out, "# corr_out %s\n", params->corr_out);
        fprintf(out, "# corr_max_lag %d\n", params->corr_max_lag);
    }
    if (params->spectrum_enabled) {
        fprintf(out, "# spectrum_out %s\n", params->spectrum_out);
        fprintf(out, "# spectrum_max_lag %d\n", params->spectrum_max_lag);
    }
    if (params->spectrum_block_enabled) {
        fprintf(out, "# spectrum_block_out %s\n", params->spectrum_block_out);
        fprintf(out, "# spectrum_block_size_saved %d\n", params->spectrum_block_size_saved);
        fprintf(out, "# spectrum_max_lag %d\n", params->spectrum_max_lag);
    }
    if (params->path_enabled) {
        fprintf(out, "# path_out %s\n", params->path_out);
    }
    fprintf(out, "# sweep y_mean y2_mean dy2_mean potential kinetic_ren energy_ren acc_rate\n");
}

int main(int argc, char **argv)
{
    qho_params_t params = qho_params_default();
    const int parse_status = qho_params_parse_args(&params, argc, argv);
    pcg32_rng_t pcg;
    qho_lattice_t lattice;
    qho_histogram_t histogram;
    qho_histogram_block_accumulator_t histogram_blocks;
    qho_y_correlator_t correlator;
    qho_spectrum_correlator_t spectrum;
    qho_spectrum_block_accumulator_t spectrum_blocks;
    int histogram_ready = 0;
    int histogram_blocks_ready = 0;
    int correlator_ready = 0;
    int spectrum_ready = 0;
    int spectrum_blocks_ready = 0;
    FILE *out = NULL;
    FILE *logbin_out = NULL;
    qho_logbin_accumulator_t logbin_acc;
    char out_staging_path[QHO_PATH_MAX];
    char logbin_staging_path[QHO_PATH_MAX];
    long sweep;
    long saved_measurements = 0L;
    double production_acceptance_sum = 0.0;

    if (parse_status > 0) {
        qho_params_print_usage(stdout, argv[0]);
        return EXIT_SUCCESS;
    }

    if (parse_status < 0) {
        qho_params_print_usage(stderr, argv[0]);
        return EXIT_FAILURE;
    }

    if (ensure_output_directories(&params) != 0) {
        return EXIT_FAILURE;
    }

    pcg32_seed_sequence(&pcg, params.seed, params.stream);

    if (qho_lattice_init(&lattice, params.nt) != 0) {
        fprintf(stderr, "error: failed to allocate lattice with nt = %d\n", params.nt);
        return EXIT_FAILURE;
    }

    if (params.init == QHO_INIT_RANDOM || params.init == QHO_INIT_UNIFORM) {
        qho_lattice_init_random_uniform(&lattice, &pcg);
    } else {
        qho_lattice_fill(&lattice, 0.0);
    }

    printf("QHO_PIMC\n");
    printf("Path-integral simulation for the quantum harmonic oscillator\n\n");
    qho_params_print(&params, stdout);

    /* Evolve the initial path to the equilibrium distribution before sampling. */
    for (sweep = 0; sweep < params.n_therm; ++sweep) {
        (void)qho_update_sweep(&lattice, &params, &pcg);
    }

    /* Allocate only the observable accumulators selected for this ensemble. */
    if (params.hist_enabled) {
        if (qho_histogram_init(&histogram, params.hist_bins, params.hist_min, params.hist_max) != 0) {
            fprintf(stderr, "[ERROR]: failed to initialize histogram\n");
            qho_lattice_free(&lattice);
            return EXIT_FAILURE;
        }
        histogram_ready = 1;
    }

    if (params.hist_block_enabled) {
        if (qho_histogram_block_init(&histogram_blocks, params.nt, params.hist_bins,
                params.hist_min, params.hist_max, params.hist_block_size_saved) != 0) {
            fprintf(stderr, "[ERROR]: failed to initialize block histogram\n");
            if (histogram_ready) qho_histogram_free(&histogram);
            qho_lattice_free(&lattice);
            return EXIT_FAILURE;
        }
        histogram_blocks_ready = 1;
    }

    if (params.corr_enabled) {
        if (qho_y_correlator_init(&correlator, params.nt, params.corr_max_lag) != 0) {
            fprintf(stderr, "[ERROR]: failed to initialize y correlator\n");
            if (histogram_ready) {
                qho_histogram_free(&histogram);
            }
            qho_lattice_free(&lattice);
            return EXIT_FAILURE;
        }
        correlator_ready = 1;
    }

    if (params.spectrum_enabled) {
        if (qho_spectrum_correlator_init(&spectrum, params.nt, params.spectrum_max_lag) != 0) {
            fprintf(stderr, "[ERROR]: failed to initialize spectrum correlator\n");
            if (correlator_ready) {
                qho_y_correlator_free(&correlator);
            }
            if (histogram_ready) {
                qho_histogram_free(&histogram);
            }
            qho_lattice_free(&lattice);
            return EXIT_FAILURE;
        }
        spectrum_ready = 1;
    }

    if (params.spectrum_block_enabled) {
        if (qho_spectrum_block_accumulator_init(&spectrum_blocks, params.nt, params.spectrum_max_lag, params.spectrum_block_size_saved) != 0) {
            fprintf(stderr, "[ERROR]: failed to initialize spectrum block accumulator\n");
            if (spectrum_ready) {
                qho_spectrum_correlator_free(&spectrum);
            }
            if (correlator_ready) {
                qho_y_correlator_free(&correlator);
            }
            if (histogram_ready) {
                qho_histogram_free(&histogram);
            }
            qho_lattice_free(&lattice);
            return EXIT_FAILURE;
        }
        spectrum_blocks_ready = 1;
    }

    if (params.output_format != QHO_OUTPUT_NONE) {
        const unsigned long long expected_records = qho_expected_records(&params);
        const unsigned long long record_size = (params.output_format == QHO_OUTPUT_BIN) ? QHO_BIN_RECORD_SIZE : 160ULL;
        const unsigned long long header_size = (params.output_format == QHO_OUTPUT_BIN) ? QHO_BIN_HEADER_SIZE : 1024ULL;
        const unsigned long long expected_bytes = header_size + expected_records * record_size;
        if (qho_preflight_output_space(params.out_path, expected_bytes) != 0) {
            if (spectrum_ready) {
                qho_spectrum_correlator_free(&spectrum);
            }
            if (correlator_ready) {
                qho_y_correlator_free(&correlator);
            }
            if (histogram_ready) {
                qho_histogram_free(&histogram);
            }
            qho_lattice_free(&lattice);
            return EXIT_FAILURE;
        }
        if (qho_write_staging_path(params.out_path, out_staging_path, sizeof(out_staging_path)) != 0) {
            if (spectrum_ready) {
                qho_spectrum_correlator_free(&spectrum);
            }
            if (correlator_ready) {
                qho_y_correlator_free(&correlator);
            }
            if (histogram_ready) {
                qho_histogram_free(&histogram);
            }
            qho_lattice_free(&lattice);
            return EXIT_FAILURE;
        }
        out = fopen(out_staging_path, params.output_format == QHO_OUTPUT_BIN ? "wb" : "w");
        if (out == NULL) {
            fprintf(stderr, "[ERROR]: cannot open output file '%s': %s\n", out_staging_path, strerror(errno));
            if (spectrum_ready) {
                qho_spectrum_correlator_free(&spectrum);
            }
            if (correlator_ready) {
                qho_y_correlator_free(&correlator);
            }
            if (histogram_ready) {
                qho_histogram_free(&histogram);
            }
            qho_lattice_free(&lattice);
            return EXIT_FAILURE;
        }
        if (params.output_format == QHO_OUTPUT_BIN) {
            if (qho_binary_write_header(out, &params) != 0) {
                fclose(out);
                if (spectrum_ready) {
                    qho_spectrum_correlator_free(&spectrum);
                }
                if (correlator_ready) {
                    qho_y_correlator_free(&correlator);
                }
                if (histogram_ready) {
                    qho_histogram_free(&histogram);
                }
                qho_lattice_free(&lattice);
                return EXIT_FAILURE;
            }
        } else {
            write_output_header(out, &params);
        }
    }

    if (params.therm_logbin_enabled) {
        const unsigned long long expected_bytes = 65536ULL;
        if (qho_preflight_output_space(params.therm_logbin_out, expected_bytes) != 0) {
            if (out != NULL) {
                fclose(out);
            }
            if (spectrum_ready) {
                qho_spectrum_correlator_free(&spectrum);
            }
            if (correlator_ready) {
                qho_y_correlator_free(&correlator);
            }
            if (histogram_ready) {
                qho_histogram_free(&histogram);
            }
            qho_lattice_free(&lattice);
            return EXIT_FAILURE;
        }
        if (qho_write_staging_path(params.therm_logbin_out, logbin_staging_path, sizeof(logbin_staging_path)) != 0) {
            if (out != NULL) {
                fclose(out);
            }
            if (spectrum_ready) {
                qho_spectrum_correlator_free(&spectrum);
            }
            if (correlator_ready) {
                qho_y_correlator_free(&correlator);
            }
            if (histogram_ready) {
                qho_histogram_free(&histogram);
            }
            qho_lattice_free(&lattice);
            return EXIT_FAILURE;
        }
        logbin_out = fopen(logbin_staging_path, "w");
        if (logbin_out == NULL) {
            fprintf(stderr, "[ERROR]: cannot open log-bin output file '%s': %s\n", logbin_staging_path, strerror(errno));
            if (out != NULL) {
                fclose(out);
            }
            if (spectrum_ready) {
                qho_spectrum_correlator_free(&spectrum);
            }
            if (correlator_ready) {
                qho_y_correlator_free(&correlator);
            }
            if (histogram_ready) {
                qho_histogram_free(&histogram);
            }
            qho_lattice_free(&lattice);
            return EXIT_FAILURE;
        }
        if (qho_logbin_write_header(logbin_out, &params) != 0) {
            fprintf(stderr, "[ERROR]: failed to write log-bin header '%s'\n", params.therm_logbin_out);
            fclose(logbin_out);
            if (out != NULL) {
                fclose(out);
            }
            if (spectrum_ready) {
                qho_spectrum_correlator_free(&spectrum);
            }
            if (correlator_ready) {
                qho_y_correlator_free(&correlator);
            }
            if (histogram_ready) {
                qho_histogram_free(&histogram);
            }
            qho_lattice_free(&lattice);
            return EXIT_FAILURE;
        }
        qho_logbin_init(&logbin_acc, params.logbin_base);
    }

    if (out == NULL && params.output_format != QHO_OUTPUT_NONE) {
        fprintf(stderr, "[ERROR]: internal output setup failure\n");
        if (spectrum_ready) {
            qho_spectrum_correlator_free(&spectrum);
        }
        if (correlator_ready) {
            qho_y_correlator_free(&correlator);
        }
        if (histogram_ready) {
            qho_histogram_free(&histogram);
        }
        qho_lattice_free(&lattice);
        return EXIT_FAILURE;
    }

    /* Sample the equilibrium Markov chain and measure at the chosen stride. */
    for (sweep = 1; sweep <= params.n_sweeps; ++sweep) {
        const double acc_rate = qho_update_sweep(&lattice, &params, &pcg);
        production_acceptance_sum += acc_rate;

        if (logbin_out != NULL) {
            const qho_measurements_t m_log = qho_measure_basic(&lattice, &params);
            if (qho_logbin_accumulate(&logbin_acc, sweep, &m_log, params.logbin_base, logbin_out, qho_init_name(params.init)) != 0) {
                fprintf(stderr, "[ERROR]: failed while writing log-bin output '%s'\n", params.therm_logbin_out);
                fclose(logbin_out);
                if (out != NULL) {
                    fclose(out);
                }
                if (spectrum_ready) {
                    qho_spectrum_correlator_free(&spectrum);
                }
                if (correlator_ready) {
                    qho_y_correlator_free(&correlator);
                }
                if (histogram_ready) {
                    qho_histogram_free(&histogram);
                }
                qho_lattice_free(&lattice);
                return EXIT_FAILURE;
            }
        }

        if (sweep % params.meas_stride == 0
            && (out != NULL || histogram_ready || histogram_blocks_ready || correlator_ready
                || spectrum_ready || spectrum_blocks_ready)) {
            const qho_measurements_t m = qho_measure_basic(&lattice, &params);

            if (out != NULL && params.output_format == QHO_OUTPUT_BIN) {
                if (qho_binary_write_record(out, (int64_t)sweep, &m, acc_rate) != 0) {
                    fprintf(stderr, "[ERROR]: failed while writing binary output file '%s'\n", params.out_path);
                    fclose(out);
                    if (logbin_out != NULL) {
                        fclose(logbin_out);
                    }
                    if (spectrum_ready) {
                        qho_spectrum_correlator_free(&spectrum);
                    }
                    if (correlator_ready) {
                        qho_y_correlator_free(&correlator);
                    }
                    if (histogram_ready) {
                        qho_histogram_free(&histogram);
                    }
                    qho_lattice_free(&lattice);
                    return EXIT_FAILURE;
                }
            } else if (out != NULL && fprintf(out, "%ld %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n",
                    sweep,
                    m.y_mean,
                    m.y2_mean,
                    m.dy2_mean,
                    m.potential,
                    m.kinetic_ren,
                    m.energy_ren,
                    acc_rate) < 0) {
                fprintf(stderr, "[ERROR]: failed while writing output file '%s'\n", params.out_path);
                fclose(out);
                if (logbin_out != NULL) {
                    fclose(logbin_out);
                }
                if (spectrum_ready) {
                    qho_spectrum_correlator_free(&spectrum);
                }
                if (correlator_ready) {
                    qho_y_correlator_free(&correlator);
                }
                if (histogram_ready) {
                    qho_histogram_free(&histogram);
                }
                qho_lattice_free(&lattice);
                return EXIT_FAILURE;
            }

            if (histogram_ready) {
                qho_histogram_accumulate_lattice(&histogram, &lattice);
            }
            if (histogram_blocks_ready) {
                if (qho_histogram_block_accumulate(&histogram_blocks, &lattice) != 0) {
                    fprintf(stderr, "[ERROR]: failed to accumulate block histogram data\n");
                    qho_histogram_block_free(&histogram_blocks);
                    if (histogram_ready) qho_histogram_free(&histogram);
                    qho_lattice_free(&lattice);
                    return EXIT_FAILURE;
                }
            }
            if (correlator_ready) {
                qho_y_correlator_accumulate(&correlator, &lattice);
            }
            if (spectrum_ready) {
                qho_spectrum_correlator_accumulate(&spectrum, &lattice);
            }
            if (spectrum_blocks_ready) {
                if (qho_spectrum_block_accumulator_accumulate(&spectrum_blocks, &lattice) != 0) {
                    fprintf(stderr, "[ERROR]: failed to accumulate spectrum block data\n");
                    if (out != NULL) {
                        fclose(out);
                    }
                    if (logbin_out != NULL) {
                        fclose(logbin_out);
                    }
                    qho_spectrum_block_accumulator_free(&spectrum_blocks);
                    if (spectrum_ready) {
                        qho_spectrum_correlator_free(&spectrum);
                    }
                    if (correlator_ready) {
                        qho_y_correlator_free(&correlator);
                    }
                    if (histogram_ready) {
                        qho_histogram_free(&histogram);
                    }
                    qho_lattice_free(&lattice);
                    return EXIT_FAILURE;
                }
            }

            ++saved_measurements;
        }
    }

    if (logbin_out != NULL) {
        if (qho_logbin_flush(&logbin_acc, params.logbin_base, logbin_out, qho_init_name(params.init)) != 0) {
            fprintf(stderr, "[ERROR]: failed to flush log-bin output '%s'\n", params.therm_logbin_out);
            fclose(logbin_out);
            if (out != NULL) {
                fclose(out);
            }
            if (spectrum_ready) {
                qho_spectrum_correlator_free(&spectrum);
            }
            if (correlator_ready) {
                qho_y_correlator_free(&correlator);
            }
            if (histogram_ready) {
                qho_histogram_free(&histogram);
            }
            qho_lattice_free(&lattice);
            return EXIT_FAILURE;
        }
        if (fclose(logbin_out) != 0) {
            fprintf(stderr, "[ERROR]: failed to close log-bin output file '%s': %s\n", params.therm_logbin_out, strerror(errno));
            if (out != NULL) {
                fclose(out);
            }
            if (spectrum_ready) {
                qho_spectrum_correlator_free(&spectrum);
            }
            if (correlator_ready) {
                qho_y_correlator_free(&correlator);
            }
            if (histogram_ready) {
                qho_histogram_free(&histogram);
            }
            qho_lattice_free(&lattice);
            return EXIT_FAILURE;
        }
        if (rename(logbin_staging_path, params.therm_logbin_out) != 0) {
            fprintf(stderr, "[ERROR]: failed to rename '%s' to '%s': %s\n", logbin_staging_path, params.therm_logbin_out, strerror(errno));
            if (out != NULL) {
                fclose(out);
            }
            if (spectrum_ready) {
                qho_spectrum_correlator_free(&spectrum);
            }
            if (correlator_ready) {
                qho_y_correlator_free(&correlator);
            }
            if (histogram_ready) {
                qho_histogram_free(&histogram);
            }
            qho_lattice_free(&lattice);
            return EXIT_FAILURE;
        }
    }

    if (out != NULL && fclose(out) != 0) {
        fprintf(stderr, "[ERROR]: failed to close output file '%s': %s\n", params.out_path, strerror(errno));
        if (spectrum_ready) {
            qho_spectrum_correlator_free(&spectrum);
        }
        if (correlator_ready) {
            qho_y_correlator_free(&correlator);
        }
        if (histogram_ready) {
            qho_histogram_free(&histogram);
        }
        qho_lattice_free(&lattice);
        return EXIT_FAILURE;
    }
    if (out != NULL && rename(out_staging_path, params.out_path) != 0) {
        fprintf(stderr, "[ERROR]: failed to rename '%s' to '%s': %s\n", out_staging_path, params.out_path, strerror(errno));
        if (spectrum_ready) {
            qho_spectrum_correlator_free(&spectrum);
        }
        if (correlator_ready) {
            qho_y_correlator_free(&correlator);
        }
        if (histogram_ready) {
            qho_histogram_free(&histogram);
        }
        qho_lattice_free(&lattice);
        return EXIT_FAILURE;
    }

    /* Normalize and write ensemble-level observables after sampling is complete. */
    if (histogram_ready) {
        if (qho_histogram_write_density(&histogram, params.hist_out, &params) != 0) {
            fprintf(stderr, "[ERROR]: failed to write histogram file '%s'\n", params.hist_out);
            if (spectrum_ready) {
                qho_spectrum_correlator_free(&spectrum);
            }
            if (correlator_ready) {
                qho_y_correlator_free(&correlator);
            }
            qho_histogram_free(&histogram);
            qho_lattice_free(&lattice);
            return EXIT_FAILURE;
        }
    }

    if (histogram_blocks_ready) {
        if (qho_histogram_block_write(&histogram_blocks, params.hist_block_out, &params) != 0) {
            fprintf(stderr, "[ERROR]: failed to write block histogram file '%s'\n", params.hist_block_out);
            qho_histogram_block_free(&histogram_blocks);
            if (histogram_ready) qho_histogram_free(&histogram);
            qho_lattice_free(&lattice);
            return EXIT_FAILURE;
        }
    }

    if (correlator_ready) {
        if (qho_y_correlator_write(&correlator, params.corr_out, &params) != 0) {
            fprintf(stderr, "[ERROR]: failed to write y correlator file '%s'\n", params.corr_out);
            if (spectrum_ready) {
                qho_spectrum_correlator_free(&spectrum);
            }
            qho_y_correlator_free(&correlator);
            if (histogram_ready) {
                qho_histogram_free(&histogram);
            }
            qho_lattice_free(&lattice);
            return EXIT_FAILURE;
        }
    }

    if (spectrum_ready) {
        if (qho_spectrum_correlator_write(&spectrum, params.spectrum_out, &params) != 0) {
            fprintf(stderr, "[ERROR]: failed to write spectrum correlator file '%s'\n", params.spectrum_out);
            if (spectrum_blocks_ready) {
                qho_spectrum_block_accumulator_free(&spectrum_blocks);
            }
            qho_spectrum_correlator_free(&spectrum);
            if (correlator_ready) {
                qho_y_correlator_free(&correlator);
            }
            if (histogram_ready) {
                qho_histogram_free(&histogram);
            }
            qho_lattice_free(&lattice);
            return EXIT_FAILURE;
        }
    }

    if (spectrum_blocks_ready) {
        if (qho_spectrum_block_accumulator_write(&spectrum_blocks, params.spectrum_block_out, &params) != 0) {
            fprintf(stderr, "[ERROR]: failed to write spectrum block file '%s'\n", params.spectrum_block_out);
            qho_spectrum_block_accumulator_free(&spectrum_blocks);
            if (spectrum_ready) {
                qho_spectrum_correlator_free(&spectrum);
            }
            if (correlator_ready) {
                qho_y_correlator_free(&correlator);
            }
            if (histogram_ready) {
                qho_histogram_free(&histogram);
            }
            qho_lattice_free(&lattice);
            return EXIT_FAILURE;
        }
    }

    if (params.path_enabled) {
        if (qho_write_path_snapshot(&lattice, &params, params.path_out) != 0) {
            fprintf(stderr, "[ERROR]: failed to write path snapshot file '%s'\n", params.path_out);
            if (spectrum_ready) {
                qho_spectrum_correlator_free(&spectrum);
            }
            if (correlator_ready) {
                qho_y_correlator_free(&correlator);
            }
            if (histogram_ready) {
                qho_histogram_free(&histogram);
            }
            qho_lattice_free(&lattice);
            return EXIT_FAILURE;
        }
    }

    printf("\nProduction summary:\n");
    printf("  average acc_rate column = %.6f\n", production_acceptance_sum / (double)params.n_sweeps);
    printf("  output path             = %s\n", params.out_path);
    printf("  saved measurements      = %ld\n", saved_measurements);
    if (histogram_ready) {
        printf("  histogram path          = %s\n", params.hist_out);
        printf("  histogram samples       = %llu\n", histogram.total_samples);
        qho_histogram_free(&histogram);
    }
    if (histogram_blocks_ready) {
        printf("  block histogram path    = %s\n", params.hist_block_out);
        printf("  histogram blocks        = %llu\n", histogram_blocks.n_blocks);
        qho_histogram_block_free(&histogram_blocks);
    }
    if (correlator_ready) {
        printf("  y correlator path       = %s\n", params.corr_out);
        printf("  y correlator measurements = %llu\n", correlator.n_measurements);
        qho_y_correlator_free(&correlator);
    }
    if (spectrum_ready) {
        printf("  spectrum path           = %s\n", params.spectrum_out);
        printf("  spectrum measurements   = %llu\n", spectrum.n_measurements);
        qho_spectrum_correlator_free(&spectrum);
    }
    if (spectrum_blocks_ready) {
        printf("  spectrum block path     = %s\n", params.spectrum_block_out);
        printf("  spectrum blocks         = %llu\n", spectrum_blocks.n_blocks);
        qho_spectrum_block_accumulator_free(&spectrum_blocks);
    }
    if (params.path_enabled) {
        printf("  path snapshot path      = %s\n", params.path_out);
    }

    qho_lattice_free(&lattice);
    return EXIT_SUCCESS;
}
