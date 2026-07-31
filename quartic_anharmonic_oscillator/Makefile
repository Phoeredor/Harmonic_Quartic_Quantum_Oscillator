# Build the Euclidean path-integral sampler for the quartic oscillator and the
# analysis targets used for beta=5 continuum observables and final figures.

CC = gcc
# C99 and strict warnings support a portable lattice implementation; -O3 is
# useful for the long Markov chains required by continuum extrapolations.
CFLAGS = -std=c99 -O3 -Wall -Wextra -pedantic -Iinclude
# The action, Metropolis probabilities, and Gaussian initialization use libm.
LDFLAGS = -lm

SRC_DIR = src
BIN_DIR = bin
TARGET = $(BIN_DIR)/anharmonic_pimc
# All C modules contribute to the sampler: lattice storage, updates, measurements,
# distributions, path output, correlators, parameters, and random numbers.
SRCS = $(wildcard $(SRC_DIR)/*.c)

.PHONY: all clean analyze plots

all: $(TARGET)

# Principal executable for sampling periodic paths with V(y)=y^2/2+lambda*y^4.
$(TARGET): $(SRCS) | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $(SRCS) $(LDFLAGS)

$(BIN_DIR):
	mkdir -p $@

# Convert beta=5 ensembles into blocked estimates and eta^2 continuum limits.
analyze:
	python3 scripts/analysis/analyze_continuum_points.py
	python3 scripts/analysis/fit_continuum_moments.py
	python3 scripts/analysis/assemble_continuum_summary.py
	python3 scripts/analysis/analyze_virial_estimators.py

# Render the continuum, virial, and position-distribution results.
plots:
	python3 scripts/plot/plot_thermodynamics.py
	python3 scripts/plot/plot_summary_observables.py

clean:
	rm -rf $(BIN_DIR)
