# -----------------------------------------------------------------------------
# Project: Quantum Harmonic Oscillator PIMC
# File: Makefile
# Purpose: Build rules for the quantum harmonic oscillator PIMC executable and consistency checks.
# The main executable samples periodic Euclidean paths; auxiliary targets build
# controlled checks and reproducible seed streams for independent ensembles.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Section: build configuration
# -----------------------------------------------------------------------------

CC = gcc
# C99 and strict warnings keep the lattice/estimator implementation portable;
# -O3 accelerates long Markov chains, and -Iinclude exposes the public interfaces.
CFLAGS = -std=c99 -O3 -Wall -Wextra -pedantic -Iinclude
# Generate header dependencies alongside objects so interface changes rebuild users.
DEPFLAGS = -MMD -MP
# The action, Gaussian updates, and estimators require the system math library.
LDFLAGS = -lm

SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin
TOOLS_DIR = tools

TARGET = $(BIN_DIR)/qho_pimc
SEED_TARGET = $(BIN_DIR)/seed_generator

SRCS = $(wildcard $(SRC_DIR)/*.c)
OBJS = $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(SRCS))
DEPS = $(OBJS:.o=.d) $(OBJ_DIR)/seed_generator.d

.PHONY: all clean sampling-efficiency equilibration-analysis thermo-beta5 path-snapshots plot-thermodynamics position-distribution-plots seed-generator

all: $(TARGET)

# Principal PIMC executable for thermodynamic, histogram, and spectrum ensembles.
$(TARGET): $(OBJS) | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LDFLAGS)

# Deterministic seed/stream pairs for statistically independent replicas.
$(SEED_TARGET): $(TOOLS_DIR)/seed_generator.c | $(BIN_DIR) $(OBJ_DIR)
	$(CC) $(CFLAGS) $(DEPFLAGS) -MF $(OBJ_DIR)/seed_generator.d -MT $@ -o $@ $(TOOLS_DIR)/seed_generator.c

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) $(DEPFLAGS) -c $< -o $@

$(OBJ_DIR) $(BIN_DIR):
	mkdir -p $@

sampling-efficiency:
	./scripts/run/run_sampling_efficiency.sh

equilibration-analysis:
	./scripts/run/run_equilibration_analysis.sh

thermo-beta5:
	./scripts/run/run_thermo_beta5_production.sh

path-snapshots:
	./scripts/run/run_path_snapshots.sh

plot-thermodynamics:
	./scripts/run/plot_thermodynamics.sh

position-distribution-plots:
	python3 scripts/plotting/plot_position_distribution.py
	python3 scripts/plotting/plot_position_distribution.py --variance

seed-generator: $(SEED_TARGET)

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

-include $(DEPS)
