# Path-Integral Monte Carlo for Harmonic and Quartic Oscillators

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Linux](https://img.shields.io/badge/Linux-FCC624.svg?style=flat&logo=linux&logoColor=black)
![C](https://img.shields.io/badge/C-00599C.svg?style=flat&logo=c&logoColor=white)
![Python](https://img.shields.io/badge/Python-%E2%89%A5%203.8-3776AB.svg?style=flat&logo=python&logoColor=white)
![GCC](https://img.shields.io/badge/GCC-00599C.svg?style=flat&logo=gnu&logoColor=white)

> **Module 3** — *Numerical Methods for Physics (Metodi Numerici per la Fisica)*, University of Pisa

This repository studies the one-dimensional quantum harmonic oscillator and
the quartic anharmonic oscillator through Euclidean path-integral Monte Carlo. Periodic
paths discretize the thermal trace, while continuum extrapolations, thermal
estimators, probability densities, and connected correlators connect the
lattice ensembles with analytical or exact-diagonalization results.

## 📄 Laboratory Report

Complete laboratory report for the *Numerical Methods for Physics - Module 3*:

**[Path-Integral Monte Carlo for Harmonic and Quartic Oscillators — Laboratory Report](Report_Path_Integral_Monte_Carlo_for_Harmonic_and_Quartic_Oscillators.pdf)**<br>
University of Pisa, A.A. 2025/2026.

---

## ✨ Key Features

- **Euclidean Path Integral:** Represent the thermal trace with periodic paths
  on a discretized imaginary-time lattice.
- **Continuum Limit:** Extrapolate observables as the lattice spacing
  $\eta\to0$.
- **Thermodynamic Estimators:** Compare bare, renormalized, and virial energy
  estimators with analytical expectations.
- **Monte Carlo Diagnostics:** Quantify autocorrelation, blocking, and
  equilibration effects.
- **Spectral Analysis:** Extract excitation gaps from connected Euclidean
  correlators.
- **Quartic Interaction:** Follow the deformation of observables and spectra
  with the coupling $\lambda$.

---

## 📁 Directory Structure

```text
Harmonic_Quartic_Quantum_Oscillator/
├── README.md
├── LICENSE
├── Report_Path_Integral_Monte_Carlo_for_Harmonic_and_Quartic_Oscillators.pdf
├── harmonic_oscillator/
│   ├── src/                 Harmonic PIMC implementation
│   ├── include/             C interfaces
│   ├── scripts/             Simulation, analysis, and plotting workflows
│   ├── data/                Git-ignored local runtime data
│   ├── plots/               20 harmonic report PNG figures
│   ├── docs/                Reproducibility notes
│   └── tools/               Seed utility source
└── quartic_anharmonic_oscillator/
    ├── src/                     Quartic PIMC implementation
    ├── include/                 C interfaces
    ├── scripts/                 Simulation, analysis, and plotting workflows
    ├── data/                    Git-ignored local runtime data
    └── plots/                   7 quartic report PNG figures
```

Detailed physical explanations and module-specific commands are given in the
[harmonic README](harmonic_oscillator/README.md) and the
[quartic README](quartic_anharmonic_oscillator/README.md).

---

## 🚀 Getting Started

The C samplers require GCC or another C99 compiler, Make, and the system
mathematics library. The numerical analyses and figures require Python 3.8 or
newer with NumPy, SciPy, and Matplotlib.

Build the modules independently from the repository root:

```bash
make -C harmonic_oscillator all
make -C quartic_anharmonic_oscillator all
```

Representative public targets are:

```bash
# Harmonic thermodynamics and figures
make -C harmonic_oscillator thermo-beta5
make -C harmonic_oscillator position-distribution-plots
make -C harmonic_oscillator plot-thermodynamics

# Quartic continuum analysis and figures
make -C quartic_anharmonic_oscillator analyze
make -C quartic_anharmonic_oscillator plots
```

These analysis and plotting commands require the corresponding simulation data
to have been generated locally first.

## 🎯 Physics Objectives

Imaginary time is divided into $N_t$ slices with

$$\eta=\frac{\beta}{N_t}.$$

For the harmonic oscillator, the dimensionless lattice action is

$$S_{\mathrm{harm}}[y] = \sum_{n=0}^{N_t-1}\left[\frac{(y_{n+1}-y_n)^2}{2\eta}+\frac{\eta}{2}y_n^2\right],\qquad y_{N_t}=y_0.$$

The quartic interaction changes the potential to

$$V(y)=\frac{1}{2}y^2+\lambda y^4.$$

The corresponding lattice action is

$$S_{\mathrm{quartic}}[y] = \sum_{n=0}^{N_t-1}\left[\frac{(y_{n+1}-y_n)^2}{2\eta}+\eta\left(\frac{1}{2}y_n^2+\lambda y_n^4\right)\right],\qquad y_{N_t}=y_0.$$

* **Continuum Limit:** Extrapolate thermal observables as $\eta\to0$.
* **Thermodynamics:** Study $\langle y^2\rangle$, $\langle y^4\rangle$, and
  energy estimators.
* **Thermal Crossover:** Reconstruct the position density $P(y)$.
* **Euclidean Spectrum:** Extract $\Delta_1$, $\Delta_2$, and $\Delta_3$ from
  connected correlators.
* **Quartic Deformation:** Study the dependence on $\lambda$ and the ratio
  $\Delta_2/(2\Delta_1)$.

---

## 📊 Results

The repository distributes exactly 27 report figures: 20 harmonic-oscillator
PNGs and 7 quartic-oscillator PNGs. The selection below illustrates the main
physical themes; the module READMEs provide access to the complete sets.

<details>
<summary><b>Harmonic representative figures</b></summary>
<br>

| Representative periodic Euclidean paths |
|:---:|
| <img src="harmonic_oscillator/plots/euclidean_path/fig_path_snapshots_beta5.png" width="950" alt="Representative periodic Euclidean paths"> |

| (a) Position variance continuum limit | (b) Renormalized energy continuum limit |
|:---:|:---:|
| <img src="harmonic_oscillator/plots/thermodynamics/fig_position_variance_continuum_beta5_eta2.png" width="460" alt="Harmonic position-variance continuum extrapolation"> | <img src="harmonic_oscillator/plots/thermodynamics/fig_renormalized_energy_continuum_beta5_eta2.png" width="460" alt="Harmonic renormalized-energy continuum extrapolation"> |

| Continuum summary of harmonic excitation gaps |
|:---:|
| <img src="harmonic_oscillator/plots/spectrum/fig_gap_continuum_summary_beta40_eta2.png" width="950" alt="Harmonic excitation-gap continuum summary"> |

| (a) Position distributions | (b) Variance crossover |
|:---:|:---:|
| <img src="harmonic_oscillator/plots/distribution/fig_position_distribution_beta_scan.png" width="460" alt="Harmonic position-distribution thermal scan"> | <img src="harmonic_oscillator/plots/distribution/fig_position_variance_crossover.png" width="460" alt="Harmonic position-variance thermal crossover"> |

</details>

<details>
<summary><b>Quartic representative figures</b></summary>
<br>

| (a) Continuum $\langle y^2\rangle$ | (b) Continuum $\langle y^4\rangle$ |
|:---:|:---:|
| <img src="quartic_anharmonic_oscillator/plots/thermodynamic/fig_y2_continuum_vs_lambda.png" width="460" alt="Quartic second moment continuum limit"> | <img src="quartic_anharmonic_oscillator/plots/thermodynamic/fig_y4_continuum_vs_lambda.png" width="460" alt="Quartic fourth moment continuum limit"> |

| Quartic virial components |
|:---:|
| <img src="quartic_anharmonic_oscillator/plots/thermodynamic/fig_virial_components_vs_lambda.png" width="950" alt="Quartic virial components versus coupling"> |

| (a) Position density | (b) Excitation gaps | (c) Gap ratio |
|:---:|:---:|:---:|
| <img src="quartic_anharmonic_oscillator/plots/distribution/fig_position_density_lambda_scan.png" width="310" alt="Quartic position density across the coupling scan"> | <img src="quartic_anharmonic_oscillator/plots/spectrum/fig_excitation_gaps_vs_lambda.png" width="310" alt="Quartic excitation gaps versus coupling"> | <img src="quartic_anharmonic_oscillator/plots/spectrum/fig_gap_ratio_vs_lambda.png" width="310" alt="Quartic gap ratio versus coupling"> |

</details>

---

## 🔧 Build System

The module Makefiles keep the two physical problems independent. The harmonic
Makefile provides `all`, `sampling-efficiency`, `equilibration-analysis`,
`thermo-beta5`, `path-snapshots`, `plot-thermodynamics`,
`position-distribution-plots`, `seed-generator`, and `clean`. The quartic
Makefile provides `all`, `analyze`, `plots`, and `clean`.

Compiled files are written below the corresponding module-level `bin/` and
`obj/` directories and can be removed with:

```bash
make -C harmonic_oscillator clean
make -C quartic_anharmonic_oscillator clean
```

---

## ♻️ Data and Reproducibility

Raw and processed simulation outputs are intentionally not distributed. Each
module writes runtime data below its local `data/` directory, where generated
content remains ignored by Git. The 27 PNG figures used in the laboratory
report are included below the two `plots/` trees.

Reproducing the numerical results requires rerunning the simulations and
analyses locally. Plotting commands must be used only after their required
runtime tables and ensembles have been generated.

## 📚 Bibliography

### Books

- W. Krauth, *Statistical Mechanics: Algorithms and Computations*, Oxford
  University Press (2006).
- D. P. Landau and K. Binder, *A Guide to Monte Carlo Simulations in
  Statistical Physics*, 5th ed., Cambridge University Press, 2021.
- L. Barone, E. Marinari, G. Organtini and F. Ricci-Tersenghi, *Scientific
  Programming: C-Language, Algorithms and Models in Science*, World Scientific,
  2013.

### Articles

- M. Creutz and B. Freedman, “A statistical approach to quantum mechanics,”
  *Annals of Physics* **132**, 427–462, 1981.
- D. M. Ceperley, *Reviews of Modern Physics* **67**, 279–355 (1995).
- C. Whitmer, “Over-relaxation methods for Monte Carlo simulations of quadratic
  and multiquadratic actions,” *Physical Review D* **29**, 306–314, 1984.
- N. Madras and A. D. Sokal, “The pivot algorithm: A highly efficient Monte
  Carlo method for the self-avoiding walk,” *Journal of Statistical Physics*
  **50**, 109–186, 1988.
- C. Bonati and M. D’Elia, *Physical Review E* **98**, 013308 (2018).

### Lecture Notes and Software

- C. Bonati, *Some notes for Metodi Numerici per la Fisica / Computational
  Physics Laboratory*, lecture notes, Università di Pisa, version of 29 June
  2026.
- M. D'Elia, *Appunti del Corso di Metodi Numerici della Fisica Teorica / Parte
  III - Applicazioni al calcolo del path-integral in meccanica quantistica*,
  lecture notes, Università di Pisa, 2016.
- **Bonati, Claudio** *Numerical Methods*.
  GitHub repository: [https://github.com/claudio-bonati/NumericalMethods](https://github.com/claudio-bonati/NumericalMethods)

## 📝 License

This project is released under the [MIT License](LICENSE).
