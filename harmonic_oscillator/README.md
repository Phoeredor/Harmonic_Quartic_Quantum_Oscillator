# Quantum Harmonic Oscillator: Path-Integral Monte Carlo

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](../LICENSE)
![Linux](https://img.shields.io/badge/Linux-FCC624.svg?style=flat&logo=linux&logoColor=black)
![C](https://img.shields.io/badge/C-00599C.svg?style=flat&logo=c&logoColor=white)
![Python](https://img.shields.io/badge/Python-%E2%89%A5%203.8-3776AB.svg?style=flat&logo=python&logoColor=white)
![GCC](https://img.shields.io/badge/GCC-00599C.svg?style=flat&logo=gnu&logoColor=white)

> **Module 3** — *Numerical Methods for Physics (Metodi Numerici per la Fisica)*, University of Pisa

This module studies the one-dimensional quantum harmonic oscillator through
its Euclidean path integral. The exact theory provides a controlled setting in
which to examine imaginary-time discretization, thermal estimators, continuum
extrapolation, Monte Carlo correlations, and the recovery of the excitation
spectrum from connected Euclidean correlators.

## 📄 Laboratory Report

Complete laboratory report for the *Numerical Methods for Physics - Module 3*:

**[Path-Integral Monte Carlo for Harmonic and Quartic Oscillators — Laboratory Report](../Report_Path_Integral_Monte_Carlo_for_Harmonic_and_Quartic_Oscillators.pdf)**<br>
University of Pisa, A.A. 2025/2026.

---

## ✨ Key Features

- **Euclidean Path Integral:** Represent the harmonic thermal trace with
  periodic imaginary-time paths.
- **Continuum Limit:** Extrapolate observables at fixed $\beta$ in $\eta^2$.
- **Thermodynamic Estimators:** Compare $\langle y^2\rangle$, bare and
  renormalized energies, and the virial relation with exact results.
- **Monte Carlo Diagnostics:** Test autocorrelation, blocking, equilibration,
  and initialization dependence.
- **Thermal Crossover:** Reconstruct the position density $P(y)$ from
  block-resolved histograms.
- **Spectral Analysis:** Extract $\Delta_1$, $\Delta_2$, and $\Delta_3$ from
  connected correlators.

---

## 📁 Directory Structure

```text
harmonic_oscillator/
├── Makefile
├── src/                 Harmonic PIMC implementation
├── include/             C interfaces
├── scripts/
│   ├── run/             Simulation workflows
│   ├── analysis/        Blocking and statistical analyses
│   └── plotting/        Report-figure generation
├── data/                Git-ignored local runtime outputs
├── plots/
│   ├── diagnostics/     Monte Carlo checks
│   ├── distribution/    Position-density results
│   ├── euclidean_path/  Representative periodic paths
│   ├── spectrum/        Correlators and excitation gaps
│   └── thermodynamics/ Thermodynamic and continuum results
├── docs/                Reproducibility notes
└── tools/               Seed utility source
```

---

## 🚀 Getting Started

### Requirements

The sampler requires GCC or another C99 compiler, Make, and the system
mathematics library. Analysis and plotting require Python 3.8 or newer with
NumPy, SciPy, and Matplotlib.

### Build

```bash
make clean
make all
```

The resulting executable is `bin/qho_pimc`.

### Minimal harmonic ensemble

```bash
./bin/qho_pimc \
  --nt 64 \
  --beta 4.0 \
  --therm 1000 \
  --sweeps 10000 \
  --stride 10 \
  --delta 1.0 \
  --seed 12345 \
  --stream 54 \
  --init zero \
  --update metro \
  --out data/raw/qho_beta4_nt64.dat
```

The output path above is generated locally; it is not a distributed dataset.

## 🎯 Physics Objectives

In units with $\hbar\omega=1$, the Hamiltonian is

$$H=\frac{p^2}{2}+\frac{y^2}{2}.$$

The thermal partition function is

$$Z(\beta)=\mathrm{Tr}\,e^{-\beta H}.$$

With $N_t$ imaginary-time slices and $\eta=\beta/N_t$, the periodic lattice
action is

$$S_{\mathrm{harm}}[y] = \sum_{n=0}^{N_t-1}\left[\frac{(y_{n+1}-y_n)^2}{2\eta}+\frac{\eta}{2}y_n^2\right],\qquad y_{N_t}=y_0.$$

* **Continuum Limit:** Determine thermal observables as $\eta\to0$ at fixed
  $\beta=5$.
* **Thermodynamics:** Test $\langle y^2\rangle$, the renormalized energy, and
  the virial relation against exact finite-temperature values.
* **Thermal Dependence:** Resolve the bare-energy variation at fixed $\eta$
  after subtracting a low-temperature reference.
* **Position Distribution:** Reconstruct $P(y)$ across the thermal crossover.
* **Euclidean Spectrum:** Use $y$, $y^2$, and
  $A(y)=y^3-\frac{3}{2}y$ to isolate the first three excitation channels with
  finite-$\beta$ cosh fits.

---

## 📊 Results

At $\beta=5$, the exact thermal variance is

$$\langle y^2\rangle_{\mathrm{exact}} = \frac{1}{2}\coth\left(\frac{5}{2}\right) = 0.50678365490630428.$$

The retained continuum analysis gives:

| Observable | Fit window | Continuum estimate | Exact value |
|:---|---:|---:|---:|
| $\langle y^2\rangle$ | $N_t=25\ldots200$ | $0.50596(80)$ | $0.50678365491$ |
| $H_{\mathrm{ren}}$ | $N_t=25\ldots80$ | $0.5053(22)$ | $0.50678365491$ |

The virial comparison uses

$$V=\frac{1}{2}\langle y^2\rangle,\qquad K_{\mathrm{ren}} = -\frac{\langle(\Delta y)^2\rangle}{2\eta^2}+\frac{1}{2\eta}.$$

For $\eta\le0.2$, 14 of 16 points are compatible within $2\sigma$ with
$\frac{1}{4}\coth(5/2)=0.253391827453$. The corresponding bare kinetic term,

$$K_{\mathrm{div}} = -\frac{\langle(\Delta y)^2\rangle}{2\eta^2},$$

shows the expected negative divergence as $\eta\to0$.

At fixed $\eta=0.05$, the temperature scan fits

$$\Delta U_b(\beta) = A\left[n_B(\beta)-n_B(\beta_0)\right],\qquad \beta_0=20,$$

with $A=0.990(12)$ and $\chi^2/\mathrm{dof}=0.735$.

The spectrum is consistent with $\Delta E_n=n$. The retained high-statistics
continuum estimates are:

| Channel | Exact gap | Continuum estimate |
|:---|---:|---:|
| $y$ | $1$ | $0.99973(26)$ |
| $y^2$ | $2$ | $1.99996(62)$ |
| $A(y)=y^3-\frac{3}{2}y$ | $3$ | Compatible with $3$ within the retained uncertainty |

### Complete harmonic figure set

All 20 harmonic report figures are accessible below.

<details>
<summary><b>Monte Carlo diagnostics</b></summary>
<br>

| (a) Sampling Algorithm efficiency | (b) Autocorrelation |
|:---:|:---:|
| <img src="plots/diagnostics/fig_sampling_efficiency_ymean_beta5_nt400.png" width="460" alt="Sampling efficiency for the mean position"> | <img src="plots/diagnostics/fig_autocorrelation_primary_beta5_nt128.png" width="460" alt="Autocorrelation of primary observables"> |

| Thermalization in logarithmic bins |
|:---:|
| <img src="plots/diagnostics/fig_thermalization_logbins_beta5_nt512.png" width="950" alt="Thermalization in logarithmic bins"> |

</details>

<details>
<summary><b>Thermodynamics and continuum extrapolation</b></summary>
<br>

| Representative periodic Euclidean paths |
|:---:|
| <img src="plots/euclidean_path/fig_path_snapshots_beta5.png" width="950" alt="Representative periodic Euclidean paths"> |

| Normalized deviation of $\langle y^2\rangle$ |
|:---:|
| <img src="plots/thermodynamics/fig_position_variance_pull_beta5_eta.png" width="950" alt="Normalized deviation of the position variance"> |

| (a) Position variance continuum limit | (b) Renormalized energy continuum limit |
|:---:|:---:|
| <img src="plots/thermodynamics/fig_position_variance_continuum_beta5_eta2.png" width="460" alt="Position-variance continuum extrapolation"> | <img src="plots/thermodynamics/fig_renormalized_energy_continuum_beta5_eta2.png" width="460" alt="Renormalized-energy continuum extrapolation"> |

| Continuum fit-window stability |
|:---:|
| <img src="plots/thermodynamics/fig_continuum_fit_window_stability_beta5.png" width="950" alt="Continuum fit-window stability"> |

| (a) Divergent kinetic term | (b) Virial relation |
|:---:|:---:|
| <img src="plots/thermodynamics/fig_divergent_kinetic_term_beta5_eta.png" width="460" alt="Bare divergent kinetic contribution"> | <img src="plots/thermodynamics/fig_virial_estimator_check_beta5_eta.png" width="460" alt="Virial-estimator comparison"> |

| (a) Bare energy versus temperature | (b) Subtracted thermal energy |
|:---:|:---:|
| <img src="plots/thermodynamics/fig_bare_energy_vs_invbeta_eta0p05.png" width="460" alt="Bare energy versus inverse temperature"> | <img src="plots/thermodynamics/fig_subtracted_energy_vs_invbeta_eta0p05.png" width="460" alt="Subtracted energy versus inverse temperature"> |

</details>

<details>
<summary><b>Spectral analysis</b></summary>
<br>

| Local effective-gap analysis at fixed $N_t$ |
|:---:|
| <img src="plots/spectrum/fig_effective_gaps_beta40_nt400.png" width="950" alt="Effective excitation gaps at fixed temporal extent"> |

| Continuum summary of excitation gaps |
|:---:|
| <img src="plots/spectrum/fig_gap_continuum_summary_beta40_eta2.png" width="950" alt="Continuum excitation-gap summary"> |

| (a) Coordinate correlator | (b) $\Delta_1$ continuum limit |
|:---:|:---:|
| <img src="plots/spectrum/fig_coordinate_correlator_beta40_eta_scan.png" width="460" alt="Coordinate correlator across lattice spacings"> | <img src="plots/spectrum/fig_gap_delta1_continuum_beta40_eta2.png" width="460" alt="First excitation-gap continuum extrapolation"> |

| (a) $\Delta_2$ continuum limit | (b) $\Delta_3$ continuum limit |
|:---:|:---:|
| <img src="plots/spectrum/fig_gap_delta2_continuum_beta40_eta2.png" width="460" alt="Second excitation-gap continuum extrapolation"> | <img src="plots/spectrum/fig_gap_delta3_continuum_beta40_eta2.png" width="460" alt="Third excitation-gap continuum extrapolation"> |

</details>

<details>
<summary><b>Position-distribution crossover</b></summary>
<br>

| (a) Position distributions | (b) Variance crossover |
|:---:|:---:|
| <img src="plots/distribution/fig_position_distribution_beta_scan.png" width="460" alt="Position-distribution thermal scan"> | <img src="plots/distribution/fig_position_variance_crossover.png" width="460" alt="Position-variance thermal crossover"> |

</details>

---

## 🔧 Build System

The Makefile exposes the retained entry points:

| Target | Purpose |
|:---|:---|
| `all` | Build `bin/qho_pimc` |
| `sampling-efficiency` | Compare update algorithms and autocorrelation costs |
| `equilibration-analysis` | Study equilibration and initialization dependence |
| `thermo-beta5` | Generate and analyze the fixed-$\beta=5$ thermodynamic scan |
| `path-snapshots` | Generate representative periodic paths |
| `plot-thermodynamics` | Plot thermodynamic and Euclidean-path results from local data |
| `position-distribution-plots` | Plot both retained position-distribution figures |
| `seed-generator` | Build the independent seed-stream utility |
| `clean` | Remove compiled outputs |

Typical module workflows are:

```bash
# Thermodynamic continuum analysis, then its figures
make thermo-beta5
make plot-thermodynamics

# Position-density ensembles, analysis, and figures
bash scripts/run/run_qho_position_distribution_production.sh
python3 scripts/analysis/analyze_qho_position_distribution.py
make position-distribution-plots

# Fixed-lattice-spacing thermal scan
TEMP_ETA=0.05 bash scripts/run/run_temperature_bare_scan.sh

# Harmonic spectrum analysis
python3 scripts/plotting/analyze_spectrum_eta_scan.py
python3 scripts/plotting/plot_effective_gaps.py
```

---

## ♻️ Data and Reproducibility

Raw and processed simulation outputs are not distributed. Local workflows
write below `data/`, where generated content remains ignored by Git. The 20
harmonic PNG figures used in the report are retained below `plots/`.

Reproducing the numerical estimates requires rerunning the relevant simulation
and analysis stages before invoking their plotters. The included figures do not
remove the need for those runtime inputs.

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

This module is released under the repository-level [MIT License](../LICENSE).
