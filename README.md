# Multiantenna Project — Capon Beamforming

This project contains Python scripts for studying **conventional Capon beamforming (SCB)** and **Robust Capon Beamforming (RCB)**, using Li, Stoica, and Wang's 2003 paper *On Robust Capon Beamforming and Diagonal Loading* as a reference.

The main goal is to investigate how steering-vector mismatch, the number of snapshots, and the size of the RCB uncertainty set affect beampatterns and desired-signal power estimation.

## Project structure

```text
Multiantenna-Project/
├── README.md
├── On_robust_Capon_beamforming_and_diagonal_loading.pdf
├── Graphs/
│   ├── beampatterns_mismatch_2_eps5.png
│   ├── beampatterns_nomismatch.png
│   ├── power_convergence_nomismatch.png
│   ├── soi_rejection_mismatch2_eps1.png
│   ├── soi_rejection_mismatch5_eps1.png
│   └── soi_rejection_mismatch5_eps8.png
└── Python Scripts/
    ├── conventional_vs_robust_nomismatch.py
    ├── conventional_vs_robust_mismatch.py
    ├── steering_vector_distances.py
    ├── ellipse.py
    └── robust_capon_practical.py
```

## Requirements

The scripts use Python and, depending on the file, the following libraries:

- `numpy`
- `matplotlib`
- `scipy` (in particular, `scipy.optimize.brentq`)

Install them with:

```bash
python -m pip install numpy matplotlib scipy
```

Run a script from a terminal, for example:

```bash
python "Python Scripts/conventional_vs_robust_nomismatch.py"
```

Scripts display plots and/or print numerical results in the terminal. Not every script automatically saves its figures to the `Graphs/` directory.

## Python scripts

### 1. `conventional_vs_robust_nomismatch.py`

**Purpose:** Compare SCB and RCB when there is no mismatch in the desired signal's direction.

- The desired signal is generated from the nominal direction supplied to the algorithms (`theta_s = 0°` in the code).
- Builds a theoretical covariance matrix and sample covariance matrices from simulated snapshots.
- Implements SCB and RCB and compares their beampatterns.
- Includes a Monte Carlo experiment for estimated signal power versus the number of snapshots, showing how the estimate changes as more data become available.
- Prints numerical power results and generates convergence and beampattern plots.

**Why it is useful:** It provides a baseline without angular mismatch and illustrates the effect of snapshot count on estimation.

### 2. `conventional_vs_robust_mismatch.py`

**Purpose:** Study the effect of a mismatch between the actual direction of arrival of the desired signal and the nominal direction supplied to the beamformers.

Main settings in the file:

- Uniform linear array with `M = 10` antennas and spacing `d = 0.5` wavelengths.
- Nominal direction: `theta_s = 0°`.
- Actual desired-signal direction: `theta_s_true = 2°`.
- Interferers at `60°` and `80°`.
- Powers: desired signal `10 dB`, each interferer `20 dB`, and noise `0 dB`.
- The theoretical covariance and snapshots are generated using the actual steering vector `a(2°)`, while SCB and RCB are given the nominal vector `a(0°)`.
- Compares SCB/RCB beampatterns using the theoretical covariance and sample covariances for several snapshot counts.
- The final section adds comparisons for different mismatch and `epsilon` scenarios (`2°/1`, `5°/1`, and `5°/8`).

**Why it is useful:** It shows whether SCB attenuates the desired signal when the nominal steering vector differs from the actual one, and explores how RCB behaves as the uncertainty set changes.

The initial beampattern section uses `epsilon = 5`; the final scenario section uses the three values listed above.

### 3. `steering_vector_distances.py`

**Purpose:** Measure how far the actual steering vector is from the nominal one at different angles.

- Uses `M = 10`, spacing `d = 0.5` wavelengths, and nominal vector `a(0°)`.
- Sweeps actual angles from `0°` to `10°` in increments of `0.1°`.
- Calculates, for each angle,

  \[
  \epsilon_{\min}(\theta)=\|\mathbf a(\theta)-\mathbf a(0^\circ)\|^2.
  \]

- Prints a table of the computed values and plots squared distance versus angle.
- Also prints the values for `2°`, `5°`, and `10°`.

**Why it is useful:** It estimates the minimum `epsilon` required for a particular angular steering vector to lie inside the uncertainty sphere centered at `a(0°)`, using this project's steering-vector and distance definitions. Set membership alone does not guarantee that RCB will recover the exact actual vector.

### 4. `ellipse.py`

**Purpose:** Visualize elliptical uncertainty regions and compare their sizes for different values of `epsilon`.

- Constructs a region using an eigendecomposition (eigenvalues and eigenvectors).
- Plots the regions for the `epsilon` values defined in the file (by default, `0.5` and `2.0`).
- Prints numerical information associated with each case.

**Why it is useful:** It provides a geometric illustration of how the uncertainty region changes with `epsilon`. Before interpreting the figure as a direct visualization of a particular RCB implementation, check the matrix and parameterization used to construct the ellipses in the script.

### 5. `robust_capon_practical.py`

**Purpose:** Explore a practical SCB/RCB implementation in two interference/noise scenarios and inspect both angular responses and numerical results.

- Uses a nominal desired-signal direction of `25°` and an interferer at `40°`.
- Constructs theoretical covariance matrices for each scenario.
- Computes a robust solution using eigendecomposition and a search for the Lagrange multiplier.
- Calculates SCB and RCB weights, compares their beampatterns, and breaks output power down into desired-signal, interference, and noise contributions.
- Prints estimates, steering-vector relative error, an uncertainty-constraint check, and execution times.

**Why it is useful:** It helps examine the RCB calculation flow and internal quantities in addition to the final plots. It is a separate example from the `0°`/`2°` angular mismatch experiments, so its parameters and formulation should be assessed separately before making quantitative comparisons.

## Included plots

The PNG files in `Graphs/` are pre-generated figures retained as reference results. Based on their filenames, they correspond to:

- `beampatterns_nomismatch.png`: beampatterns without mismatch.
- `power_convergence_nomismatch.png`: desired-signal power-estimation convergence without mismatch.
- `beampatterns_mismatch_2_eps5.png`: beampatterns with a `2°` mismatch and `epsilon = 5`.
- `soi_rejection_mismatch2_eps1.png`: desired-signal rejection for a `2°` mismatch and `epsilon = 1`.
- `soi_rejection_mismatch5_eps1.png`: desired-signal rejection for a `5°` mismatch and `epsilon = 1`.
- `soi_rejection_mismatch5_eps8.png`: desired-signal rejection for a `5°` mismatch and `epsilon = 8`.

These descriptions are inferred from the filenames. To confirm how each image was produced, check the script and exact parameter values used for that run.

## Key concepts

- **SCB (Standard/Conventional Capon Beamforming):** Computes weights that minimize output power using the covariance matrix, subject to a unit-response constraint for the nominal steering vector.
- **RCB (Robust Capon Beamforming):** Incorporates an uncertainty set so that the actual steering vector can differ from the nominal one.
- **Mismatch:** A discrepancy between the actual steering vector and the one supplied to the algorithm. In the main angular experiments, the actual signal may arrive from `2°` or `5°`, while the algorithms use `a(0°)`.
- **Snapshots:** Time samples of the received array signal used to estimate the covariance matrix.
- **Theoretical and sample covariance:** The theoretical covariance is constructed from the signal model; the sample covariance is estimated from simulated snapshots and varies between realizations.
- **`epsilon`:** Controls the size of the RCB uncertainty set. A larger value permits a larger region, but does not automatically guarantee better performance.

## Recommended order for exploring the project

1. Read the paper PDF to understand the SCB and RCB formulations.
2. Run `conventional_vs_robust_nomismatch.py` as a baseline.
3. Run `conventional_vs_robust_mismatch.py` to investigate angular mismatch.
4. Use `steering_vector_distances.py` to relate angular mismatch to the squared distance between steering vectors and the `epsilon` needed to include them.
5. Run `ellipse.py` to visualize uncertainty regions.
6. Inspect `robust_capon_practical.py` as a complementary example of calculations, checks, and execution times.

## Notes on interpreting results

- The actual steering vector being inside the uncertainty set does not guarantee that RCB will estimate it exactly.
- Normalized beampatterns show the relative shape of the pattern; on their own, they do not establish that desired-signal power has been estimated correctly.
- Sample-covariance results depend on random snapshots. Some scripts fix a random seed, while others do not.
- Before presenting the results as an exact reproduction of the paper, verify that the parameters, definition of `epsilon`, implemented equations, and simulation conditions match those in the paper.
