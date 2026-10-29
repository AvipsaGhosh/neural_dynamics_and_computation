# Benchmarking Divisive Normalization Against Machine Learning in Mouse V1

An equalized empirical study evaluating whether canonical cortical divisive normalization outgeneralizes an unconstrained multi-layer perceptron on in vivo two-photon calcium imaging recordings from the mouse visual cortex.

---

## Overview

Cortical computation in primary visual cortex (VISp) is classically characterized by linear receptive field tuning modulated by non-linear recurrent surround suppression—a computation formalized as divisive normalization. In modern computational neuroscience, a fundamental question emerges:

Does an unconstrained feedforward neural network trained via backpropagation naturally synthesize biological gain control, or does an analytical three-parameter biophysical prior better capture real population-mediated gain dynamics?

This repository benchmarks a classical three-parameter divisive normalization equation against a 625-parameter multi-layer perceptron (MLP). Both models are evaluated on identical, strictly equalized feature sets constructed from physiological recordings released by the Allen Brain Observatory.

---

## Biological Data and Preprocessing Pipeline

### 1. In Vivo Calcium Imaging Acquisition
Recordings were obtained from two-photon optical physiology sessions in awake mice viewing drifting sinusoidal gratings across 8 orientations (0° to 315° in 45° increments) and 5 temporal frequencies (1, 2, 4, 8, and 15 Hz).

### 2. Physiological Filtering and Receptive Field Characterization
The biological population was filtered to isolate an exemplary orientation-selective cortical unit:
* Visual responsiveness criterion: p < 0.05
* Orientation Selectivity Index: OSI > 0.5

The target unit's classical receptive field was parameterized across orientation space using a circular von Mises distribution:

R(theta) = A * exp(kappa * cos(2 * (theta - theta_pref))) + baseline

This established the cell's optimal orientation preference (theta_pref) and concentration parameter (kappa) at its optimal temporal frequency.

### 3. The Empirical Cortical Normalization Pool
Unlike synthetic benchmarks that approximate surround inhibition via idealized filter banks, this pipeline measures the true biological normalization pool:
* The instantaneous population activity across all other simultaneously recorded neurons within the local V1 column is extracted for each condition.
* The leave-one-out population mean acts as an authentic biological proxy for local recurrent network drive:

Pool(theta, tf) = mean(all_other_cells(theta, tf))

---

## Experimental Parity: The 2D Equalized Drive Space

To eliminate inductive asymmetry, both models are supplied with the exact same two-dimensional driving vector:

x = [D_target(theta, tf), D_pool(theta, tf)]

Where:
* D_target: The target unit's classical excitatory drive, parameterized via circular Gaussian orientation tuning and log-Gaussian temporal frequency tuning.
* D_pool: The empirical normalization pool drive extracted from the surrounding cortical network, rescaled to the unit interval [0, 1].

The dataset comprises all valid stimulus conditions randomly partitioned into an 80/20 train/test split.

---

## Model Architectures

### Model 1: Biophysical Divisive Normalization (3 Parameters)
The standard canonical formulation parameterized with bounded non-linear optimization (L-BFGS-B):

R = R_max * (D_target^p) / (sigma^p + w * D_pool^p)

* Free parameters: R_max (asymptotic scale), sigma (semi-saturation constant), and w (pool coupling weight).
* Transduction exponent: p = 2.0.

### Model 2: Equalized Backprop MLP (625 Parameters)
A feedforward neural network receiving the continuous input pair [D_target, D_pool]:
* Architecture: Linear(2 -> 32) -> ReLU -> Linear(32 -> 16) -> ReLU -> Linear(16 -> 1)
* Optimizer: Adam (lr = 0.01) with L2 weight decay (0.001) over 400 epochs.

---

## Quantitative Results

Evaluated on out-of-sample held-out cortical conditions:

| Model | Free Parameters | Pearson r | Coefficient of Determination (R²) |
| :--- | :---: | :---: | :---: |
| Divisive Normalization | 3 | 0.897 | 0.522 |
| Equalized Backprop MLP | 625 | 0.890 | 0.762 |

### Key Findings
* Tuning Fidelity (Pearson r around 0.90): The 3-parameter biophysical equation slightly outperforms the 625-parameter network in rank-order tracking, capturing the natural rise and fall of neural firing across conditions.
* Scale Concordance (R²): The unconstrained MLP achieves higher absolute mean squared calibration (R² = 0.762) because its flexible hidden projections directly rescale output offsets against calcium fluorescence dynamics.

---

## Diagnostic Visualizations and Mechanistic Analysis

### 1. Parity and Residual Concordance
* Divisive Normalization: Displays a tight, linear trajectory tracking the relative amplitude of test conditions, with a moderate vertical scaling overshoot at peak firing rates that accounts for the lower R² score[cite: 1].
* Backprop MLP: Predictions cluster symmetrically along the identity line with uniform residual dispersion across the entire dynamic range[cite: 1].

### 2. Learned 2D Response Surfaces
* Divisive Normalization Manifold: Exhibits a smooth, continuous hyperbolic decay flaring outward toward maximum excitatory drive in low-inhibition regimes[cite: 2].
* MLP Manifold: Forms piece-wise linear steps and localized plateaus directly surrounding empirical training clusters, demonstrating how unconstrained gradient descent fits local density rather than discovering global scale invariance[cite: 2].

### 3. Dynamic Gain Control Profiles
Evaluating model transfer curves across increasing surround pool strengths (D_pool in [0.1, 0.5, 0.9]):
* Divisive Normalization: Demonstrates canonical multiplicative gain control: as surround drive intensifies, the slope bends downward while strictly pinning responses to zero at zero drive (D_target = 0)[cite: 3].
* Backprop MLP: Captures qualitative suppression but exhibits non-linear saturation kinks and artificial baseline drift, predicting non-zero spontaneous rates when target excitatory drive is completely absent[cite: 3].

---

## Repository Structure

```text
├── notebooks/
│   ├── 01_in_silico_plaid_suppression.ipynb       # Theoretical Gabor & CNN plaid simulation
│   └── 02_in_vivo_cortical_benchmark.ipynb        # Primary Allen V1 empirical benchmark
├── allen_data_light/                              # Local cache directory for Allen SDK metadata
├── requirements.txt                               # Environment dependencies
└── README.md                                      # Project documentation
