# Neural Communication Subspaces

This repository contains a modular and rigorous computational pipeline implementing the framework of Semedo and colleagues regarding inter-areal communication subspaces between neural populations, such as V1 and V2, enhanced with custom extensions for robust cross-validation, dynamic subspace tracking with uncertainty quantification, subpopulation multiplexing analysis, and surrogate null validation.

---

## Architecture and Repository Structure

The codebase is segregated into seven self-contained modules, orchestrated together to ensure methodological soundness and prevent common pitfalls like information leakage:

1. **preprocessing.py**
   - Handles raw session data loading via SciPy.
   - Filters out inactive neurons based on a customizable minimum firing rate threshold.
   - Computes trial-to-trial peri-stimulus time histogram residuals to isolate trial-specific fluctuations.
   - Provides temporal extraction utilities for multi-dimensional neural tensors.

2. **cv_preprocessor.py**
   - Implements trial-wise cross-validation to maintain temporal and trial integrity across folds.
   - Computes condition-specific peri-stimulus time histograms exclusively on training trials.
   - Residualizes both training and test data using training-derived baselines to strictly prevent data leakage.

3. **semedo_baseline.py**
   - Implements regularized Reduced-Rank Regression using numerically stable linear solvers.
   - Extracts source communication subspace bases and target projection matrices for a specified subspace rank.
   - Computes out-of-sample predictive performance across held-out test folds.

4. **subspace_tracker.py**
   - Tracks temporal rotation dynamics of communication subspaces across sliding time windows.
   - Incorporates rank-deficiency safeguards and QR orthonormalization.
   - Integrates trial-level bootstrapping to compute confidence intervals for inter-window rotation angles over time.

5. **multiplexed_analyzer.py**
   - Partitions target population units based on unresidualized baseline firing metrics or biological attributes.
   - Computes cross-subpopulation subspace overlap using principal angles derived from singular value decomposition.

6. **subspace_null_validator.py**
   - Generates surrogate null distributions by shuffling trial alignments between source and target tensors.
   - Evaluates empirical p-values to test the statistical significance of observed subspace geometry against chance models.

7. **main.py**
   - The central orchestration script that chains all modules together into a unified end-to-end execution pipeline.

---

## Requirements and Dependencies

Ensure your Python environment is equipped with the following core numerical and scientific packages:

* numpy
* scipy
* scikit-learn

You can install them via pip:

```bash
pip install numpy scipy scikit-learn
