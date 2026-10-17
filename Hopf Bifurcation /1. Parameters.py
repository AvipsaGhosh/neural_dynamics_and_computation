# ==============================================================================
# 1. NUMERICAL INTEGRATION CONSTANTS AND MODEL PARAMETERS
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt

# Integration configuration
DT = 0.005           # Time step (seconds)
T_MAX = 120.0        # Total simulation horizon (seconds)
STEPS = int(T_MAX / DT)

# Intrinsic angular frequency of the oscillator
OMEGA = 15.0         # Rad/s

# Stochastic diffusion intensity (set to 0.0 for pure theoretical validation)
NOISE_STD = 0.0

# Bifurcation control parameter sweep range
Z_MIN = -0.5
Z_MAX = 0.5
N_POINTS = 80
Z_SWEEP = np.linspace(Z_MIN, Z_MAX, N_POINTS)
