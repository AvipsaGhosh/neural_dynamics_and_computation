# 1. SIMULATION PARAMETERS AND NUMERICAL CONSTANTS

from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np

# Numerical integration constants
DT = 0.001  # Integration time step (seconds)
MAX_TIME = 3.0  # Maximum trial duration ceiling (seconds)
SQRT_DT = np.sqrt(DT)  # Precomputed square root for Wiener process increments

# Experimental parameters
N_TRIALS = 1000  # Number of simulated trials per clinical condition
RANDOM_SEED = 42  # Seed for deterministic reproducibility

# Set random seed
np.random.seed(RANDOM_SEED)
