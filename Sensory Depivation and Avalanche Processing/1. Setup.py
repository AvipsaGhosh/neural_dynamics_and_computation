# ==============================================================================
# 1. SIMULATION SETUP AND BIOPHYSICAL PARAMETERS
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Population dimensions
N_E = 80                           # Excitatory pyramidal neurons
N_I = 20                           # Inhibitory interneurons
N = N_E + N_I                      # Total network size

# Integration and membrane time constants
T = 10000                          # Total simulation steps (ms)
dt = 1                             # Time resolution step (ms)
tau_m = 10                         # Membrane leak time constant (ms)
v_thr = 1.0                        # Spike generation threshold
v_reset = 0.0                      # Post-spike reset potential

# Homeostatic plasticity parameters
tau_h = 500                        # Integration window for estimated firing rate (ms)
eta = 0.01                         # Homeostatic learning rate
r_target = 0.2                     # Target firing rate setpoint (spikes/ms)

# Sensory protocol epochs
t_deprivation = 3000               # Onset of sensory input reduction
t_recovery = 7000                  # Observation boundary for homeostatic compensation

# Neuronal identity masks
is_excitatory = np.zeros(N, dtype=bool)
is_excitatory[:N_E] = True
is_inhibitory = ~is_excitatory
