# ==============================================================================
# 2. DALE'S LAW CONNECTIVITY AND DYNAMIC STATE INITIALIZATION
# ==============================================================================

# Initialize baseline absolute coupling magnitudes
W_mag = np.random.uniform(0.0, 0.1, (N, N))
np.fill_diagonal(W_mag, 0)

# Enforce strict Dale's principle: presynaptic columns carry uniform sign
W = W_mag.copy()
W[:, is_inhibitory] *= -2.5       # Heightened inhibitory efficacy ensures initial balance

# Dynamic state vectors and monitoring arrays
V = np.zeros(N)
firing_rates = np.zeros(N)         # Running low-pass estimate of individual activity
activity_history = np.zeros(T)
weight_history_E = np.zeros(T)
spike_history = np.zeros((N, T))

# Driving currents across conditions
I_baseline = np.zeros(N)
I_baseline[is_excitatory] = 0.4
I_baseline[is_inhibitory] = 0.2

I_deprived = np.zeros(N)
I_deprived[is_excitatory] = 0.1    # Substantial loss of incoming sensory excitation
I_deprived[is_inhibitory] = 0.1
