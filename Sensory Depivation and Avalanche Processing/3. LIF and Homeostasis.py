# ==============================================================================
# 3. VECTORIZED LIF DYNAMICS AND HOMEOSTATIC SYNAPTIC SCALING
# ==============================================================================

for t in range(T):
    # Select external sensory drive regime
    I_ext = I_baseline if t < t_deprivation else I_deprived
    noise = np.random.normal(0, 0.1, N)

    # Synaptic integration via matrix multiplication over signed presynaptic inputs
    synaptic_input = W @ spike_history[:, t - 1] if t > 0 else np.zeros(N)

    # Voltage integration step
    dV = (-(V - v_reset) + I_ext + synaptic_input + noise) * (dt / tau_m)
    V += dV

    # Threshold evaluation and refractory reset
    spikes = (V >= v_thr).astype(int)
    spike_history[:, t] = spikes
    V[spikes == 1] = v_reset

    # Continuous running estimate of local firing rate
    firing_rates += (dt / tau_h) * (-firing_rates + spikes)

    # Diagnostic recording
    activity_history[t] = np.mean(spikes)
    weight_history_E[t] = np.mean(np.abs(W[:, is_excitatory]))

    # Multiplicative homeostatic scaling of incoming excitatory conductances
    if t > 0:
        scaling_factor = eta * (1.0 - (firing_rates / r_target))
        W[:, is_excitatory] *= (1.0 + scaling_factor[:, None])

        # Impose bounds preserving signed Dale's Law topology
        W[:, is_excitatory] = np.clip(W[:, is_excitatory], 0.0, 1.5)
        W[:, is_inhibitory] = np.clip(W[:, is_inhibitory], -1.5, 0.0)
        np.fill_diagonal(W, 0)
