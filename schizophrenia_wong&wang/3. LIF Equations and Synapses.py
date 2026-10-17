# 3. CONDUCTANCE-BASED LIF EQUATIONS & SYNAPSES

# Leaky integrate-and-fire equations with slow recurrent NMDA gating
# xi captures continuous background synaptic fluctuations
neuron_eqs = '''
dv/dt = (
    (v_rest - v)
    + g_nmda * (E_exc - v)
    + R_membrane * I_ext
) / tau_m + sigma_noise * xi * tau_m**-0.5 : volt

dg_nmda/dt = -g_nmda / tau_nmda : 1
I_ext : amp
'''

# Instantiate prefrontal neuronal pool
G = NeuronGroup(
    N_neurons,
    neuron_eqs,
    threshold='v > v_threshold',
    reset='v = v_reset',
    method='euler'
)

# Initialize states
G.v = v_rest
G.g_nmda = 0

# Recurrent excitatory connections mediated by NMDA channels
S = Synapses(G, G, on_pre=f'g_nmda += {nmda_w}')
S.connect(p=connection_p)
