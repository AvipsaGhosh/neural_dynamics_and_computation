# 1. SIMULATION ENGINE AND GLOBAL SETUP


from brian2 import *
import matplotlib.pyplot as plt

# Ensure reproducible NumPy code generation and reset runtime scope
prefs.codegen.target = 'numpy'
start_scope()

# Network architectural dimensions
N_exc = 80   # Excitatory pyramidal neuron population
N_inh = 20   # Inhibitory interneuron population

# Passive membrane properties
tau_m = 20 * ms
v_rest = -65 * mV
v_reset = -65 * mV
v_threshold = -50 * mV
R_membrane = 100 * Mohm

# Synaptic reversal potentials
E_exc = 0 * mV     # Glutamatergic reversal
E_inh = -80 * mV   # GABAergic reversal

# Synaptic kinetic time constants
tau_nmda = 100 * ms   # Slow NMDA decay (substrate for persistent attractor states)
tau_gaba = 10 * ms    # Rapid GABAergic feedback inhibition
