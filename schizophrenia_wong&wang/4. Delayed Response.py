# 4. DELAYED-RESPONSE TASK PROTOCOL

# Recording electrodes
spike_monitor = SpikeMonitor(G)
rate_monitor = PopulationRateMonitor(G)

# Phase A: Pre-stimulus baseline (0 to 200 ms)
G.I_ext = 0 * nA
run(200 * ms)

# Phase B: Cue presentation (200 to 500 ms)
# Transient depolarizing drive triggers the memory ensemble
G.I_ext = 0.4 * nA
run(300 * ms)

# Phase C: Delay period / Working memory phase (500 to 1500 ms)
# Depolarizing drive removed; network must sustain firing via recurrent loop
G.I_ext = 0 * nA
run(1000 * ms)
