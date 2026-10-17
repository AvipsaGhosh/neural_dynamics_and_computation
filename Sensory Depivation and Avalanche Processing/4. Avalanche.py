# ==============================================================================
# 4. NEURAL AVALANCHE EXTRACTION ENGINE (SIZE AND DURATION)
# ==============================================================================

def analyze_avalanches(spikes_matrix):
    """
    Extracts contiguous cascade sizes (total spike counts) and temporal
    durations across population activity sequences delimited by quiet bins.
    """
    pop_activity = np.sum(spikes_matrix, axis=0)
    sizes = []
    durations = []

    current_size = 0
    current_duration = 0
    in_avalanche = False

    for act in pop_activity:
        if act > 0:
            in_avalanche = True
            current_size += act
            current_duration += 1
        elif in_avalanche:
            sizes.append(current_size)
            durations.append(current_duration)
            current_size = 0
            current_duration = 0
            in_avalanche = False

    return sizes, durations

# Extract cascade profiles across experimental epochs
baseline_sizes, baseline_durs = analyze_avalanches(spike_history[:, :t_deprivation])
deprived_sizes, deprived_durs = analyze_avalanches(spike_history[:, t_deprivation:t_recovery])
recovered_sizes, recovered_durs = analyze_avalanches(spike_history[:, t_recovery:])
