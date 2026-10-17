# ==============================================================================
# 5. MULTI-PANEL CRITICALITY DIAGNOSTICS AND POWER-LAW ANALYSIS
# ==============================================================================

fig, axes = plt.subplots(4, 1, figsize=(10, 15))

# Panel 1: Network Population Firing Rate Trajectory
axes[0].plot(activity_history, color='indigo', alpha=0.6, label='Network Firing Rate')
axes[0].axvline(t_deprivation, color='darkorange', linestyle='--', linewidth=1.5, label='Sensory Deprivation Onset')
axes[0].axhline(r_target, color='black', linestyle=':', linewidth=1.2, label='Homeostatic Target Setpoint')
axes[0].set_title('E/I Network Mean Firing Rate Dynamics', fontsize=11, weight='bold')
axes[0].set_ylabel('Mean Population Rate', fontsize=10)
axes[0].legend(loc='upper right')

# Panel 2: Homeostatic Synaptic Weight Adaptation
axes[1].plot(weight_history_E, color='forestgreen', linewidth=1.8, label='Mean Excitatory Weight Magnitude')
axes[1].axvline(t_deprivation, color='darkorange', linestyle='--', linewidth=1.5)
axes[1].set_title('Multiplicative Synaptic Scaling of Excitatory Projections', fontsize=11, weight='bold')
axes[1].set_xlabel('Time (ms)', fontsize=10)
axes[1].set_ylabel('Synaptic Weight', fontsize=10)
axes[1].legend(loc='upper left')

# Panel 3: Log-Log Avalanche Size Distributions
epoch_labels = ['Baseline Epoch', 'Immediate Post-Deprivation', 'Post-Homeostatic Recovery']
colors = ['teal', 'crimson', 'darkorchid']

has_size_artists = False
for size_data, label, color in zip([baseline_sizes, deprived_sizes, recovered_sizes], epoch_labels, colors):
    if len(size_data) > 0:
        sizes, counts = np.unique(size_data, return_counts=True)
        prob = counts / np.sum(counts)
        axes[2].loglog(sizes, prob, marker='o', linestyle='none', label=label, color=color, alpha=0.7)
        has_size_artists = True

axes[2].set_title('Neural Avalanche Size Distributions (P(S) vs S)', fontsize=11, weight='bold')
axes[2].set_xlabel('Avalanche Size (Total Spikes)', fontsize=10)
axes[2].set_ylabel('P(Size)', fontsize=10)
if has_size_artists:
    axes[2].legend(loc='upper right')

# Panel 4: Log-Log Avalanche Duration Distributions
has_dur_artists = False
for dur_data, label, color in zip([baseline_durs, deprived_durs, recovered_durs], epoch_labels, colors):
    if len(dur_data) > 0:
        durs, counts = np.unique(dur_data, return_counts=True)
        prob = counts / np.sum(counts)
        axes[3].loglog(durs, prob, marker='s', linestyle='none', label=label, color=color, alpha=0.7)
        has_dur_artists = True

axes[3].set_title('Neural Avalanche Duration Distributions (P(T) vs T)', fontsize=11, weight='bold')
axes[3].set_xlabel('Avalanche Duration (ms)', fontsize=10)
axes[3].set_ylabel('P(Duration)', fontsize=10)
if has_dur_artists:
    axes[3].legend(loc='upper right')

plt.tight_layout()

# Save figure for documentation
plt.savefig('sensory_deprivation_avalanches.png', dpi=300, bbox_inches='tight')
plt.show()
