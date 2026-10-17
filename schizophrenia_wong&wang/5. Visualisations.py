# 5. POPULATION DYNAMICS & VISUAL DIAGNOSTICS

plt.figure(figsize=(12, 6))

# Subplot 1: Spiking Raster Plot
plt.subplot(2, 1, 1)
plt.plot(spike_monitor.t / ms, spike_monitor.i, '.k', markersize=2.5)
plt.axvspan(200, 500, color='gray', alpha=0.25, label='Stimulus ON (Cue)')
plt.title(f'Prefrontal Microcircuit Dynamics: {condition_label}', fontsize=12)
plt.ylabel('Neuron Index')
plt.xlim(0, 1500)
plt.legend(loc='upper right')

# Subplot 2: Smoothed Population Firing Rate
plt.subplot(2, 1, 2)
smoothed_rate = rate_monitor.smooth_rate(window='flat', width=20.1 * ms)
plt.plot(rate_monitor.t / ms, smoothed_rate / Hz, color=plot_color, lw=1.8)
plt.axvspan(200, 500, color='gray', alpha=0.25)
plt.xlabel('Time (ms)')
plt.ylabel('Population Rate (Hz)')
plt.xlim(0, 1500)

plt.tight_layout()

# Save high-resolution figure for GitHub repository assets
fig_filename = "wang_memory_healthy.png" if is_healthy else "wang_memory_schizophrenia.png"
plt.savefig(fig_filename, dpi=300, bbox_inches="tight")
plt.show()
