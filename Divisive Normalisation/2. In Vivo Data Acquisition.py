# 1. Filter for a cell that is responsive (p < 0.05) and orientation-selective (OSI > 0.5)
responsive_cells = peak_table[
    (peak_table["ptest_dg"] < 0.05) & 
    (peak_table["osi_dg"] > 0.5)
]

print(f"Number of sharply tuned cells found: {len(responsive_cells)}")

# Pick the first tuned neuron
chosen_cell_id = responsive_cells.index[0]
cell_loc = peak_table.index.get_loc(chosen_cell_id)

# Find the temporal frequency that triggered its peak response
best_tf = responsive_cells.loc[chosen_cell_id, "tf_dg"]
tf_idx = np.where(dg_data.tfvals == best_tf)[0][0]

# 2. Extract empirical mean responses across orientations (index 0 is mean dF/F)
empirical_responses = response_array[:, tf_idx, cell_loc, 0]

print(f"\nAnalyzing Cell Specimen ID: {chosen_cell_id}")
print(f"OSI: {responsive_cells.loc[chosen_cell_id, 'osi_dg']:.2f}")
print(f"Preferred Temporal Frequency: {best_tf} Hz")
print(f"Empirical mean dF/F values: {np.round(empirical_responses, 3)}")

# 3. Fit the von Mises tuning function
def von_mises_model(theta, A, kappa, theta_pref, baseline):
    return A * np.exp(kappa * np.cos(2 * (theta - theta_pref))) + baseline

angles_rad = np.deg2rad(angles)
init_guess = [
    np.max(empirical_responses) - np.min(empirical_responses),
    2.0,
    angles_rad[np.argmax(empirical_responses)] % np.pi,
    np.min(empirical_responses)
]
bounds = ([0, 0, 0, -np.inf], [np.inf, 20, np.pi, np.inf])

popt, _ = curve_fit(von_mises_model, angles_rad, empirical_responses, p0=init_guess, bounds=bounds)
A_fit, kappa_fit, theta_pref_fit, baseline_fit = popt

print(f"\n--- Fitted Parameters ---")
print(f"Fitted Preferred Orientation: {np.rad2deg(theta_pref_fit):.1f}°")
print(f"Tuning Sharpness (kappa): {kappa_fit:.2f}")

# 4. Plot the empirical responses vs. fitted curve
fine_angles = np.linspace(0, 2 * np.pi, 300)
fitted_curve = von_mises_model(fine_angles, *popt)

plt.figure(figsize=(7, 4))
plt.scatter(angles, empirical_responses, color="crimson", zorder=3, label="Recorded Allen Data (dF/F)")
plt.plot(np.rad2deg(fine_angles), fitted_curve, color="navy", linestyle="--", label="Von Mises Fit")
plt.xlabel("Drifting Grating Orientation (degrees)")
plt.ylabel("Mean Response (dF/F)")
plt.title(f"V1 Orientation Tuning Curve (Cell {chosen_cell_id})")
plt.xticks(angles)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
