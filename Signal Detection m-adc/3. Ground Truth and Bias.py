import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Step 6: Define Symmetric / Bias-Free Fitting Function
# Enforces uniform criteria (c1 = c2 = c)
# ---------------------------------------------------------
def nll_unbiased(params, counts):
    d1, d2, c = params
    if d1 < 0 or d2 < 0:
        return 1e10
    prob_matrix = get_stimulus_response_matrix(d1, d2, c, c)
    prob_matrix = np.clip(prob_matrix, 1e-12, 1.0)
    return -np.sum(counts * np.log(prob_matrix))

# Fit unbiased model to the same simulated counts
res_unbiased = minimize(
    fun=nll_unbiased,
    x0=[0.1, 0.1, 0.0],
    args=(observed_counts,),
    method="BFGS"
)

d1_unb, d2_unb, c_unb = res_unbiased.x
print("\n--- Unbiased Model Fit (Forcing c1 = c2) ---")
print(f"Apparent d1: {d1_unb:.3f} (True: 1.00)")
print(f"Apparent d2: {d2_unb:.3f} (True: 1.00)")
print(f"Pooled criterion c: {c_unb:.3f}")

# ---------------------------------------------------------
# Step 7: Recreate Figure 3 Visualizations
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Recovery Parameter Comparison
labels = ["d1", "d2", "c1", "c2"]
x_pos = np.arange(len(labels))
axes[0].bar(x_pos - 0.2, true_params, width=0.4, label="True Parameters", color="gray", alpha=0.7)
axes[0].bar(x_pos + 0.2, result.x, width=0.4, yerr=param_standard_errors, capsize=5, label="2-ADC Recovered", color="royalblue", alpha=0.9)
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(labels)
axes[0].set_ylabel("Parameter Value")
axes[0].set_title("Ground Truth vs. 2-ADC Model Recovery")
axes[0].legend()
axes[0].grid(axis="y", linestyle="--", alpha=0.5)

# Plot 2: Masking of Bias Under Symmetry Assumption
# Comparing sensitivity estimates across models
sens_labels = ["d1 (Favored Location)", "d2 (Unfavored Location)"]
sens_x = np.arange(len(sens_labels))
axes[1].axhline(1.0, color="black", linestyle="--", label="True Sensitivity")
axes[1].scatter(sens_x, [result.x[0], result.x[1]], color="royalblue", s=100, label="2-ADC (With Bias)", zorder=3)
axes[1].scatter(sens_x, [d1_unb, d2_unb], color="crimson", s=100, marker="x", linewidth=3, label="Standard Model (No Bias)", zorder=3)
axes[1].set_xticks(sens_x)
axes[1].set_xticklabels(sens_labels)
axes[1].set_ylabel("Perceptual Sensitivity (d)")
axes[1].set_title("Sensitivity Distortion Under Biased Decision Making")
axes[1].legend()
axes[1].grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()
