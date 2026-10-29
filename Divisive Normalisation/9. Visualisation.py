fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)

max_val = max(y_test.max(), y_pred_dn.max(), y_pred_mlp.max()) * 1.15
lims = [0, max_val]

# Divisive Normalization
axes[0].scatter(y_test, y_pred_dn, color="steelblue", s=70, edgecolors="k", alpha=0.85)
axes[0].plot(lims, lims, "k--", alpha=0.7, label="Unity")
axes[0].set_title(f"Divisive Normalization\nr = {r_dn:.2f} | R² = {r2_dn:.2f}")
axes[0].set_xlabel("Empirical Response (dF/F)")
axes[0].set_ylabel("Predicted Response")
axes[0].set_xlim(lims)
axes[0].set_ylim(lims)
axes[0].grid(True, linestyle=":", alpha=0.6)
axes[0].legend(frameon=False)

# Multi-Layer Perceptron
axes[1].scatter(y_test, y_pred_mlp, color="crimson", s=70, edgecolors="k", alpha=0.85)
axes[1].plot(lims, lims, "k--", alpha=0.7, label="Unity")
axes[1].set_title(f"Equalized Backprop MLP\nr = {r_mlp:.2f} | R² = {r2_mlp:.2f}")
axes[1].set_xlabel("Empirical Response (dF/F)")
axes[1].set_xlim(lims)
axes[1].grid(True, linestyle=":", alpha=0.6)
axes[1].legend(frameon=False)

plt.tight_layout()
plt.show()
