# -------------------------------------------------------------
# 3. Dynamic Gain Control Slices (Target Drive vs Response)
# -------------------------------------------------------------
# Continuous evaluation across the full target drive spectrum
eval_dt = np.linspace(0.0, 1.0, 100)

# Select three distinct levels of empirical pool suppression
pool_levels = [0.1, 0.5, 0.9]
palette = ["#2b5c8f", "#d95f02", "#7570b3"]

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# Left Panel: Divisive Normalization Slices
for p_level, col in zip(pool_levels, palette):
    x_probe = np.column_stack([eval_dt, np.full_like(eval_dt, p_level)])
    curve_dn = dn_pool_predict(x_probe, rmax_opt, sigma_opt, w_opt)
    axes[0].plot(
        eval_dt,
        curve_dn,
        color=col,
        linewidth=2.2,
        label=f"Pool Drive = {p_level:.1f}"
    )

axes[0].set_title("Divisive Normalization Gain Control")
axes[0].set_xlabel("Target Excitatory Drive (D_target)")
axes[0].set_ylabel("Predicted Firing Rate")
axes[0].set_xlim(0, 1)
axes[0].set_ylim(bottom=0)
axes[0].grid(True, linestyle=":", alpha=0.5)
axes[0].legend(frameon=False)

# Right Panel: Multi-Layer Perceptron Slices
mlp_equal.eval()
with torch.no_grad():
    for p_level, col in zip(pool_levels, palette):
        x_probe = np.column_stack([eval_dt, np.full_like(eval_dt, p_level)])
        probe_tensor = torch.tensor(x_probe, dtype=torch.float32)
        curve_mlp = mlp_equal(probe_tensor).squeeze().numpy()
        axes[1].plot(
            eval_dt,
            curve_mlp,
            color=col,
            linewidth=2.2,
            label=f"Pool Drive = {p_level:.1f}"
        )

axes[1].set_title("MLP Transfer Characteristics")
axes[1].set_xlabel("Target Excitatory Drive (D_target)")
axes[1].set_xlim(0, 1)
axes[1].grid(True, linestyle=":", alpha=0.5)
axes[1].legend(frameon=False)

plt.tight_layout()
plt.show()
