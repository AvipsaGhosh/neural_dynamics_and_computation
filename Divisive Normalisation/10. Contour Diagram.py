# Generate a dense evaluation grid across the feature domain
grid_dt, grid_dp = np.meshgrid(np.linspace(0, 1, 60), np.linspace(0, 1, 60))
X_surface = np.column_stack([grid_dt.ravel(), grid_dp.ravel()])

# Predict surface with Divisive Normalization
surf_dn = dn_pool_predict(X_surface, rmax_opt, sigma_opt, w_opt).reshape(60, 60)

# Predict surface with MLP
with torch.no_grad():
    surf_mlp = mlp_equal(torch.tensor(X_surface, dtype=torch.float32)).numpy().reshape(60, 60)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Divisive Normalization Contour
c0 = axes[0].contourf(grid_dt, grid_dp, surf_dn, levels=20, cmap="viridis")
axes[0].scatter(X_equalized[:, 0], X_equalized[:, 1], c=y, cmap="viridis", edgecolors="w", s=40)
axes[0].set_title("Divisive Normalization Manifold")
axes[0].set_xlabel("Target Drive (D_target)")
axes[0].set_ylabel("Empirical Pool Drive (D_pool)")
fig.colorbar(c0, ax=axes[0])

# MLP Contour
c1 = axes[1].contourf(grid_dt, grid_dp, surf_mlp, levels=20, cmap="viridis")
axes[1].scatter(X_equalized[:, 0], X_equalized[:, 1], c=y, cmap="viridis", edgecolors="w", s=40)
axes[1].set_title("MLP Manifold")
axes[1].set_xlabel("Target Drive (D_target)")
axes[1].set_ylabel("Empirical Pool Drive (D_pool)")
fig.colorbar(c1, ax=axes[1])

plt.tight_layout()
plt.show()
