# ==============================================================================
# 5. MULTI-PANEL BIFURCATION & DYNAMICAL DIAGNOSTICS
# ==============================================================================

fig, axs = plt.subplots(2, 2, figsize=(13, 10))

# Panel 1: Bifurcation Diagram (Simulated vs Analytical sqrt(Z))
axs[0, 0].scatter(Z_SWEEP, measured_amp, s=35, color="royalblue", alpha=0.8, label="Numerical Simulation")
axs[0, 0].plot(Z_SWEEP, theoretical_amp, color="firebrick", linewidth=2.2, label=r"Theory: $r = \sqrt{Z}$")
axs[0, 0].axvline(0, color="black", linestyle="--", alpha=0.7, label=r"Bifurcation Criticality ($Z_c=0$)")
axs[0, 0].set_title("Supercritical Hopf Bifurcation Diagram", fontsize=12, weight="bold")
axs[0, 0].set_xlabel("Bifurcation Parameter (Z)", fontsize=11)
axs[0, 0].set_ylabel("Asymptotic Orbit Radius (r)", fontsize=11)
axs[0, 0].legend(loc="upper left")
axs[0, 0].grid(True, linestyle=":", alpha=0.5)

# Panel 2: Subcritical Phase Plane (Stable Spiral / Focus)
axs[0, 1].plot(E_sub, I_sub, color="darkcyan", lw=1.2)
axs[0, 1].plot(E_sub[0], I_sub[0], 'go', markersize=6, label="Initial State")
axs[0, 1].plot(0, 0, 'rx', markersize=8, label="Stable Fixed Point")
axs[0, 1].set_title(f"Subcritical Regime (Z = {Z_sub}): Damped Focus", fontsize=12, weight="bold")
axs[0, 1].set_xlabel("Excitatory Variable (E)", fontsize=11)
axs[0, 1].set_ylabel("Inhibitory Variable (I)", fontsize=11)
axs[0, 1].legend(loc="upper right")
axs[0, 1].grid(True, linestyle=":", alpha=0.5)

# Panel 3: Supercritical Phase Plane (Stable Limit Cycle)
axs[1, 0].plot(E_super, I_super, color="navy", lw=1.2)
axs[1, 0].plot(E_super[0], I_super[0], 'go', markersize=6, label="Initial Perturbation")
axs[1, 0].set_title(f"Supercritical Regime (Z = {Z_super}): Limit Cycle", fontsize=12, weight="bold")
axs[1, 0].set_xlabel("Excitatory Variable (E)", fontsize=11)
axs[1, 0].set_ylabel("Inhibitory Variable (I)", fontsize=11)
axs[1, 0].legend(loc="upper right")
axs[1, 0].grid(True, linestyle=":", alpha=0.5)

# Panel 4: Radius Evolution over Time
axs[1, 1].plot(time_vec, radius_super, color="darkorchid", lw=1.5, label="Trajectory Radius r(t)")
axs[1, 1].axhline(np.sqrt(Z_super), color="firebrick", linestyle="--", lw=2, label=r"Attractor Target $r^* = \sqrt{Z}$")
axs[1, 1].set_title("Transient Orbit Convergence to Limit Cycle", fontsize=12, weight="bold")
axs[1, 1].set_xlabel("Time (s)", fontsize=11)
axs[1, 1].set_ylabel("Radius r", fontsize=11)
axs[1, 1].legend(loc="lower right")
axs[1, 1].grid(True, linestyle=":", alpha=0.5)

plt.tight_layout()

# Export figure for GitHub project documentation
plt.savefig("assets/hopf_bifurcation_analysis.png", dpi=300, bbox_inches="tight")
plt.show()
