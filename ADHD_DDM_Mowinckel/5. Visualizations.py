# 5. EXPERIMENTAL SIMULATION AND CLINICAL VISUALIZATION

# Initialize mapper and simulation engine
mapper = BiologicalMapper()
model = BioDDM()

# Translate neurobiological states into latent DDM parameters
params_control = mapper.map_to_ddm(control_state)
params_impulsive = mapper.map_to_ddm(impulsive_state)
params_compensatory = mapper.map_to_ddm(compensatory_state)

# Execute cohort runs
control_rts, control_acc = gather_cohort_data(model, params_control)
impulsive_rts, impulsive_acc = gather_cohort_data(model, params_impulsive)
compensatory_rts, compensatory_acc = gather_cohort_data(
    model, params_compensatory
)

# Render publication-quality distribution comparisons
plt.figure(figsize=(10, 6))

plt.hist(
    control_rts,
    bins=50,
    density=True,
    alpha=0.45,
    color="steelblue",
    label=f"Control (Accuracy: {control_acc:.1%})",
)

plt.hist(
    impulsive_rts,
    bins=50,
    density=True,
    alpha=0.45,
    color="crimson",
    label=f"Impulsive ADHD (Accuracy: {impulsive_acc:.1%})",
)

plt.hist(
    compensatory_rts,
    bins=50,
    density=True,
    alpha=0.45,
    color="forestgreen",
    label=f"Compensatory ADHD (Accuracy: {compensatory_acc:.1%})",
)

# Reference markers for cohort distribution medians / means
plt.axvline(
    np.mean(control_rts),
    color="steelblue",
    linestyle="dashed",
    linewidth=1.8,
    label=f"Control Mean: {np.mean(control_rts):.3f}s",
)
plt.axvline(
    np.mean(impulsive_rts),
    color="crimson",
    linestyle="dashed",
    linewidth=1.8,
    label=f"Impulsive Mean: {np.mean(impulsive_rts):.3f}s",
)
plt.axvline(
    np.mean(compensatory_rts),
    color="forestgreen",
    linestyle="dashed",
    linewidth=1.8,
    label=f"Compensatory Mean: {np.mean(compensatory_rts):.3f}s",
)

plt.title(
    "Biologically Mapped Drift Diffusion Model: Reaction Time Profiles",
    fontsize=13,
)
plt.xlabel("Reaction Time (seconds)", fontsize=11)
plt.ylabel("Probability Density", fontsize=11)
plt.legend(frameon=True, facecolor="white", edgecolor="none")
plt.tight_layout()

# Export figure for documentation
plt.savefig("assets/rt_clinical_comparison.png", dpi=300)
plt.show()
