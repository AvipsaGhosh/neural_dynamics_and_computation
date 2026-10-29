# ---------------------------------------------------------
# Fast, guarded response probability function
# ---------------------------------------------------------
def get_response_probabilities_fast(d1, d2, c1, c2, x1, x2):
    # Using a practical upper bound of 8.0 prevents quad from hanging in tails
    integrand_1 = lambda e: norm.cdf(e + d1*x1 - d2*x2 - c1 + c2) * norm.pdf(e)
    lower_1 = min(c1 - d1*x1, 7.9)
    p1, _ = quad(integrand_1, lower_1, 8.0, limit=50)
    
    integrand_2 = lambda e: norm.cdf(e + d2*x2 - d1*x1 - c2 + c1) * norm.pdf(e)
    lower_2 = min(c2 - d2*x2, 7.9)
    p2, _ = quad(integrand_2, lower_2, 8.0, limit=50)
    
    p0 = norm.cdf(c1 - d1*x1) * norm.cdf(c2 - d2*x2)
    
    probs = np.array([p0, p1, p2])
    probs = np.clip(probs, 1e-12, 1.0)
    return probs / np.sum(probs)

def get_stimulus_response_matrix_fast(d1, d2, c1, c2):
    p_catch = get_response_probabilities_fast(d1, d2, c1, c2, 0, 0)
    p_stim1 = get_response_probabilities_fast(d1, d2, c1, c2, 1, 0)
    p_stim2 = get_response_probabilities_fast(d1, d2, c1, c2, 0, 1)
    return np.vstack([p_catch, p_stim1, p_stim2])

def nll_detection_fast(params, counts):
    d, c1, c2 = params
    probs = get_stimulus_response_matrix_fast(d, d, c1, c2)
    return -np.sum(counts * np.log(probs))

def nll_unbiased_fast(params, counts):
    d, c = params
    probs = get_stimulus_response_matrix_fast(d, d, c, c)
    return -np.sum(counts * np.log(probs))

# ---------------------------------------------------------
# Run Model Comparison Across All 18 Observers
# ---------------------------------------------------------
N_trials = 600
comparison_records = []
bounds_2adc = [(0.01, 5.0), (-3.0, 3.0), (-3.0, 3.0)]
bounds_unb  = [(0.01, 5.0), (-3.0, 3.0)]

print("Fitting all 18 observers...")
for obs_name, counts in observer_data.items():
    # 1. 2-ADC fit (3 parameters)
    res_2adc = minimize(
        fun=nll_detection_fast,
        x0=[1.8, 1.0, 1.0],
        args=(counts,),
        method="L-BFGS-B",
        bounds=bounds_2adc
    )
    nll_2adc = res_2adc.fun
    k_2adc = 3
    bic_2adc = k_2adc * np.log(N_trials) + 2 * nll_2adc
    
    # 2. Unbiased fit (2 parameters)
    res_unb = minimize(
        fun=nll_unbiased_fast,
        x0=[1.8, 1.0],
        args=(counts,),
        method="L-BFGS-B",
        bounds=bounds_unb
    )
    nll_unb = res_unb.fun
    k_unb = 2
    bic_unb = k_unb * np.log(N_trials) + 2 * nll_unb
    
    comparison_records.append({
        "obs": obs_name,
        "d": res_2adc.x[0],
        "bias": res_2adc.x[1] - res_2adc.x[2],
        "delta_bic": bic_2adc - bic_unb
    })

print("Done! Plotting results...")

# ---------------------------------------------------------
# Visualizations
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

obs_ids = np.arange(1, len(comparison_records) + 1)
biases = [r["bias"] for r in comparison_records]
delta_bics = [r["delta_bic"] for r in comparison_records]

# Plot 1: Interval Biases
axes[0].axhline(0, color="black", linestyle="--", alpha=0.7)
axes[0].stem(obs_ids, biases)
axes[0].set_xlabel("Observer ID")
axes[0].set_ylabel("Interval Bias (c1 - c2)")
axes[0].set_title("Systematic Choice Bias Across Observers")
axes[0].grid(axis="y", linestyle="--", alpha=0.5)

# Plot 2: Model Selection (Delta BIC)
axes[1].axhline(0, color="black", linestyle="--")
axes[1].bar(obs_ids, delta_bics, color=["seagreen" if b < 0 else "crimson" for b in delta_bics])
axes[1].set_xlabel("Observer ID")
axes[1].set_ylabel("Delta BIC (2-ADC minus Unbiased)")
axes[1].set_title("Model Preference (Negative favors 2-ADC)")
axes[1].grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()

print("Mean Interval Bias across observers:", np.mean(biases))
print("Observers where 2-ADC outperforms Unbiased model:", sum(1 for b in delta_bics if b < 0))
