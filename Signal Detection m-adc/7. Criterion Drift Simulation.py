import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm

# ---------------------------------------------------------
# Step 0: Vectorized Probability Engine & Likelihood Function
# ---------------------------------------------------------
E_GRID = np.linspace(-6.0, 6.0, 150)
DE = E_GRID[1] - E_GRID[0]
PDF_GRID = norm.pdf(E_GRID)

def get_response_probabilities_vec(d1, d2, c1, c2, x1, x2):
    low_1 = c1 - d1 * x1
    mask_1 = E_GRID >= low_1
    val_1 = norm.cdf(E_GRID + d1 * x1 - d2 * x2 - c1 + c2) * PDF_GRID
    p1 = np.sum(val_1[mask_1]) * DE

    low_2 = c2 - d2 * x2
    mask_2 = E_GRID >= low_2
    val_2 = norm.cdf(E_GRID + d2 * x2 - d1 * x1 - c2 + c1) * PDF_GRID
    p2 = np.sum(val_2[mask_2]) * DE

    p0 = norm.cdf(c1 - d1 * x1) * norm.cdf(c2 - d2 * x2)
    probs = np.array([p0, p1, p2])
    probs = np.clip(probs, 1e-12, 1.0)
    return probs / np.sum(probs)

def get_stimulus_response_matrix_vec(d1, d2, c1, c2):
    p_catch = get_response_probabilities_vec(d1, d2, c1, c2, 0, 0)
    p_stim1 = get_response_probabilities_vec(d1, d2, c1, c2, 1, 0)
    p_stim2 = get_response_probabilities_vec(d1, d2, c1, c2, 0, 1)
    return np.vstack([p_catch, p_stim1, p_stim2])

def nll_2adc(params, counts):
    d, c1, c2 = params
    probs = get_stimulus_response_matrix_vec(d, d, c1, c2)
    return -np.sum(counts * np.log(probs))


# ---------------------------------------------------------
# Step 1: Simulate Experiment with Trial-by-Trial Criterion Drift
# ---------------------------------------------------------
def simulate_drift_experiment(d_true=2.0, c1_mean=0.8, c2_mean=1.2, 
                              sigma_c=0.0, n_trials_per_stim=500, seed=42):
    np.random.seed(seed)
    counts = np.zeros((3, 3), dtype=int)
    stim_conditions = np.repeat([0, 1, 2], n_trials_per_stim)
    
    for stim in stim_conditions:
        c1_t = np.random.normal(c1_mean, sigma_c)
        c2_t = np.random.normal(c2_mean, sigma_c)
        
        x1 = 1 if stim == 1 else 0
        x2 = 1 if stim == 2 else 0
        
        psi1 = d_true * x1 + np.random.normal(0, 1)
        psi2 = d_true * x2 + np.random.normal(0, 1)
        
        if psi1 > c1_t and psi1 > psi2:
            resp = 1
        elif psi2 > c2_t and psi2 > psi1:
            resp = 2
        else:
            resp = 0
            
        counts[stim, resp] += 1
        
    return counts


# ---------------------------------------------------------
# Step 2: Test Degradation Across Varying Drift Levels (sigma_c)
# ---------------------------------------------------------
sigma_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
recovered_d_values = []
d_ground_truth = 2.0

print("Testing sensitivity degradation under criterion drift...")

for s in sigma_values:
    drift_counts = simulate_drift_experiment(
        d_true=d_ground_truth,
        c1_mean=0.8,
        c2_mean=1.2,
        sigma_c=s,
        n_trials_per_stim=1000,
        seed=100
    )
    
    res = minimize(
        nll_2adc,
        x0=[1.8, 1.0, 1.0],
        args=(drift_counts,),
        method="L-BFGS-B",
        bounds=[(0.01, 5.0), (-3.0, 3.0), (-3.0, 3.0)]
    )
    
    recovered_d = res.x[0]
    recovered_d_values.append(recovered_d)
    print(f"Drift SD (sigma_c) = {s:3.1f} | Recovered d = {recovered_d:.3f} (True d = {d_ground_truth:.2f})")


# ---------------------------------------------------------
# Step 3: Plot Result
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(sigma_values, recovered_d_values, marker='o', linewidth=2.5, color='darkorange', label='Static Model Fit')
plt.axhline(d_ground_truth, color='black', linestyle='--', linewidth=1.5, label='True Perceptual Sensitivity (d)')

plt.title("Impact of Unmodeled Criterion Drift on Sensitivity Estimation", fontsize=13)
plt.xlabel("Criterion Instability / Drift SD ($\sigma_c$)", fontsize=11)
plt.ylabel("Estimated Perceptual Sensitivity ($d$)", fontsize=11)
plt.ylim(0.5, 2.5)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()
