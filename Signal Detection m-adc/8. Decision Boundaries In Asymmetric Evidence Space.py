import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# ---------------------------------------------------------
# Step 1: Theoretical Boundary Comparison in Decision Space
# ---------------------------------------------------------
# Asymmetric sensitivities (e.g., attention cued strongly to Location 1)
d1 = 2.5
d2 = 1.0
c1 = 1.0
c2 = 1.0
pi1 = 0.5  # Equal prior probabilities
pi2 = 0.5

# Sensory evidence grid
psi1_vals = np.linspace(-1.0, 4.5, 300)
psi2_vals = np.linspace(-1.0, 4.5, 300)
P1, P2 = np.meshgrid(psi1_vals, psi2_vals)

# True Bayes-optimal boundary where Log-Likelihood Ratio = 0:
# d1*Psi1 - d2*Psi2 - 0.5*(d1^2 - d2^2) + ln(pi1/pi2) = 0
bayes_boundary = d1 * P1 - d2 * P2 - 0.5 * (d1**2 - d2**2) + np.log(pi1 / pi2)

# Standard heuristic linear boundary: Psi1 - Psi2 = c1 - c2
linear_boundary = (P1 - P2) - (c1 - c2)

# ---------------------------------------------------------
# Step 2: Simulate Task Performance Under Both Decision Rules
# ---------------------------------------------------------
np.random.seed(42)
N_sim_trials = 10000

# Present target randomly at Loc 1 or Loc 2
true_locations = np.random.choice([1, 2], size=N_sim_trials, p=[0.5, 0.5])

# Draw sensory evidence
psi1 = np.where(true_locations == 1, d1, 0.0) + np.random.normal(0, 1, N_sim_trials)
psi2 = np.where(true_locations == 2, d2, 0.0) + np.random.normal(0, 1, N_sim_trials)

# Strategy A: Bayes-Optimal Boundary
# Choose 1 if sensory evidence for 1 beats the threshold AND beats 2 under optimal rule
llr = d1 * psi1 - d2 * psi2 - 0.5 * (d1**2 - d2**2) + np.log(pi1 / pi2)
choices_optimal = np.where((psi1 > c1) & (llr > 0), 1,
                  np.where((psi2 > c2) & (llr <= 0), 2, 0))

# Strategy B: Standard Linear Heuristic Boundary
# Choose 1 if psi1 > c1 AND psi1 - psi2 > c1 - c2
linear_diff = (psi1 - psi2) - (c1 - c2)
choices_linear = np.where((psi1 > c1) & (linear_diff > 0), 1,
                 np.where((psi2 > c2) & (linear_diff <= 0), 2, 0))

acc_optimal = np.mean(choices_optimal == true_locations) * 100
acc_linear = np.mean(choices_linear == true_locations) * 100

print(r"--- Performance Under Asymmetric Sensitivities (d1 = 2.5, d2 = 1.0) ---")
print(f"Bayes-Optimal Decision Accuracy: {acc_optimal:.2f}%")
print(f"Linear Heuristic Accuracy:       {acc_linear:.2f}%")
print(f"Efficiency Loss from Heuristic:   {acc_optimal - acc_linear:.2f} percentage points")

# ---------------------------------------------------------
# Step 3: Visualization
# ---------------------------------------------------------
plt.figure(figsize=(7.5, 6.5))

# Plot target evidence distributions (contours)
cov = [[1, 0], [0, 1]]
dist1 = multivariate_normal([d1, 0], cov)
dist2 = multivariate_normal([0, d2], cov)
pos = np.dstack((P1, P2))
plt.contour(P1, P2, dist1.pdf(pos), levels=3, colors='blue', alpha=0.35)
plt.contour(P1, P2, dist2.pdf(pos), levels=3, colors='red', alpha=0.35)

# Plot decision boundaries
plt.contour(P1, P2, bayes_boundary, levels=[0], colors='darkgreen', linewidths=2.5)
plt.plot([], [], color='darkgreen', linewidth=2.5, label='Curved Bayes-Optimal Boundary')

plt.contour(P1, P2, linear_boundary, levels=[0], colors='purple', linestyles='--', linewidths=2.0)
plt.plot([], [], color='purple', linestyle='--', linewidth=2.0, label='Heuristic Linear Boundary')

# Criterion thresholds
plt.axvline(c1, color='gray', linestyle=':', label='Criterion c1')
plt.axhline(c2, color='gray', linestyle=':', label='Criterion c2')

plt.scatter([d1], [0], color='blue', s=80, zorder=5, label='Mean Evidence: Stim 1 (d1=2.5)')
plt.scatter([0], [d2], color='red', s=80, zorder=5, label='Mean Evidence: Stim 2 (d2=1.0)')

plt.title(r"Decision Boundaries in Asymmetric Evidence Space ($d_1 \neq d_2$)", fontsize=12)
plt.xlabel(r"Sensory Evidence Channel 1 ($\Psi_1$)", fontsize=11)
plt.ylabel(r"Sensory Evidence Channel 2 ($\Psi_2$)", fontsize=11)
plt.xlim(-1.0, 4.0)
plt.ylim(-1.0, 3.5)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc='lower right', fontsize=9.5)
plt.tight_layout()
plt.show()
