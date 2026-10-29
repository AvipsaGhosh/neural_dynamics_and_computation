import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# 1. Synthetic stimulus generator: oriented sinusoidal gratings
def create_grating(size=31, theta=0.0, spatial_freq=0.1, contrast=1.0):
    """Generates an image patch of a sinusoidal grating at angle theta (radians)."""
    half = size // 2
    y, x = np.mgrid[-half:half+1, -half:half+1]
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    return contrast * np.cos(2 * np.pi * spatial_freq * x_theta)

# 2. Gabor filter representing the primary receptive field
def make_gabor(size=31, theta=0.0, sigma=5.0, lambd=10.0, gamma=0.5):
    half = size // 2
    y, x = np.mgrid[-half:half+1, -half:half+1]
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)
    gaussian = np.exp(-0.5 * (x_theta**2 + (gamma * y_theta)**2) / (sigma**2))
    carrier = np.cos(2 * np.pi * x_theta / lambd)
    kernel = gaussian * carrier
    return kernel - kernel.mean()

# Build orientation pool (8 orientations covering the circle)
pref_angle_rad = theta_pref_fit
pool_angles = np.linspace(0, np.pi, 8, endpoint=False)
filter_bank = [make_gabor(theta=angle) for angle in pool_angles]
target_filter = make_gabor(theta=pref_angle_rad)

# 3. Model definitions
def compute_linear_drive(stimulus, kernel):
    return np.maximum(0.0, np.sum(stimulus * kernel))

def divisive_normalization_model(stimulus, r_max, sigma_c, p=2.0):
    """
    R = R_max * (D_target^p) / (sigma^p + sum(D_pool^p))
    """
    # Drive of target cell
    target_drive = compute_linear_drive(stimulus, target_filter) ** p
    
    # Drive of the normalization pool (all orientations combined)
    pool_drive = sum(compute_linear_drive(stimulus, f) ** p for f in filter_bank)
    
    denominator = (sigma_c ** p) + pool_drive
    return r_max * (target_drive / denominator)

def linear_nonlinear_model(stimulus, r_max, gain=1.0, p=2.0):
    """Classic LN model without normalization: R = gain * [D_target]^p"""
    target_drive = compute_linear_drive(stimulus, target_filter)
    return np.minimum(r_max, gain * (target_drive ** p))

# 4. Simulation: Contrast Response Curves
contrasts = np.linspace(0.05, 1.0, 20)
r_max_est = np.max(empirical_responses)
sigma_val = 15.0  # Semi-saturation constant

# Test A: Preferred grating alone at varying contrasts
dn_contrast_curve = []
ln_contrast_curve = []

for c in contrasts:
    stim = create_grating(theta=pref_angle_rad, contrast=c)
    dn_contrast_curve.append(divisive_normalization_model(stim, r_max=r_max_est, sigma_c=sigma_val))
    ln_contrast_curve.append(linear_nonlinear_model(stim, r_max=r_max_est, gain=0.015))

# Test B: Cross-Orientation Suppression (Plaid)
# Preferred grating (contrast 0.5) + orthogonal mask (contrast varied 0.0 -> 0.8)
mask_angle = pref_angle_rad + (np.pi / 2)
mask_contrasts = np.linspace(0.0, 0.8, 20)

dn_plaid_curve = []
ln_plaid_curve = []

base_stim = create_grating(theta=pref_angle_rad, contrast=0.5)

for mc in mask_contrasts:
    mask = create_grating(theta=mask_angle, contrast=mc)
    plaid = base_stim + mask
    dn_plaid_curve.append(divisive_normalization_model(plaid, r_max=r_max_est, sigma_c=sigma_val))
    ln_plaid_curve.append(linear_nonlinear_model(plaid, r_max=r_max_est, gain=0.015))

# 5. Plotting the results
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Plot 1: Contrast Saturation
axes[0].plot(contrasts, dn_contrast_curve, color="navy", lw=2, label="Divisive Normalization")
axes[0].plot(contrasts, ln_contrast_curve, color="orange", lw=2, linestyle="--", label="Standard LN Model")
axes[0].set_title("Contrast Response (Saturation)")
axes[0].set_xlabel("Stimulus Contrast")
axes[0].set_ylabel("Predicted Response (dF/F)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Cross-Orientation Suppression
axes[1].plot(mask_contrasts, dn_plaid_curve, color="navy", lw=2, label="Divisive Normalization")
axes[1].plot(mask_contrasts, ln_plaid_curve, color="orange", lw=2, linestyle="--", label="Standard LN Model")
axes[1].set_title("Cross-Orientation Masking (Plaid Stimulus)")
axes[1].set_xlabel("Orthogonal Mask Contrast")
axes[1].set_ylabel("Predicted Response to Preferred Stimulus")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
