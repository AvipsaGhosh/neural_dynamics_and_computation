import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import minimize
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# 1. Biological Population Extraction (The Allen Cortical Pool)
# -------------------------------------------------------------
# response_array shape: (n_ori=8, n_tf=5, n_cells, stats)
all_cells_matrix = response_array[:, :, :, 0].copy()  # Shape: (8, 5, n_cells)
n_cells = all_cells_matrix.shape[2]

# Target cell responses across all 40 conditions
y_target_raw = all_cells_matrix[:, :, cell_loc]
baseline_target = np.nanmin(y_target_raw)
y_target_clean = np.maximum(0.0, y_target_raw - baseline_target)

# Compute the REAL Cortical Normalization Pool:
# Average response across all OTHER recorded cells in mouse V1 for each condition
other_cells_mask = np.ones(n_cells, dtype=bool)
other_cells_mask[cell_loc] = False

# Population drive matrix: (8, 5)
empirical_pool_matrix = np.nanmean(all_cells_matrix[:, :, other_cells_mask], axis=2)
# Baseline correct the pool
empirical_pool_matrix = np.maximum(0.0, empirical_pool_matrix - np.nanmin(empirical_pool_matrix))

# -------------------------------------------------------------
# 2. Extract Conditions with Identical Feature Representation
# -------------------------------------------------------------
# We extract target drive D_target using the cell's empirically fitted preference
def circular_ori_drive(theta, pref_theta, width=np.deg2rad(25.0)):
    d_theta = np.abs(np.angle(np.exp(1j * 2 * (theta - pref_theta)))) / 2.0
    return np.exp(-0.5 * (d_theta / width) ** 2)

def log_tf_drive(tf, pref_tf=4.0, width=1.2):
    return np.exp(-0.5 * ((np.log2(tf) - np.log2(pref_tf)) / width) ** 2)

d_target_list = []
d_pool_list = []
y_list = []

for ori_i, ori_rad in enumerate(angles_rad):
    for tf_i, tf in enumerate(dg_data.tfvals):
        r_target = y_target_clean[ori_i, tf_i]
        r_pool = empirical_pool_matrix[ori_i, tf_i]
        
        if np.isfinite(r_target) and np.isfinite(r_pool):
            # Target linear receptive field drive
            dt = circular_ori_drive(ori_rad, theta_pref_fit) * log_tf_drive(tf, best_tf)
            
            d_target_list.append(dt)
            d_pool_list.append(r_pool)
            y_list.append(r_target)

D_t = np.array(d_target_list, dtype=np.float32)
D_p = np.array(d_pool_list, dtype=np.float32)
y = np.array(y_list, dtype=np.float32)

# Normalize D_p to [0, 1] range for numerical stability
D_p = D_p / np.max(D_p)

# Pair the exact same 2 inputs: [D_target, D_pool] for BOTH models
X_equalized = np.column_stack([D_t, D_p])

idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.25, random_state=42)

# -------------------------------------------------------------
# 3. Model 1: Empirical Divisive Normalization (3 parameters)
# -------------------------------------------------------------
# R = R_max * (D_t^p) / (sigma^p + w * D_pool^p)
def dn_pool_predict(X_subset, r_max, sigma, w, p=2.0):
    dt = X_subset[:, 0]
    dp = X_subset[:, 1]
    num = dt ** p
    denom = (sigma ** p) + w * (dp ** p)
    return r_max * (num / denom)

def dn_cost(params):
    r_max, sigma, w = params
    preds = dn_pool_predict(X_equalized[idx_tr], r_max, sigma, w)
    return np.mean((preds - y[idx_tr]) ** 2)

opt = minimize(dn_cost, [np.max(y), 0.5, 0.5], bounds=[(1.0, 150.0), (0.01, 5.0), (0.001, 10.0)])
rmax_opt, sigma_opt, w_opt = opt.x

# -------------------------------------------------------------
# 4. Model 2: Flexible Backprop MLP (Equal 2D Input: [D_t, D_p])
# -------------------------------------------------------------
class FairMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.net(x)

mlp_equal = FairMLP()
optimizer = optim.Adam(mlp_equal.parameters(), lr=0.01, weight_decay=1e-3)
criterion = nn.MSELoss()

X_tr_t = torch.tensor(X_equalized[idx_tr], dtype=torch.float32)
y_tr_t = torch.tensor(y[idx_tr], dtype=torch.float32).unsqueeze(1)
X_te_t = torch.tensor(X_equalized[idx_te], dtype=torch.float32)

for _ in range(400):
    optimizer.zero_grad()
    loss = criterion(mlp_equal(X_tr_t), y_tr_t)
    loss.backward()
    optimizer.step()

# -------------------------------------------------------------
# 5. Evaluation on Held-Out Cortical Data
# -------------------------------------------------------------
y_test = y[idx_te]
y_pred_dn = dn_pool_predict(X_equalized[idx_te], rmax_opt, sigma_opt, w_opt)

mlp_equal.eval()
with torch.no_grad():
    y_pred_mlp = mlp_equal(X_te_t).squeeze().numpy()

r_dn, _ = pearsonr(y_test, y_pred_dn)
r_mlp, _ = pearsonr(y_test, y_pred_mlp)

r2_dn = r2_score(y_test, y_pred_dn)
r2_mlp = r2_score(y_test, y_pred_mlp)

print("--- Scientifically Equalized Benchmark on Real Mouse V1 ---")
print(f"Divisive Normalization (3 params) -> Pearson r: {r_dn:.3f} | R²: {r2_dn:.3f}")
print(f"Backprop MLP (625 params)          -> Pearson r: {r_mlp:.3f} | R²: {r2_mlp:.3f}")
