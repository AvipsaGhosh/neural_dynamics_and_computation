import numpy as np

class CorrectedMultiplexedAnalyzer:
    """
    Partitions target populations using unresidualized spike metrics or biological 
    metadata, then measures subspace overlap via principal angles.
    """
    def __init__(self, X_flat, Y_flat, Y_raw_tensor):
        self.X = X_flat
        self.Y_res = Y_flat
        self.raw_mean_fr = np.mean(Y_raw_tensor, axis=(0, 1))
        self.p_source = X_flat.shape[1]
        self.q_target = Y_flat.shape[1]

    def partition_by_raw_firing_rate(self):
        median_fr = np.median(self.raw_mean_fr)
        group1_idx = self.raw_mean_fr <= median_fr
        group2_idx = self.raw_mean_fr > median_fr
        
        Y_sub1 = self.Y_res[:, group1_idx]
        Y_sub2 = self.Y_res[:, group2_idx]
        return Y_sub1, Y_sub2

    def compute_subspace_overlap(self, Y_sub1, Y_sub2, rank=2, l2_reg=1e-3):
        XtX = self.X.T @ self.X
        ridge_matrix = l2_reg * np.eye(self.p_source)
        inv_XtX = np.linalg.solve(XtX + ridge_matrix, np.eye(self.p_source))
        
        B_ols_1 = inv_XtX @ (self.X.T @ Y_sub1)
        _, _, Vt1 = np.linalg.svd(self.X @ B_ols_1, full_matrices=False)
        B1 = B_ols_1 @ Vt1[:rank, :].T
        Q1, _ = np.linalg.qr(B1)
        
        B_ols_2 = inv_XtX @ (self.X.T @ Y_sub2)
        _, _, Vt2 = np.linalg.svd(self.X @ B_ols_2, full_matrices=False)
        B2 = B_ols_2 @ Vt2[:rank, :].T
        Q2, _ = np.linalg.qr(B2)
        
        cross_overlap = Q1.T @ Q2
        singular_values = np.linalg.svd(cross_overlap, compute_uv=False)
        angles_deg = np.degrees(np.arccos(np.clip(singular_values, -1.0, 1.0)))
        
        return angles_deg
