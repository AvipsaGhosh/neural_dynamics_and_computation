import numpy as np

class SubspaceNullValidator:
    """
    Generates surrogate null distributions via trial shuffling to test statistical 
    significance of principal angles and communication subspace geometry.
    """
    def __init__(self, X_tensor, Y_tensor, l2_reg=1e-3):
        self.X_tensor = X_tensor
        self.Y_tensor = Y_tensor
        self.l2_reg = l2_reg

    def generate_shuffled_null_angles(self, window_size=20, step_size=10, rank=2, n_shuffles=100):
        print(f"Generating shuffled null distribution ({n_shuffles} iterations)...")
        n_trials, n_timepoints, p_source = self.X_tensor.shape
        q_target = self.Y_tensor.shape[2]
        
        null_max_angles = []
        
        for _ in range(n_shuffles):
            shuffled_indices = np.random.permutation(n_trials)
            Y_shuffled = self.Y_tensor[shuffled_indices, :, :]
            
            subspace_bases = []
            for start_t in range(0, n_timepoints - window_size + 1, step_size):
                end_t = start_t + window_size
                X_win = self.X_tensor[:, start_t:end_t, :].reshape(-1, p_source)
                Y_win = Y_shuffled[:, start_t:end_t, :].reshape(-1, q_target)
                
                XtX = X_win.T @ X_win
                ridge = self.l2_reg * np.eye(p_source)
                B_ols = np.linalg.solve(XtX + ridge, X_win.T @ Y_win)
                _, _, Vt = np.linalg.svd(X_win @ B_ols, full_matrices=False)
                
                B_bar = B_ols @ Vt[:rank, :].T
                Q, _ = np.linalg.qr(B_bar)
                subspace_bases.append(Q)
                
            run_angles = []
            for i in range(len(subspace_bases) - 1):
                overlap = subspace_bases[i].T @ subspace_bases[i+1]
                sv = np.linalg.svd(overlap, compute_uv=False)
                run_angles.append(np.max(np.degrees(np.arccos(np.clip(sv, -1.0, 1.0)))))
                
            null_max_angles.append(np.mean(run_angles))
            
        return np.array(null_max_angles)

    def evaluate_significance(self, observed_mean_angle, null_distribution):
        p_val = np.sum(null_distribution <= observed_mean_angle) / len(null_distribution)
        return p_val
