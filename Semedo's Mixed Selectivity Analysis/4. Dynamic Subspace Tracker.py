import numpy as np

class RobustDynamicSubspaceTracker:
    """
    Tracks rolling time-window communication subspace rotations with trial-level 
    bootstrapping for uncertainty quantification and rank-deficiency safeguards.
    """
    def __init__(self, X_tensor, Y_tensor, l2_reg=1e-3):
        self.X_tensor = X_tensor
        self.Y_tensor = Y_tensor
        self.l2_reg = l2_reg
        self.n_trials, self.n_timepoints, self.p_source = X_tensor.shape
        _, _, self.q_target = Y_tensor.shape

    def _fit_window_basis(self, X_win, Y_win, rank):
        XtX = X_win.T @ X_win
        ridge_matrix = self.l2_reg * np.eye(self.p_source)
        
        target_term = X_win.T @ Y_win
        B_ols = np.linalg.solve(XtX + ridge_matrix, target_term)
        
        Y_ols = X_win @ B_ols
        _, S, Vt = np.linalg.svd(Y_ols, full_matrices=False)
        
        effective_rank = min(rank, np.sum(S > 1e-5))
        V_reduced = Vt[:effective_rank, :].T
        
        B_bar = B_ols @ V_reduced
        Q, _ = np.linalg.qr(B_bar)
        return Q

    def compute_window_bases(self, window_size=20, step_size=10, rank=2, trial_indices=None):
        if trial_indices is None:
            trial_indices = np.arange(self.n_trials)
            
        subspace_bases = []
        window_centers = []
        
        for start_t in range(0, self.n_timepoints - window_size + 1, step_size):
            end_t = start_t + window_size
            
            x_slice = self.X_tensor[trial_indices, start_t:end_t, :]
            y_slice = self.Y_tensor[trial_indices, start_t:end_t, :]
            
            X_win = x_slice.reshape(-1, self.p_source)
            Y_win = y_slice.reshape(-1, self.q_target)
            
            Q_t = self._fit_window_basis(X_win, Y_win, rank=rank)
            subspace_bases.append(Q_t)
            window_centers.append((start_t + end_t) / 2.0)
            
        return np.array(window_centers), subspace_bases

    def _calculate_rotation_angles(self, bases):
        rotation_angles = []
        for i in range(len(bases) - 1):
            B1, B2 = bases[i], bases[i+1]
            cross_overlap = B1.T @ B2
            singular_values = np.linalg.svd(cross_overlap, compute_uv=False)
            clipped_sv = np.clip(singular_values, -1.0, 1.0)
            angles_deg = np.degrees(np.arccos(clipped_sv))
            rotation_angles.append(np.max(angles_deg))
        return np.array(rotation_angles)

    def track_dynamics_with_bootstrapping(self, window_size=20, step_size=10, rank=2, n_bootstraps=50):
        print(f"Running bootstrap dynamic tracking ({n_bootstraps} resamples)...")
        bootstrapped_angles = []
        centers = None
        
        for b in range(n_bootstraps):
            resampled_trials = np.random.choice(self.n_trials, size=self.n_trials, replace=True)
            centers, bases = self.compute_window_bases(window_size, step_size, rank, trial_indices=resampled_trials)
            angles = self._calculate_rotation_angles(bases)
            bootstrapped_angles.append(angles)
            
        bootstrapped_angles = np.array(bootstrapped_angles)
        
        mean_angles = np.mean(bootstrapped_angles, axis=0)
        ci_lower = np.percentile(bootstrapped_angles, 2.5, axis=0)
        ci_upper = np.percentile(bootstrapped_angles, 97.5, axis=0)
        
        return {
            "window_centers": centers[1:],
            "mean_rotation_angles_deg": mean_angles,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "raw_bootstraps": bootstrapped_angles
        }
