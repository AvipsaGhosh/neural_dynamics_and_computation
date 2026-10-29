import numpy as np

class SemedoBaselineEngine:
    """
    Implements the baseline Reduced-Rank Regression (RRR) engine using stable 
    linear solvers and evaluates performance on held-out trial folds.
    """
    def __init__(self, l2_reg=1e-3):
        self.l2_reg = l2_reg

    def fit_and_evaluate(self, X_train_res, Y_train_res, X_test_res, Y_test_res, rank=2):
        """
        Fits ridge regression and RRR on training data, then evaluates out-of-sample 
        predictive performance (R-squared) on test data.
        """
        p_source = X_train_res.shape[1]
        
        XtX = X_train_res.T @ X_train_res
        ridge_matrix = self.l2_reg * np.eye(p_source)
        
        target_term = X_train_res.T @ Y_train_res
        B_ols = np.linalg.solve(XtX + ridge_matrix, target_term)
        
        Y_ols_train = X_train_res @ B_ols
        _, _, Vt = np.linalg.svd(Y_ols_train, full_matrices=False)
        V = Vt.T
        
        V_m = V[:, :rank]
        B_rrr = B_ols @ V_m @ V_m.T
        B_bar = B_ols @ V_m
        
        Y_pred_test = X_test_res @ B_rrr
        
        residual_variance = np.var(Y_test_res - Y_pred_test, axis=0)
        total_variance = np.var(Y_test_res, axis=0)
        r2_per_neuron = 1.0 - (residual_variance / (total_variance + 1e-10))
        mean_test_r2 = np.mean(r2_per_neuron)
        
        return {
            "B_ols": B_ols,
            "B_rrr": B_rrr,
            "B_bar": B_bar,
            "test_r2": mean_test_r2,
            "r2_per_neuron": r2_per_neuron
        }
