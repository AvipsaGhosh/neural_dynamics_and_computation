import numpy as np
from sklearn.model_selection import KFold

class CrossValidatedPreprocessor:
    """
    Handles trial-wise cross-validation and condition-specific PSTH residualization 
    to prevent information leakage during model training and evaluation.
    """
    def __init__(self, X_trials, Y_trials, conditions, n_splits=10):
        self.X = X_trials
        self.Y = Y_trials
        self.conditions = np.array(conditions)
        self.n_splits = n_splits

    def get_trial_splits(self):
        """
        Generates trial indices for cross-validation, ensuring entire trials 
        (and all their timepoints) remain together in either train or test sets.
        """
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        return list(kf.split(self.X))

    def preprocess_fold(self, train_idx, test_idx):
        """
        Computes condition-specific PSTHs exclusively on training trials, 
        then residualizes both training and test trials to prevent information leakage.
        """
        X_train_raw, X_test_raw = self.X[train_idx], self.X[test_idx]
        Y_train_raw, Y_test_raw = self.Y[train_idx], self.Y[test_idx]
        
        cond_train = self.conditions[train_idx]
        cond_test = self.conditions[test_idx]
        
        unique_conditions = np.unique(self.conditions)
        
        X_train_res = np.empty_like(X_train_raw)
        X_test_res = np.empty_like(X_test_raw)
        Y_train_res = np.empty_like(Y_train_raw)
        Y_test_res = np.empty_like(Y_test_raw)
        
        for cond in unique_conditions:
            train_cond_mask = (cond_train == cond)
            
            if np.sum(train_cond_mask) > 0:
                psth_v1_cond = np.mean(X_train_raw[train_cond_mask], axis=0)
                psth_v2_cond = np.mean(Y_train_raw[train_cond_mask], axis=0)
            else:
                psth_v1_cond = np.mean(X_train_raw, axis=0)
                psth_v2_cond = np.mean(Y_train_raw, axis=0)
            
            train_target_mask = (cond_train == cond)
            if np.sum(train_target_mask) > 0:
                X_train_res[train_target_mask] = X_train_raw[train_target_mask] - psth_v1_cond[np.newaxis, :, :]
                Y_train_res[train_target_mask] = Y_train_raw[train_target_mask] - psth_v2_cond[np.newaxis, :, :]
                
            test_target_mask = (cond_test == cond)
            if np.sum(test_target_mask) > 0:
                X_test_res[test_target_mask] = X_test_raw[test_target_mask] - psth_v1_cond[np.newaxis, :, :]
                Y_test_res[test_target_mask] = Y_test_raw[test_target_mask] - psth_v2_cond[np.newaxis, :, :]
                
        return X_train_res, Y_train_res, X_test_res, Y_test_res
