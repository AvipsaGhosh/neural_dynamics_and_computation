import numpy as np
from preprocessing import load_and_preprocess_crcns_v1v2
from cv_preprocessor import CrossValidatedPreprocessor
from semedo_baseline import SemedoBaselineEngine
from subspace_tracker import RobustDynamicSubspaceTracker
from multiplexed_analyzer import CorrectedMultiplexedAnalyzer
from subspace_null_validator import SubspaceNullValidator

def run_communication_subspace_pipeline(mat_file_path):
    print("=" * 60)
    print("STARTING COMMUNICATION SUBSPACE PIPELINE")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Load and Preprocess Dataset
    # ---------------------------------------------------------
    # Note: Adjust bin_size_ms and min_firing_rate as needed for your session
    X_tensor, Y_tensor = load_and_preprocess_crcns_v1v2(
        mat_file_path=mat_file_path, 
        bin_size_ms=100, 
        min_firing_rate=0.5
    )
    
    n_trials, n_timepoints, p_source = X_tensor.shape
    q_target = Y_tensor.shape[2]
    
    # Placeholder for trial conditions (e.g., stimulus orientation labels).
    # Replace this with your actual condition labels loaded from the session file.
    conditions = np.random.randint(0, 4, size=n_trials)

    # ---------------------------------------------------------
    # 2. Cross-Validation & Residualization Setup
    # ---------------------------------------------------------
    print("\n[Step 2] Initializing Cross-Validated Preprocessor...")
    cv_prep = CrossValidatedPreprocessor(X_tensor, Y_tensor, conditions, n_splits=5)
    splits = cv_prep.get_trial_splits()

    # ---------------------------------------------------------
    # 3. Baseline Reduced-Rank Regression (RRR) & Evaluation
    # ---------------------------------------------------------
    print("\n[Step 3] Running Cross-Validated Semedo RRR Baseline...")
    baseline_engine = SemedoBaselineEngine(l2_reg=1e-3)
    fold_r2_scores = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        # Process fold without data leakage
        X_train_res, Y_train_res, X_test_res, Y_test_res = cv_prep.preprocess_fold(train_idx, test_idx)
        
        # Flatten time and trial dimensions for standard regression fitting: (Trials * Time, Neurons)
        X_train_flat = X_train_res.reshape(-1, p_source)
        Y_train_flat = Y_train_res.reshape(-1, q_target)
        X_test_flat = X_test_res.reshape(-1, p_source)
        Y_test_flat = Y_test_res.reshape(-1, q_target)
        
        # Fit and evaluate model
        results = baseline_engine.fit_and_evaluate(
            X_train_flat, Y_train_flat, X_test_flat, Y_test_flat, rank=2
        )
        fold_r2_scores.append(results["test_r2"])
        print(f"  -> Fold {fold_idx + 1} Test R²: {results['test_r2']:.4f}")

    print(f"Mean Cross-Validated Test R²: {np.mean(fold_r2_scores):.4f}")

    # ---------------------------------------------------------
    # 4. Dynamic Subspace Tracking with Bootstrapping
    # ---------------------------------------------------------
    print("\n[Step 4] Tracking Subspace Dynamics & Uncertainty...")
    # Use full residualized representation or raw tensor across time windows
    tracker = RobustDynamicSubspaceTracker(X_tensor, Y_tensor, l2_reg=1e-3)
    dynamics_results = tracker.track_dynamics_with_bootstrapping(
        window_size=20, step_size=10, rank=2, n_bootstraps=20
    )
    print(f"  -> Tracked {len(dynamics_results['window_centers'])} temporal windows.")
    print(f"  -> Mean Rotation Angle across windows: {np.mean(dynamics_results['mean_rotation_angles_deg']):.2f}°")

    # ---------------------------------------------------------
    # 5. Multiplexed Subpopulation Analysis
    # ---------------------------------------------------------
    print("\n[Step 5] Analyzing Target Subpopulations...")
    # Flatten tensors for static subpopulation comparison
    X_flat_all = X_tensor.reshape(-1, p_source)
    Y_flat_all = Y_tensor.reshape(-1, q_target)
    
    analyzer = CorrectedMultiplexedAnalyzer(X_flat_all, Y_flat_all, Y_tensor)
    Y_sub1, Y_sub2 = analyzer.partition_by_raw_firing_rate()
    subspace_angles = analyzer.compute_subspace_overlap(Y_sub1, Y_sub2, rank=2)
    print(f"  -> Subpopulation Principal Angles: {subspace_angles}")

    # ---------------------------------------------------------
    # 6. Statistical Validation via Surrogate Nulls
    # ---------------------------------------------------------
    print("\n[Step 6] Running Subspace Null Validation...")
    validator = SubspaceNullValidator(X_tensor, Y_tensor, l2_reg=1e-3)
    null_distribution = validator.generate_shuffled_null_angles(
        window_size=20, step_size=10, rank=2, n_shuffles=20
    )
    
    observed_mean_angle = np.mean(dynamics_results["mean_rotation_angles_deg"])
    p_value = validator.evaluate_significance(observed_mean_angle, null_distribution)
    print(f"  -> Observed Mean Angle: {observed_mean_angle:.2f}°")
    print(f"  -> Surrogate Null p-value: {p_value:.4f}")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)

if __name__ == "__main__":
    # Example entry point execution
    sample_mat_path = "path_to_crcns_session.mat"
    # run_communication_subspace_pipeline(sample_mat_path)
