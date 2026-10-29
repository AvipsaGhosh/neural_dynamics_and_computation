import numpy as np
import scipy.io as sio

def load_and_preprocess_crcns_v1v2(mat_file_path, bin_size_ms=100, min_firing_rate=0.5):
    """
    Loads and preprocesses the CRCNS v1v2-1 dataset into a 3D tensor structure 
    (Trials, Timepoints, Neurons) and computes trial-to-trial PSTH residuals.
    
    Parameters:
    - mat_file_path: Path to the downloaded CRCNS .mat session file.
    - bin_size_ms: Spike counting bin size in milliseconds (default: 100ms as per Semedo et al., 2019).
    - min_firing_rate: Minimum average firing rate threshold to keep a neuron.
    
    Returns:
    - X_tensor: 3D numpy array for Source V1 (Trials, Timepoints, Neurons)
    - Y_tensor: 3D numpy array for Target V2 (Trials, Timepoints, Neurons)
    """
    print(f"Loading dataset from {mat_file_path}...")
    mat_data = sio.loadmat(mat_file_path, squeeze_me=True, struct_as_dict=False)
    
    raw_v1_trials = mat_data.get('v1_trials', None) 
    raw_v2_trials = mat_data.get('v2_trials', None)
    
    if raw_v1_trials is None or raw_v2_trials is None:
        raise ValueError("Could not locate expected trial keys in the .mat file. Check structure via sio.whosmat().")

    # 1. Filter out neurons with mean firing rate below the threshold across all trials
    v1_mean_fr = np.mean(raw_v1_trials, axis=(0, 1)) / (bin_size_ms / 1000.0)
    v2_mean_fr = np.mean(raw_v2_trials, axis=(0, 1)) / (bin_size_ms / 1000.0)
    
    v1_valid_idx = v1_mean_fr >= min_firing_rate
    v2_valid_idx = v2_mean_fr >= min_firing_rate
    
    X_filtered = raw_v1_trials[:, :, v1_valid_idx]
    Y_filtered = raw_v2_trials[:, :, v2_valid_idx]
    
    print(f"Retained V1 neurons: {np.sum(v1_valid_idx)} / {len(v1_valid_idx)}")
    print(f"Retained V2 neurons: {np.sum(v2_valid_idx)} / {len(v2_valid_idx)}")

    # 2. Compute and subtract the Peri-Stimulus Time Histogram (PSTH) per condition
    psth_v1 = np.mean(X_filtered, axis=0) 
    psth_v2 = np.mean(Y_filtered, axis=0)
    
    X_residuals = X_filtered - psth_v1[np.newaxis, :, :]
    Y_residuals = Y_filtered - psth_v2[np.newaxis, :, :]
    
    print(f"Processed 3D Tensor Shapes -> X_residuals (Source V1): {X_residuals.shape}, Y_residuals (Target V2): {Y_residuals.shape}")
    
    return X_residuals, Y_residuals

def extract_window_from_tensor(X_tensor, Y_tensor, window_start, window_end):
    """
    Safely extracts a temporal window across all trials without flattening issues.
    
    X_tensor: (Trials, Timepoints, Neurons)
    Returns flattened matrices for regression fitting within that specific window: 
    X_window: (Trials * Window_Timepoints, Neurons)
    """
    x_slice = X_tensor[:, window_start:window_end, :]
    y_slice = Y_tensor[:, window_start:window_end, :]
    
    X_win_flat = x_slice.reshape(-1, X_tensor.shape[2])
    Y_win_flat = y_slice.reshape(-1, Y_tensor.shape[2])
    
    return X_win_flat, Y_win_flat
