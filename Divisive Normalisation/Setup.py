import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from allensdk.core.brain_observatory_cache import BrainObservatoryCache

# Setup lightweight cache directory
data_dir = "./allen_data_light"
boc = BrainObservatoryCache(manifest_file=os.path.join(data_dir, "manifest.json"))

# Query experiments directly in V1 (VISp) showing drifting gratings
v1_experiments = boc.get_ophys_experiments(
    targeted_structures=["VISp"],
    stimuli=["drifting_gratings"]
)

print(f"Found {len(v1_experiments)} matching V1 drifting grating sessions.")

# Pick the first valid session
experiment_id = v1_experiments[0]["id"]
print(f"Using experiment ID: {experiment_id}")

# Download ONLY the lightweight analysis data (~10-20 MB)
dg_data = boc.get_ophys_experiment_analysis(
    ophys_experiment_id=experiment_id, 
    stimulus_type="drifting_gratings"
)

# Extract pre-computed metrics
peak_table = dg_data.peak
response_array = dg_data.response
angles = dg_data.orivals

print(f"Successfully loaded. Total cells available: {len(peak_table)}")
