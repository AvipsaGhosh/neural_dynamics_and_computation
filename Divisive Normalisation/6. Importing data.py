import os
from allensdk.core.brain_observatory_cache import BrainObservatoryCache

data_dir = "./allen_data_light"
boc = BrainObservatoryCache(manifest_file=os.path.join(data_dir, "manifest.json"))

v1_experiments = boc.get_ophys_experiments(
    targeted_structures=["VISp"],
    stimuli=["drifting_gratings"]
)

experiment_id = v1_experiments[0]["id"]
dg_data = boc.get_ophys_experiment_analysis(
    ophys_experiment_id=experiment_id, 
    stimulus_type="drifting_gratings"
)

peak_table = dg_data.peak
response_array = dg_data.response
angles = dg_data.orivals
