import numpy as np
from scipy.optimize import curve_fit

responsive_cells = peak_table[
    (peak_table["ptest_dg"] < 0.05) & 
    (peak_table["osi_dg"] > 0.5)
]

chosen_cell_id = responsive_cells.index[0]
cell_loc = peak_table.index.get_loc(chosen_cell_id)

best_tf = responsive_cells.loc[chosen_cell_id, "tf_dg"]
tf_idx = np.where(dg_data.tfvals == best_tf)[0][0]
empirical_responses = response_array[:, tf_idx, cell_loc, 0]

def von_mises_model(theta, A, kappa, theta_pref, baseline):
    return A * np.exp(kappa * np.cos(2 * (theta - theta_pref))) + baseline

angles_rad = np.deg2rad(angles)
init_guess = [
    np.max(empirical_responses) - np.min(empirical_responses),
    2.0,
    angles_rad[np.argmax(empirical_responses)] % np.pi,
    np.min(empirical_responses)
]
bounds = ([0.0, 0.0, 0.0, -np.inf], [np.inf, 20.0, np.pi, np.inf])

popt, _ = curve_fit(von_mises_model, angles_rad, empirical_responses, p0=init_guess, bounds=bounds)
A_fit, kappa_fit, theta_pref_fit, baseline_fit = popt
