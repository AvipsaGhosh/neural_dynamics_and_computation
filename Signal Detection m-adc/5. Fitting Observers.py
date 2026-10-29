# ---------------------------------------------------------
# Objective function with single sensitivity (d1 = d2 = d)
# Parameters: [d, c1, c2]
# ---------------------------------------------------------
def nll_detection_experiment(params, counts):
    d, c1, c2 = params
    if d < 0:
        return 1e10
    prob_matrix = get_stimulus_response_matrix(d, d, c1, c2)
    prob_matrix = np.clip(prob_matrix, 1e-12, 1.0)
    return -np.sum(counts * np.log(prob_matrix))

# Fit every observer and store parameter results
results_dict = {}

print("Fitting observers...")
for obs_name, counts in observer_data.items():
    res = minimize(
        fun=nll_detection_experiment,
        x0=[2.0, 1.0, 1.0],
        args=(counts,),
        method="BFGS"
    )
    
    d_est, c1_est, c2_est = res.x
    bias = c1_est - c2_est
    results_dict[obs_name] = {
        "d": d_est,
        "c1": c1_est,
        "c2": c2_est,
        "bias": bias,
        "nll": res.fun
    }
    print(f"{obs_name} -> Sensitivity d: {d_est:.2f} | Bias (c1 - c2): {bias:+.3f}")
