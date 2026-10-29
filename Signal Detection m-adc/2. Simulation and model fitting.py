from scipy.optimize import minimize

# ---------------------------------------------------------
# Step 1: Compute Full 3x3 Response Matrix for 2-ADC Task
# Stimulus conditions:
#   Row 0: Catch trial (x1 = 0, x2 = 0)
#   Row 1: Stimulus at Location 1 (x1 = 1, x2 = 0)
#   Row 2: Stimulus at Location 2 (x1 = 0, x2 = 1)
# Columns: [P(NoGo), P(Loc 1), P(Loc 2)]
# ---------------------------------------------------------
def get_stimulus_response_matrix(d1, d2, c1, c2):
    p_catch = get_response_probabilities(d1, d2, c1, c2, x1=0, x2=0)
    p_stim1 = get_response_probabilities(d1, d2, c1, c2, x1=1, x2=0)
    p_stim2 = get_response_probabilities(d1, d2, c1, c2, x1=0, x2=1)
    return np.vstack([p_catch, p_stim1, p_stim2])


# ---------------------------------------------------------
# Step 2: Simulate Synthetic Observation Counts
# True parameters from the paper: d1 = 1.0, d2 = 1.0, c1 = -0.25, c2 = 0.75
# ---------------------------------------------------------
true_params = [1.0, 1.0, -0.25, 0.75]
true_probabilities = get_stimulus_response_matrix(*true_params)

# Fixed trial numbers: 2000 catch trials, 1000 stimulus 1 trials, 1000 stimulus 2 trials
trials_per_condition = [2000, 1000, 1000]

np.random.seed(42)
observed_counts = np.array([
    np.random.multinomial(n, probs)
    for n, probs in zip(trials_per_condition, true_probabilities)
])

print("Simulated contingency table (Rows: Stimulus, Cols: Response [NoGo, Loc1, Loc2]):")
print(observed_counts)


# ---------------------------------------------------------
# Step 3: Multinomial Negative Log-Likelihood Objective
# ---------------------------------------------------------
def negative_log_likelihood(params, counts):
    d1, d2, c1, c2 = params
    
    # Penalize unphysical negative sensitivities
    if d1 < 0 or d2 < 0:
        return 1e10
    
    prob_matrix = get_stimulus_response_matrix(d1, d2, c1, c2)
    
    # Guard against zero-probability edge cases in log
    prob_matrix = np.clip(prob_matrix, 1e-12, 1.0)
    
    # Sum over all stimulus conditions and behavioral responses
    nll = -np.sum(counts * np.log(prob_matrix))
    return nll


# ---------------------------------------------------------
# Step 4: Maximum Likelihood Estimation (MLE)
# ---------------------------------------------------------
initial_guess = [0.1, 0.1, 0.0, 0.0]

result = minimize(
    fun=negative_log_likelihood,
    x0=initial_guess,
    args=(observed_counts,),
    method="BFGS"
)

# Recover standard errors from the inverted Hessian matrix
hessian_inv = result.hess_inv
param_standard_errors = np.sqrt(np.diag(hessian_inv))

# ---------------------------------------------------------
# Step 5: Display Parameter Recovery Verification
# ---------------------------------------------------------
param_names = ["d1", "d2", "c1", "c2"]

print("\n--- Model Fitting Results ---")
for name, true_val, est_val, se in zip(param_names, true_params, result.x, param_standard_errors):
    print(f"{name:2s} -> True: {true_val:6.2f} | Recovered: {est_val:6.3f} +/- {se:5.3f}")
