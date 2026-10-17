# 4. STOCHASTIC INTEGRATION ENGINE (EULER-MARUYAMA DDM)

class BioDDM:

    def __init__(self, dt: float = DT, max_time: float = MAX_TIME):
        self.dt = dt
        self.max_time = max_time
        self.sqrt_dt = np.sqrt(dt)

    def simulate_trial(self, drift, sigma, boundary, nondecision):
        evidence = 0.0
        t = 0.0

        # SDE Euler-Maruyama integration until absorbing boundary (+a or -a) is reached
        while abs(evidence) < boundary and t < self.max_time:
            evidence += (
                drift * self.dt + sigma * self.sqrt_dt * np.random.normal()
            )
            t += self.dt

        # Binary accuracy outcome (1 if positive threshold reached, else 0)
        choice = 1 if evidence >= boundary else 0
        total_reaction_time = t + nondecision

        return total_reaction_time, choice


def gather_cohort_data(model: BioDDM, params: tuple, n_trials: int = N_TRIALS):
    results = [model.simulate_trial(*params) for _ in range(n_trials)]
    reaction_times = [r[0] for r in results]
    accuracy = sum([r[1] for r in results]) / n_trials
    return reaction_times, accuracy
