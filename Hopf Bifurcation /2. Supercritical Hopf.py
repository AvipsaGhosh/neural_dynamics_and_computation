# ==============================================================================
# 2. NORMAL FORM DYNAMICAL CORE (SUPERCRITICAL HOPF)
# ==============================================================================

def simulate_hopf(Z: float, omega: float = OMEGA, dt: float = DT, 
                  steps: int = STEPS, noise: float = NOISE_STD, 
                  init_state: tuple = (0.01, 0.0)):
    """
    Integrates the canonical normal form of a supercritical Hopf bifurcation:
        dE/dt = Z*E - omega*I - E*(E^2 + I^2)
        dI/dt = omega*E + Z*I - I*(E^2 + I^2)
    """
    E = np.zeros(steps)
    I = np.zeros(steps)
    E[0], I[0] = init_state
    
    sqrt_dt = np.sqrt(dt)
    
    # Forward Euler numerical integration
    for t in range(steps - 1):
        r2 = E[t]**2 + I[t]**2
        dE = Z * E[t] - omega * I[t] - E[t] * r2
        dI = omega * E[t] + Z * I[t] - I[t] * r2
        
        stoch_E = noise * sqrt_dt * np.random.randn() if noise > 0 else 0.0
        stoch_I = noise * sqrt_dt * np.random.randn() if noise > 0 else 0.0
        
        E[t+1] = E[t] + dE * dt + stoch_E
        I[t+1] = I[t] + dI * dt + stoch_I
        
    return E, I
