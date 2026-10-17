# ==============================================================================
# 3. PARAMETER SWEEP ENGINE & STEADY-STATE ESTIMATION
# ==============================================================================

def compute_bifurcation_diagram(z_values, transient_ratio: float = 0.6):
    """
    Sweeps control parameter Z, isolates post-transient steady states,
    and extracts empirical vs. analytical limit cycle radii.
    """
    measured_radii = []
    theoretical_radii = []
    
    transient_cutoff = int(transient_ratio * STEPS)
    
    for z in z_values:
        E, I = simulate_hopf(Z=z, init_state=(0.01, 0.0))
        
        # Discard initial transient trajectory
        E_ss = E[transient_cutoff:]
        I_ss = I[transient_cutoff:]
        
        # Mean steady-state orbit radius
        r_ss = np.sqrt(E_ss**2 + I_ss**2)
        measured_radii.append(np.mean(r_ss))
        
        # Supercritical normal form analytical prediction: r = sqrt(Z) for Z > 0
        if z <= 0.0:
            theoretical_radii.append(0.0)
        else:
            theoretical_radii.append(np.sqrt(z))
            
    return np.array(measured_radii), np.array(theoretical_radii)

# Execute empirical parameter sweep
measured_amp, theoretical_amp = compute_bifurcation_diagram(Z_SWEEP)
