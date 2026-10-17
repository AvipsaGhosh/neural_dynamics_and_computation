# 2. CLINICAL PHENOTYPE PARAMETERIZATION

# Clinical condition toggle:
# Set to True for healthy working memory maintenance.
# Set to False to model cortical NMDA hypofunction and attractor collapse.
is_healthy = True

if is_healthy:
    # Intact recurrent NMDA coupling and balanced cortical noise
    nmda_strength = 0.005
    inhibition_strength = 0.12
    sigma_noise = 1.2 * mV
    p_recurrent = 0.25
else:
    # Schizophrenia model: NMDA hypofunction, impaired interneuron recruitment,
    # and elevated baseline cortical stochasticity
    nmda_strength = 0.002
    inhibition_strength = 0.08
    sigma_noise = 2.2 * mV
    p_recurrent = 0.15
