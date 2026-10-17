# 3. TRANSLATION LAYER: CIRCUIT TO LATENT DDM PARAMETERS

class BiologicalMapper:

    @staticmethod
    def map_to_ddm(brain: BrainState):
        # Drift rate (v): Cortical evidence discounted by perceived cognitive effort
        drift = brain.dlPFC_signal - brain.dmPFC_effort

        # Diffusion noise (sigma): Inversely scaled by attentional precision
        sigma = 0.10 / brain.SC_precision

        # Decision threshold (a): Controlled by pre-SMA response caution
        boundary = brain.preSMA_caution

        # Non-decision time (Ter): Base sensory-motor delay augmented by oculomotor conflict
        nondecision = 0.20 + brain.FEF_conflict

        return drift, sigma, boundary, nondecision
