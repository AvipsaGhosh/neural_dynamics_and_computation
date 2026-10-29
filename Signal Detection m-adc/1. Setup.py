import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

def get_response_probabilities(d1, d2, c1, c2, x1, x2):
    # Probability of responding at Location 1
    integrand_1 = lambda e: norm.cdf(e + d1*x1 - d2*x2 - c1 + c2) * norm.pdf(e)
    p1, _ = quad(integrand_1, c1 - d1*x1, np.inf)
    
    # Probability of responding at Location 2
    integrand_2 = lambda e: norm.cdf(e + d2*x2 - d1*x1 - c2 + c1) * norm.pdf(e)
    p2, _ = quad(integrand_2, c2 - d2*x2, np.inf)
    
    # Probability of a NoGo response
    p0 = norm.cdf(c1 - d1*x1) * norm.cdf(c2 - d2*x2)
    
    return np.array([p0, p1, p2])

# Test catch trial probabilities from Figure 4F
probs_catch = get_response_probabilities(d1=1.0, d2=1.0, c1=-0.25, c2=0.75, x1=0, x2=0)
print("Catch trial probabilities [P(NoGo), P(Loc 1), P(Loc 2)]:", probs_catch)
