"""List of objectives for minimization
"""
import numpy as np

def compute_elongation(cross_section):
    """
    Computes elongatio gicing cross section
    """
    z_max = np.max(cross_section[:, 2])
    z_min = np.min(cross_section[:, 2])
    r_val = np.sqrt(cross_section[:, 0]**2 + cross_section[:, 1]**2)
    r_max = np.max(r_val)
    r_min = np.min(r_val)
    epsilon = (z_max - z_min) / (r_max - r_min)

    return epsilon
