"""List of objectives for minimization
"""
import numpy as np

def compute_elongation(cross_section):
    """Compute the elongation of a given cross section."""
    z_max = np.max(cross_section[:, 2])
    z_min = np.min(cross_section[:, 2])
    r = np.sqrt(cross_section[:, 0]**2 + cross_section[:, 1]**2)
    r_max = np.max(r)
    r_min = np.min(r)
    denom = r_max - r_min
    if denom < 1e-12:
        return np.inf  # or a large penalty value
    return (z_max - z_min) / denom
