import numpy as np

def compute_elongation(cross_section):
    Z_max = np.max(cross_section[:, 2])
    Z_min = np.min(cross_section[:, 2])
    R = np.sqrt(cross_section[:, 0]**2 + cross_section[:, 1]**2)
    R_max = np.max(R)
    R_min = np.min(R)
    epsilon = (Z_max - Z_min) / (R_max - R_min)

    return epsilon