"""
Geometry check for DONUT project.
This module contains functions to check the geometry parameters
for consistency and correctness.
"""
def param_num_sector(ns, theta, radius):
    """Check the geometry parameters for consistency.
    Args:        N_s (int): Number of sectors.
        theta (list): List of theta values for control points.
        radius (list): List of radius values for control points.
    Raises:        ValueError: If the parameters are inconsistent.
    """
    print(f"Checking geometry parameters: N_s={ns}, theta={theta}, radius={radius}")
    if len(theta) != ns - 1:
        raise ValueError(f"Number of theta values should be {ns-1}")
    if len(radius) != ns:
        raise ValueError(f"Number of radius values should be {ns}")
    print("Geometry parameters are consistent.")
