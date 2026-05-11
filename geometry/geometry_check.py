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
    if len(theta) != ns:
        raise ValueError(f"Number of theta values should be {ns}")
    if len(radius) != ns:
        raise ValueError(f"Number of radius values should be {ns}")
    print("Geometry parameters are consistent.")

def param_num_sector_toroidal(nt, theta, phi, radius):
    """Check the geometry parameters for consistency for toroidal section.
    Args:        N_t (int): Number of toroidal sectors.
        theta (list): List of theta values for control points.
        phi (list): List of phi values for control points.
        radius (list): List of radius values for control points.
    Raises:        ValueError: If the parameters are inconsistent.
    """
    print(f"Checking toroidal geometry parameters: N_t={nt},"
          f"theta={theta}, phi={phi}, radius={radius}")
    if len(theta) != nt:
        raise ValueError(f"Number of theta values should be {nt}")
    if len(phi) != nt:
        raise ValueError(f"Number of phi values should be {nt}")
    if len(radius) != nt:
        raise ValueError(f"Number of radius values should be {nt}")
    print("Toroidal geometry parameters are consistent.")
