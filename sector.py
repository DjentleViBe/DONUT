"""
Sector sketching functions for DONUT geometry.
This module contains functions to build the sketch of a sector based on the geometry parameters
read from the input files. It uses the geometry operations to compute the NURBS curve points
and control points, and prepares the data for plotting.
"""
from geometry_operations import nurbs_curve_periodic, \
                                get_cartesian_coordinates_2d, \
                                get_cartesian_coordinates_3d, \
                                generate_periodic_data

def build_sketch_sector(theta, radius, degree, weights):
    """Build the sketch of a sector based on the geometry parameters.
    Args:   
    theta: list of angles in degrees    
    radius: list of corresponding radii
    degree: degree of the NURBS curve
    weights: list of weights for the control points
    Returns:
    x: list of x coordinates of the curve points
    y: list of y coordinates of the curve points
    ctrl_x: list of x coordinates of the control points
    ctrl_y: list of y coordinates of the control points"""
    ctrl_pts = get_cartesian_coordinates_2d(theta, radius)
    ctrl_ext, knot, (u_start, u_end) = generate_periodic_data(
        ctrl_pts,
        degree
    )
    weights_ext = weights + weights[:degree]
    # Compute curve points
    curve_points = nurbs_curve_periodic(
        [ctrl_ext,
        weights_ext,
        degree,
        knot,
        u_start,
        u_end]
    )
    x, y = zip(*curve_points)
    ctrl_x, ctrl_y = zip(*ctrl_pts)
    return x, y, ctrl_x, ctrl_y

def build_sketch_sector_toroidal(theta, phi, radius, degree, weights):
    """Build the sketch of a toroidal sector based on the geometry parameters.
    Args:   
    theta: list of angles in degrees for the poloidal direction    
    phi: list of angles in degrees for the toroidal direction
    radius: list of corresponding radii
    degree: degree of the NURBS curve
    weights: list of weights for the control points
    Returns:
    x: list of x coordinates of the curve points
    y: list of y coordinates of the curve points
    ctrl_x: list of x coordinates of the control points
    ctrl_y: list of y coordinates of the control points"""
    ctrl_pts = get_cartesian_coordinates_3d(theta, phi, radius)
    ctrl_ext, knot, (u_start, u_end) = generate_periodic_data(
        ctrl_pts,
        degree
    )
    weights_ext = weights + weights[:degree]
    # Compute curve points
    curve_points = nurbs_curve_periodic(
                                [ctrl_ext,
                                weights_ext,
                                degree,
                                knot,
                                u_start,
                                u_end])
    return zip(*curve_points), zip(*ctrl_pts)
