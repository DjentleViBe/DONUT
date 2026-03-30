"""
Sector sketching functions for DONUT geometry.
This module contains functions to build the sketch of a sector based on the geometry parameters
read from the input files. It uses the geometry operations to compute the NURBS curve points
and control points, and prepares the data for plotting.
"""
from geometry.geometry_operations import nurbs_curve_periodic, \
                                get_cartesian_coordinates_2d, \
                                get_cartesian_coordinates_3d, \
                                generate_periodic_data
from geometry.geometry_operations import nurbs_curve
import numpy as np

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
    weights_ext = list(weights) + list(weights[:degree])
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
    weights_ext = list(weights) + list(weights[:degree])
    # Compute curve points
    curve_points = nurbs_curve_periodic(
                                [ctrl_ext,
                                weights_ext,
                                degree,
                                knot,
                                u_start,
                                u_end])
    return zip(*curve_points), zip(*ctrl_ext)

def get_toroidal_coordinates_tangent(section_limits, points_3d):
    """Extract the toroidal coordinates (x, y, z) from the 3D points of 
    the toroidal section.
    Args:
        section_limits: list of tuples defining the limits of the toroidal 
        section in terms of theta and phi points_3d: list of (x, y, z) 
        coordinates for the toroidal section
    Returns:
        xyz: list of x, y, z coordinates of the toroidal section
    """
    toroidal_coordinates = []
    toroidal_tangents = []
    for sect in section_limits:
        index = int(len(points_3d[0]) * sect)
        toroidal_coordinates.append((points_3d[0][index],
                                     points_3d[1][index],
                                     points_3d[2][index]))
        if index < len(points_3d[0]) - 1:
            tangent = (points_3d[0][index + 1] - points_3d[0][index],
                       points_3d[1][index + 1] - points_3d[1][index],
                       points_3d[2][index + 1] - points_3d[2][index])
        else:
            tangent = (points_3d[0][index] - points_3d[0][index - 1],
                       points_3d[1][index] - points_3d[1][index - 1],
                       points_3d[2][index] - points_3d[2][index - 1])
        toroidal_tangents.append(tangent)
    return toroidal_coordinates, toroidal_tangents

def guide_vane(startpoint, endpoint, startvector, endvector, start_w, end_w):
    """
    Constructs spline between start and end point with 2 poins in between
    """
    P0 = np.array(startpoint)
    P3 = np.array(endpoint)

    T0 = np.array(startvector)
    T1 = np.array(endvector)

    P1 = P0 + start_w * T0
    P2 = P3 + end_w * T1   # scaling vector only

    ctrl_pts = [P0, P1, P2, P3]

    spline_3d = nurbs_curve(ctrl_pts, [1.0]*4, 3)
    return spline_3d

def build_guide_vane(section_1, section_2, tangent_1, tangent_2):
    """
    Builds guide vane from 2 closed sections
    """
    guide_vanes = []
    for j, section in enumerate(section_1[0]):
        u_unit = tangent_1 / np.linalg.norm(tangent_1)
        v_unit = tangent_2 / np.linalg.norm(tangent_2)
        guide_vanes.append(guide_vane([section_1[0][j], section_1[1][j],section_1[2][j]],
                                      [section_2[0][j], section_2[1][j],section_2[2][j]],
                                      u_unit, v_unit,
                                      0.5, -0.5))
        
    return guide_vanes
