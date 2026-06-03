"""
Sector sketching functions for DONUT geometry.
This module contains functions to build the sketch of a sector based on the geometry parameters
read from the input files. It uses the geometry operations to compute the NURBS curve points
and control points, and prepares the data for plotting.
"""
from classes_geometry import ToroidalGeometry, \
                            SketchGeometryPoloidal, \
                            SketchGeometryToroidal
from geometry.geometry_operations import nurbs_curve_periodic, \
                                get_cartesian_coordinates_2d, \
                                get_cartesian_coordinates_3d, \
                                generate_periodic_data
from geometry.geometry_operations import nurbs_curve
import numpy as np

def build_sketch_sector(theta, radius, degree, weights, num_points=100):
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
        u_end],
        num_points
    )
    x, y = zip(*curve_points)
    ctrl_x, ctrl_y = zip(*ctrl_pts)
    curvegeom = SketchGeometryPoloidal(x, y, ctrl_x, ctrl_y)
    return curvegeom

def build_sketch_sector_toroidal(theta, phi, radius, degree, weights, num_points=100):
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
                                u_end], num_points)
    curvegeom = SketchGeometryToroidal(list(zip(*curve_points)),  list(zip(*ctrl_pts)))
    return curvegeom

def get_toroidal_coordinates_tangent(section_limits, points_3d):
    coords = []
    tangents = []

    x = np.asarray(points_3d[0])
    y = np.asarray(points_3d[1])
    z = np.asarray(points_3d[2])

    # --- remove duplicate closure point if present ---
    if np.allclose([x[0], y[0], z[0]], [x[-1], y[-1], z[-1]]):
        x = x[:-1]
        y = y[:-1]
        z = z[:-1]

    n = len(x)
    if n < 2:
        raise ValueError("points_3d must contain at least 2 unique points")

    for s in section_limits:

        # --- enforce periodicity ---
        s = float(np.mod(s, 1.0))  # ensures s=1 -> s=0

        idx = int(round(s * n)) % n

        coords.append((x[idx], y[idx], z[idx]))

        # --- cyclic central difference ---
        idx_prev = (idx - 1) % n
        idx_next = (idx + 1) % n

        dx = 0.5 * (x[idx_next] - x[idx_prev])
        dy = 0.5 * (y[idx_next] - y[idx_prev])
        dz = 0.5 * (z[idx_next] - z[idx_prev])

        tangents.append((dx, dy, dz))
    tangents.append(tangents[0])
    coords.append(coords[0])
    return ToroidalGeometry(coords, tangents)

def is_inside_torus_axis(P0, startcenter):
    """
    Returns True if P0 is on the inboard side (closer to torus axis than startcenter).
    All distances measured in XY plane only.
    """
    R = np.sqrt(startcenter[0]**2 + startcenter[1]**2)        # major radius
    r = np.sqrt(P0[0]**2 + P0[1]**2)                          # P0's radial distance
    inside = r < R
    #thickness = 0.25 if inside else 0.25
    t = np.clip(abs(r - R) / 0.15, 0.0, 1.0)
    thickness = 0.25 * np.clip(t * 2, 1, 2)
    return thickness

def guide_vane(startpoint, endpoint, startvector, endvector, startcenter, endcenter, num_points):
    """
    Constructs spline between start and end point with 2 poins in between
    """
    P0 = np.array(startpoint)
    P3 = np.array(endpoint)

    T0 = np.array(startvector)
    T1 = np.array(endvector)

    start_w = is_inside_torus_axis(P0, startcenter)
    end_w = is_inside_torus_axis(P3, endcenter)

    P1 = P0 + start_w * T0 
    P2 = P3 - end_w * T1  # scaling vector only

    ctrl_pts = [P0, P1, P2, P3]

    spline_3d = nurbs_curve(ctrl_pts, [1.0]*4, 3, num_points = num_points)
    return spline_3d

def build_guide_vane(section_1, section_2, tangent_1, tangent_2, center_1, center_2, num_points):
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
                                      center_1, center_2,
                                      num_points))
        
    return guide_vanes
