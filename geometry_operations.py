"""
Geometry operations for DONUT project.
This module contains functions for performing geometric 
operations such as evaluating NURBS curves, converting polar to Cartesian coordinates,
and generating knot vectors for B-splines.
"""
import math

def cox_de_boor(u, i, p, knot):
    """
    Cox-de Boor recursion for B-spline basis functions
    u : parameter value
    i : control point index
    p : degree
    knot : knot vector (non-decreasing)
    """
    if p == 0:
        return 1.0 if knot[i] <= u < knot[i+1] else 0.0
    left = 0.0
    right = 0.0
    if knot[i+p] != knot[i]:
        left = (u - knot[i]) / (knot[i+p] - knot[i]) * cox_de_boor(u, i, p-1, knot)
    if knot[i+p+1] != knot[i+1]:
        right = (knot[i+p+1] - u) / (knot[i+p+1] - knot[i+1]) * cox_de_boor(u, i+1, p-1, knot)
    return left + right

def nurbs_curve(ctrl_pts, weights, degree, knot, num_points=100):
    """
    Evaluate a NURBS curve without external libraries.
    ctrl_pts : list of [x,y] or [x,y,z] control points
    weights : list of weights for each control point
    degree : degree of the curve
    knot : knot vector
    num_points : number of points along the curve to compute
    """
    n = len(ctrl_pts)
    curve_pts = []

    u_start = knot[degree]
    u_end = knot[-degree-1]

    for j in range(num_points):
        u = u_start + (u_end - u_start) * j / (num_points - 1)
        # Clamp u to last knot to avoid zero division
        if u == u_end:
            u = u_end - 1e-12
        numerator = [0.0 for _ in ctrl_pts[0]]
        denominator = 0.0
        for i in range(n):
            nval = cox_de_boor(u, i, degree, knot) * weights[i]
            numerator = [numerator[k] + nval * ctrl_pts[i][k] for k in range(len(ctrl_pts[0]))]
            denominator += nval
        curve_pts.append([coord / denominator for coord in numerator])
    return curve_pts

def get_cartesian_coordinates(theta, radius):
    """
    Convert polar coordinates to Cartesian coordinates.
    theta : list of angles in degrees
    radius : list of corresponding radii
    Returns a list of (x, y) tuples.
    """
    cartesian_coords = []
    # cartesian_coords.append([radius[0], 0.0])
    for t, r in zip(theta, radius):
        x = round(r * math.cos(math.radians(t)), 4)
        y = round(r * math.sin(math.radians(t)), 4)
        cartesian_coords.append([x, y])
    # cartesian_coords.append([radius[-1], 0.0])
    return cartesian_coords

def generate_open_clamped_knots(n_ctrl_pts, degree):
    """
    Generate an open clamped B-spline knot vector.

    n_ctrl_pts : number of control points (n)
    degree     : spline degree (p)

    Returns:
        Knot vector of length n + p + 1
    """

    n = n_ctrl_pts
    p = degree

    # total number of knots
    m = n + p + 1

    knots = []

    # First p+1 knots (clamped start)
    knots.extend([0] * (p + 1))

    # Interior knots
    num_internal = m - 2*(p + 1)
    for i in range(1, num_internal + 1):
        knots.append(i)

    # Last p+1 knots (clamped end)
    last_value = num_internal + 1
    knots.extend([last_value] * (p + 1))

    return knots

def generate_periodic_data(ctrl_pts, degree):
    """
    Generate control points and knots for a periodic (closed) B-spline.

    ctrl_pts : list of unique control points
    degree   : spline degree (p)

    Returns:
        extended_ctrl_pts
        knot_vector
        valid_parameter_range (u_start, u_end)
    """

    p = degree
    base = ctrl_pts

    # Extend first p control points
    extended_ctrl = base + base[:p]

    n = len(extended_ctrl)

    # Uniform knot vector
    knot = list(range(n + p + 1))

    # Valid evaluation domain
    u_start = p
    u_end = len(base) + p

    return extended_ctrl, knot, (u_start, u_end)

def nurbs_curve_periodic(inputs, num_points=200):
    """Evaluate a periodic NURBS curve without external libraries.
    ctrl : list of control points (extended for periodicity)
    w : list of weights (extended for periodicity)
    p : degree of the curve
    knot : knot vector (uniform for periodic)
    u_start, u_end : valid parameter range for evaluation
    num_points : number of points along the curve to compute"""
    curve = []
    for j in range(num_points):
        u = inputs[4] + (inputs[5]-inputs[4])*j/(num_points-1)

        num = [0.0]*len(inputs[0][0])
        den = 0.0

        for i, ci in enumerate(inputs[0]):
            nval = cox_de_boor(u, i, inputs[2], inputs[3]) * inputs[1][i]
            for k, _ in enumerate(num):
                num[k] += nval * ci[k]
            den += nval

        curve.append([c/den for c in num])

    return curve
