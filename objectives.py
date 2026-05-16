"""List of objectives for minimization
"""
import numpy as np
import scipy.optimize as optimize
import scipy.special as special

def compute_elongation(cross_section):
    centroid = cross_section.mean(axis=0)
    pts = cross_section - centroid
    
    # PCA to find the two principal axes of the cross-section
    # The smallest eigenvector = normal to the plane
    # The other two = local axes within the plane
    cov = pts.T @ pts
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # eigh returns ascending order
    # eigenvectors[:, 0] = normal (smallest variance = out of plane)
    # eigenvectors[:, 1] = minor axis (e2)
    # eigenvectors[:, 2] = major axis (e1)
    e1 = eigenvectors[:, 2]  # largest variance direction
    e2 = eigenvectors[:, 1]  # second largest
    
    # Project onto local 2D frame
    u = pts @ e1
    v = pts @ e2
    
    a = 0.5 * (np.max(u) - np.min(u))
    b = 0.5 * (np.max(v) - np.min(v))
    
    return max(a, b) / min(a, b)

def compute_elongation_fit(cross_section):
    pts = cross_section  # shape (N, 3)
    # Perimeter
    diffs = np.diff(pts, axis=0, append=pts[:1])
    perimeter = np.sum(np.linalg.norm(diffs, axis=1))
    
    # Area via cross product (works for 3D planar polygon)
    centroid = pts.mean(axis=0)
    vecs = pts - centroid
    crosses = np.cross(vecs, np.roll(vecs, -1, axis=0))
    area = 0.5 * np.linalg.norm(crosses.sum(axis=0))
    
    # Fit equivalent ellipse: b = area / (pi * a)
    # Solve: perimeter = 4a * E(1 - (b/a)^2)
    def residual(a):
        b = area / (np.pi * a)
        e2 = 1.0 - (b / a) ** 2
        return perimeter - 4.0 * a * special.ellipe(e2)

    a_sol = optimize.fsolve(residual, x0=np.sqrt(area / np.pi))[0]
    b_sol = area / (np.pi * a_sol)
    return max(a_sol, b_sol) / min(a_sol, b_sol)
