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
    pts = np.asarray(cross_section)

    # perimeter
    diffs = np.diff(pts, axis=0, append=pts[:1])
    P = np.sum(np.linalg.norm(diffs, axis=1))

    # area
    centroid = pts.mean(axis=0)
    vecs = pts - centroid
    crosses = np.cross(vecs, np.roll(vecs, -1, axis=0))
    A = 0.5 * np.linalg.norm(crosses.sum(axis=0))

    def residual(kappa):
        a = np.sqrt(A * kappa / np.pi)
        e2 = 1.0 - 1.0 / (kappa**2)
        Pell = 4.0 * a * special.ellipe(e2)
        return Pell - P

    # circle is minimum perimeter
    r1 = residual(1.0)

    if r1 > 0:
        raise ValueError(
            "Measured perimeter is smaller than circle perimeter "
            "for same area. Geometry likely under-resolved."
        )

    # grow upper bound until sign changes
    kupper = 2.0
    while residual(kupper) < 0:
        kupper *= 2.0

        if kupper > 1e6:
            raise ValueError("Could not bracket elongation root.")

    kappa = optimize.brentq(residual, 1.0, kupper)

    return kappa
