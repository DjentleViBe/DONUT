import numpy as np
import matplotlib.pyplot as plt

def gen_circle(radius):
    theta = np.linspace(0, 90, 10)
    x = radius * np.cos(np.radians(theta))
    y = radius * np.sin(np.radians(theta))
    return x, y

def rotate_point(point, axis, angle_rad):
    """
    Rotate a 3D point around an axis passing through the origin.

    Parameters
    ----------
    point : array-like, shape (3,)
        Point to rotate.
    axis : array-like, shape (3,)
        Rotation axis.
    angle_rad : float
        Rotation angle in radians.

    Returns
    -------
    np.ndarray
        Rotated point.
    """
    p = np.asarray(point, dtype=float)
    k = np.asarray(axis, dtype=float)

    # Normalize axis
    k /= np.linalg.norm(k)

    c = np.cos(angle_rad)
    s = np.sin(angle_rad)

    return (
        p * c
        + np.cross(k, p) * s
        + k * np.dot(k, p) * (1 - c)
    )
# ----------------------------
# 1. Base closed curve (U direction)
# ----------------------------
curve_base = np.array([
    [0.3, 0, 0],
    [0.5, 0, 0.4],
    [0.7, 0, 0],
    [0.5, 0, -0.4]
])

deg = 3

# ----------------------------
# 2. Generate series of curves (V direction stack)
# ----------------------------
curves = []
theta_rad = np.radians(np.linspace(0, 360, 20))
for i in range (len(theta_rad)):
    points = []
    for j in range(0, len(curve_base)):
        points.append(rotate_point(curve_base[j], [0, 0, 1], theta_rad[i]))
    curves.append(points)

curves.extend(curves[:deg])
nv = len(curves)
nu = len(curve_base)

# ----------------------------
# 3. Periodic closure in U only
# ----------------------------
def periodic(P, p):
    return np.vstack([P, P[:p]])

U = periodic(curve_base, deg)

nu = len(U)

# ----------------------------
# 4. Build control net (THIS is the key change)
# ----------------------------
ctrl = np.zeros((nu, nv, 3))

for j in range(nv):
    for i in range(nu):
        ctrl[i, j] = curves[j][i % len(curve_base)]

# ----------------------------
# 5. Knot vectors
# ----------------------------
ku = np.arange(0, nu + deg + 1)
kv = np.arange(0, nv + deg + 1)

# ----------------------------
# 6. Cox–de Boor basis
# ----------------------------
def basis(i, k, t, knots):
    if k == 0:
        return 1.0 if knots[i] <= t < knots[i+1] else 0.0

    d1 = knots[i+k] - knots[i]
    d2 = knots[i+k+1] - knots[i+1]

    a = b = 0.0

    if d1 != 0:
        a = (t - knots[i]) / d1 * basis(i, k-1, t, knots)

    if d2 != 0:
        b = (knots[i+k+1] - t) / d2 * basis(i+1, k-1, t, knots)

    return a + b

# ----------------------------
# 7. Surface evaluation
# ----------------------------
def surface(u, v):
    u = ku[deg] + u * (ku[-deg-1] - ku[deg])
    v = kv[deg] + v * (kv[-deg-1] - kv[deg])

    S = np.zeros(3)
    W = 0.0

    for i in range(nu):
        Nu = basis(i, deg, u, ku)
        for j in range(nv):
            Nv = basis(j, deg, v, kv)

            w = Nu * Nv
            S += w * ctrl[i, j]
            W += w

    return S / W

# ----------------------------
# 8. Sampling grid
# ----------------------------
res = 50
Uvals = np.linspace(0, 1, res)
Vvals = np.linspace(0, 1, res)

X = np.zeros((res, res))
Y = np.zeros((res, res))
Z = np.zeros((res, res))

for i, u in enumerate(Uvals):
    for j, v in enumerate(Vvals):
        p = surface(u, v)
        X[i, j], Y[i, j], Z[i, j] = p

# ----------------------------
# 9. Plots
# ----------------------------
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-1.0, 1.0)
ax.set_ylim(-1.0, 1.0)
ax.set_zlim(-1.0, 1.0)
ax.set_xticks(np.arange(-1.0, 1.1, step=0.5))
ax.set_yticks(np.arange(-1.0, 1.1, step=0.5))
ax.set_zticks(np.arange(-1.0, 1.1, step=0.5))   
ax.xaxis._axinfo["grid"]['linestyle'] = ':'
ax.xaxis._axinfo["grid"]['linewidth'] = 0.5
ax.yaxis._axinfo["grid"]['linestyle'] = ':'
ax.yaxis._axinfo["grid"]['linewidth'] = 0.5
ax.zaxis._axinfo["grid"]['linestyle'] = ':'
ax.zaxis._axinfo["grid"]['linewidth'] = 0.5
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5)
ax.set_aspect('equal', adjustable='box')
# control points
ctrl_flat = ctrl.reshape(-1, 3)
print(f"Total number of design variables : {len(ctrl_flat)*3}")
ax.scatter(ctrl_flat[:, 0], ctrl_flat[:, 1], ctrl_flat[:, 2],
           color='k', s=10)

# U-direction
for i in range(nu):
    ax.plot(ctrl[i, :, 0], ctrl[i, :, 1], ctrl[i, :, 2], 'k--', linewidth=0.6)

# V-direction
for j in range(nv):
    ax.plot(ctrl[:, j, 0], ctrl[:, j, 1], ctrl[:, j, 2], 'k--', linewidth=0.6)

# ax.set_title("NURBS Surface from Curve Stack (Closed U, Discrete V)")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.savefig("./_paper/typefive_geometry.pdf")