import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# Fourier coefficients
# --------------------------

Nfp = 5

R_modes = [
    (0, 0, 1.0),   # major radius
    (1, 0, 0.25),  # elliptical cross-section
    (1, 1.0, 0.12),  # stellarator twist
]

Z_modes = [
    (1, 0, 0.25),
    (1, 1, 0.12),
]

# --------------------------
# Angle grids
# --------------------------

ntheta = 150
nphi = 150

theta = np.linspace(0, 2*np.pi, ntheta)
phi   = np.linspace(0, 2*np.pi, nphi)

TH, PH = np.meshgrid(theta, phi, indexing='ij')

# --------------------------
# Evaluate Fourier surface
# --------------------------

R = np.zeros_like(TH)
Z = np.zeros_like(TH)

for m, n, coeff in R_modes:
    R += coeff * np.cos(m*TH - n*Nfp*PH)

for m, n, coeff in Z_modes:
    Z += coeff * np.sin(m*TH - n*Nfp*PH)

X = R * np.cos(PH)
Y = R * np.sin(PH)

# --------------------------
# Plot
# --------------------------

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(
    X, Y, Z,
    linewidth=0,
    antialiased=True,
    alpha=0.8
)

ax.set_box_aspect((1,1,1))
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
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
plt.savefig("./_paper/typezero_geometry.pdf")