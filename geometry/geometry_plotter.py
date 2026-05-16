"""Plotting functions for DONUT geometry."""
import matplotlib.pyplot as plt
import config as cfg
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from stl import mesh
import geometry.geometry_process as gp

def poloidal_cross_section(x, y, ctrl_x, ctrl_y):
    """Plot the poloidal cross section of the geometry.
    Args:        x: list of x coordinates for each section
        y: list of y coordinates for each section
        ctrl_x: list of x coordinates of control points for each section
        ctrl_y: list of y coordinates of control points for each section
    """
    print("Plotting poloidal cross section...")
    _, ax = plt.subplots()
    for i, x_section in enumerate(x):
        plt.plot(x_section, y[i], label=f"Section {i + 1}", color = cfg.color[i])
    for j, ctrl_x_section in enumerate(ctrl_x):
        plt.scatter(ctrl_x_section, ctrl_y[j], label=f"Control Points {j + 1}",
                    color = cfg.color[j], marker='x')
    plt.grid(linestyle='--', color='gray', linewidth=0.2)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.suptitle("Poloidal Cross Section")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig("poloidal_cross_section.pdf")
    return ax

def toroidal_cross_section(xyz, ctrl_pts):
    """Plot the toroidal cross section of the geometry.
    Args:        xyz: list of (x, y, z) coordinates for the toroidal section
        ctrl_pts: list of (x, y, z) coordinates of control points for the toroidal section
    """
    print("Plotting toroidal cross section...")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = zip(*xyz)
    ax.plot(x, y, z, color = 'k', label="Toroidal Section")
    for _, ctrl_pt in enumerate(ctrl_pts):
        ax.scatter(*ctrl_pt, color = 'k', marker='x')
    ax.grid(linestyle='--', color='gray', linewidth=0.2)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(-1.0, 1.0)
    plt.tight_layout()
    plt.suptitle("Toroidal Cross Section")
    plt.savefig("toroidal_cross_section.pdf")
    return ax

def plot_stl(ax, filename, color='lightblue', alpha=0.2):
    m = mesh.Mesh.from_file(filename)

    # triangles: (N,3,3)
    triangles = m.vectors

    poly = Poly3DCollection(triangles, alpha=alpha)
    poly.set_facecolor(color)
    poly.set_edgecolor('k')

    ax.add_collection3d(poly)

    # autoscale
    scale = m.points.flatten()
    ax.auto_scale_xyz(scale, scale, scale)
    poly.set_linewidth(0.05)

def plot_geometry(points_2d, ctrl_2d,
                  points_3d, ctrl_3d,
                  moved_points_3d,
                  guide_vane_collections,
                  stlfile, current_elongation, max_elongation,
                  filename="combined_geometry"):
    """Plot the combined geometry of the poloidal and toroidal sections.
    Args:        points_2d: list of (x, y) coordinates for each poloidal section
        ctrl_2d: list of (x, y) coordinates of control points for each poloidal section
        points_3d: list of (x, y, z) coordinates for the toroidal section
        ctrl_3d: list of (x, y, z) coordinates of control points for the toroidal section
        z: list of z coordinates for the toroidal section
        ctrl_x: list of x coordinates of control points for the toroidal section
        ctrl_y: list of y coordinates of control points for the toroidal section
        ctrl_z: list of z coordinates of control points for the toroidal section"""
    print("Plotting combined geometry...")
    fig = plt.figure(figsize=(11,8))
    gs = fig.add_gridspec(2,3, width_ratios=[1, 1, 1], height_ratios=[1, 1], hspace=0.1)
    #ax1.set_axis_off()
    # Left 3D axis
    ax1 = fig.add_subplot(gs[0,0], projection='3d')
    # Middle 2D axis
    ax2 = fig.add_subplot(gs[1,0])
    # Right 2D axis
    ax3 = fig.add_subplot(gs[0:,1:], projection='3d')
    ax1.plot(points_3d[0], points_3d[1], points_3d[2], color = 'k', label="Toroidal Section")
    ax1.view_init(elev=90, azim=0)
    ax1.set_proj_type('ortho')
    ctrl_3d = np.asarray(ctrl_3d)
    ctrl_3d = ctrl_3d[:3].T
    ax1.scatter(ctrl_3d[:,0], ctrl_3d[:,1], ctrl_3d[:,2],
                marker='x', color = 'k',
                alpha=1.0,
                depthshade=False)
    ax1.grid(linestyle='--', color='gray', linewidth=0.2)
    ax1.set_box_aspect([1, 1, 0.01])
    ax1.set_xlabel("X")
    ax1.set_ylabel("")
    ax1.set_zlabel("")
    ax1.set_xlim(-1.0, 1.0)
    ax1.set_ylim(-1.0, 1.0)
    ax1.set_zlim(-1.0, 1.0)
    ax1.set_xticks(np.arange(-1.0, 1.1, step=0.5))
    ax1.set_yticks(np.arange(-1.0, 1.1, step=0.5))
    ax1.set_zticks([])
    ax1.set_title("Toroidal Cross Section")

    for i, x_section in enumerate(points_2d[0]):
        ax2.plot(x_section, points_2d[1][i], label=f"Section {i + 1}", color = cfg.color[i])
    for j, ctrl_x_section in enumerate(ctrl_2d[0]):
        ax2.scatter(ctrl_x_section, ctrl_2d[1][j], label=f"Control Points {j + 1}",
                    color = cfg.color[j], marker='x')
    ax2.grid(linestyle='--', color='gray', linewidth=0.2)
    ax2.set_aspect('equal', adjustable='box')
    ax2.set_xticks(np.arange(-1.0, 1.1, step=0.5))
    ax2.set_yticks(np.arange(-1.0, 1.1, step=0.5))
    ax2.set_title("Poloidal Cross Section")

    ax3.set_aspect('equal', adjustable='box')
    ax3.set_title("Geometry")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_zlabel("Z")
    for i, x_section in enumerate(moved_points_3d[0]):
        ax3.plot(x_section, moved_points_3d[1][i], moved_points_3d[2][i], color=cfg.color[i])
    for j, poloidal_section in enumerate(guide_vane_collections):
        for k, guide_vane in enumerate(poloidal_section):
            x = [p[0] for p in guide_vane]
            y = [p[1] for p in guide_vane]
            z = [p[2] for p in guide_vane]
            ax1.plot(x, y, z, color = 'k', lw = 0.1, alpha=0.2)

    ax3.plot(points_3d[0], points_3d[1], points_3d[2], color = 'k', linestyle = '--')
    ax3.set_xlim(-1.0, 1.0)
    ax3.set_ylim(-1.0, 1.0)
    ax3.set_zlim(-1.0, 1.0)
    ax3.set_xticks(np.arange(-1.0, 1.1, step=0.5))
    ax3.set_yticks(np.arange(-1.0, 1.1, step=0.5))
    ax3.set_zticks(np.arange(-1.0, 1.1, step=0.5))
    plot_stl(ax3, stlfile)
    plt.suptitle(f"Study type : {cfg.STUDY_NAME}, Method : {cfg.METHOD}", 
                 fontweight='bold', fontsize=16)
    fig.text(0.40, 0.10,
    f"Max elongation: {gp.CURRENT_ELONGATION:.4f}, \nMax aspect ratio: {gp.CURRENT_AR:.4f}, \nMax triangularity: {gp.CURRENT_TRIANGULARITY:.4f}",
    ha='left',
    va='center',
    fontsize=12)
    plt.savefig(f"./results/{filename}.pdf")
