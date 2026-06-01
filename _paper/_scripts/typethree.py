import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from geometry import geometry_process as gp
from geometry.geometry_operations import nurbs_curve, nurbs_gen, nurbs_curve_periodic
from geometry import geometry_fourier as gf
import matplotlib.pyplot as plt
import numpy as np
import config as cfg
from geometry.geometry_sector import build_sketch_sector, build_sketch_sector_toroidal, \
                    get_toroidal_coordinates_tangent, build_guide_vane
from geometry.geometry_reader import get_poloidal_sections_from_toroidal_file, \
                            get_geometry_parameters_from_poloidal_file,\
                            get_geometry_parameters_from_toroidal_file
from geometry.geometry_operations import rotate_poloidal_section, generate_periodic_data
from launch_geometry import geometry_elongation
import json

def plot_typethree(points_2d, ctrl_2d,
                  points_3d, ctrl_3d,
                  moved_points_3d,
                  moved_ctrl_points,
                  toroidal_coordinates,
                  guide_vane_collections,
                  stlfile,
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
    fig = plt.figure(figsize=(7,7))
    gs = fig.add_gridspec(1,1, width_ratios=[1], height_ratios=[1], hspace=0.1)
    #ax1.set_axis_off()
    # Left 3D axis
    ax3 = fig.add_subplot(gs[:,:], projection='3d')
    
    ctrl_3d = np.asarray(ctrl_3d)
    ctrl_3d = ctrl_3d[:3].T

    ax3.set_aspect('equal', adjustable='box')
    # ax3.set_title("Geometry")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_zlabel("Z")
    ax3.plot([0,ctrl_3d[1,0]], [0,ctrl_3d[1,1]], [0,ctrl_3d[1,2]],
             color = 'k')
    
    ax3.plot([0,0], [0,0], [-2,2],
             color = 'k', linewidth = 0.3, alpha = 0.3)
    ax3.plot([0,0], [-2,2], [0,0],
             color = 'k', linewidth = 0.3, alpha = 0.3)
    ax3.plot([-2,2], [0,0], [0,0],
             color = 'k', linewidth = 0.3, alpha = 0.3)
    
    ax3.scatter(ctrl_3d[:,0], ctrl_3d[:,1], ctrl_3d[:,2],
                marker='x', color = 'k',
                alpha=1.0,
                depthshade=False,
                label = 'toroidal control points')
    # for i, x_section in enumerate(moved_points_3d[0]):
    ax3.plot(moved_points_3d[0][3], moved_points_3d[1][3], moved_points_3d[2][3], color=cfg.color[3], label = 'poloidal section')
    ax3.scatter(moved_ctrl_points[0][3], moved_ctrl_points[1][3], moved_ctrl_points[2][3], color=cfg.color[3], marker = '+',
                depthshade=False)
    ax3.scatter(toroidal_coordinates[3][0],
                toroidal_coordinates[3][1],
                toroidal_coordinates[3][2], color='k', depthshade=False)

    # original point
    x0 = moved_ctrl_points[0][3]
    y0 = moved_ctrl_points[1][3]
    z0 = moved_ctrl_points[2][3]
    # angle range (40 degrees)
    theta = np.linspace(0, np.deg2rad(40), 200)
    # rotation
    for i in range(4):
        x = x0[i] * np.cos(theta) - y0[i] * np.sin(theta)
        y = x0[i] * np.sin(theta) + y0[i] * np.cos(theta)
        z = np.full_like(theta, z0[i])
        # plot
        ax3.plot(x, y, z, color = cfg.color[3], linestyle = ':')
    ax3.plot(points_3d[0], points_3d[1], points_3d[2], 
             color = 'k', linestyle = '--', label = 'toroidal guide vane')
    ax3.set_xlim(-1.0, 1.0)
    ax3.set_ylim(-1.0, 1.0)
    ax3.set_zlim(-1.0, 1.0)
    ax3.set_xticks(np.arange(-1.0, 1.1, step=0.5))
    ax3.set_yticks(np.arange(-1.0, 1.1, step=0.5))
    ax3.set_zticks(np.arange(-1.0, 1.1, step=0.5))   
    ax3.xaxis._axinfo["grid"]['linestyle'] = ':'
    ax3.xaxis._axinfo["grid"]['linewidth'] = 0.5
    ax3.yaxis._axinfo["grid"]['linestyle'] = ':'
    ax3.yaxis._axinfo["grid"]['linewidth'] = 0.5
    ax3.zaxis._axinfo["grid"]['linestyle'] = ':'
    ax3.zaxis._axinfo["grid"]['linewidth'] = 0.5
    # plot_stl(ax3, stlfile)
    # plt.suptitle(f"Study type : {cfg.STUDY_NAME}, Method : {cfg.METHOD}", 
    #            fontweight='bold', fontsize=16)
    ax3.text(0.1, 1.0, 0, r'$\mathbf{T_1}$($r_{t_1}, \theta_{1}, \phi_{1}$)')
    ax3.text(-0.3, 1.0, 0.16, r'$P_1$($r_{p_1}, \psi_{1}$)', color=cfg.color[3])
    ax3.text(-0.33, 1.0, -0.1, r'$\mathbf{Q}$')
    # ax3.text(0.0, -0.8, -0.1, r'$\alpha$')
    
    plt.legend()
    plt.savefig(f"./_paper/{filename}.pdf")

def geometry_preprocess(toroidal_sections, poloidal_sections, mode):
    """
    Preprocess the geometry data by building the sketches of the toroidal and poloidal sections.
    This function prepares the data for further processing and optimization.
    """
    x_collections = []
    y_collections = []
    x_moved_collections = []
    y_moved_collections = []
    z_moved_collections = []
    x_moved_ctrl_collections = []
    y_moved_ctrl_collections = []
    z_moved_ctrl_collections = []
    ctrl_x_collections = []
    ctrl_y_collections = []
    curvegeo = build_sketch_sector_toroidal(toroidal_sections['theta'],
                                            toroidal_sections['phi'],
                                            toroidal_sections['radius'],
                                            toroidal_sections['degree'],
                                            toroidal_sections['weights'])
    x, y, z = curvegeo.curve
    ctrl_x, ctrl_y, ctrl_z = curvegeo.ctrl
    toroidal_geo = get_toroidal_coordinates_tangent(toroidal_sections['sections'],
                                                        [x, y, z])
    for i, poloidal_file in enumerate(poloidal_sections):
        if mode == 0:
            poloid_file = get_geometry_parameters_from_poloidal_file("./inputs/" +
                                                    poloidal_file + ".json")
            bss = build_sketch_sector(poloid_file['theta'],
                                                    poloid_file['radius'],
                                                    poloid_file['degree'],
                                                    poloid_file['weights'])
        else:
            bss = build_sketch_sector(poloidal_file['psi'],
                                                    poloidal_file['radius'],
                                                    poloidal_file['degree'],
                                                    poloidal_file['weights'])
        x_collections.append(bss.x_p)
        y_collections.append(bss.y_p)
        ctrl_x_collections.append(bss.ctrl_xp)
        ctrl_y_collections.append(bss.ctrl_yp)
        # moved_points = move_poloidal_section_origin([x, y], toroidal_coordinates[i])
        moved_points = rotate_poloidal_section([bss.x_p, bss.y_p, [0.0]*len(bss.x_p)],
                                               toroidal_geo.toroidal_coordinates[i],
                                               toroidal_geo.toroidal_tangents[i])
        moved_ctrl_points = rotate_poloidal_section([bss.ctrl_xp, bss.ctrl_yp,
                                                     [0.0]*len(bss.ctrl_xp)],
                                               toroidal_geo.toroidal_coordinates[i],
                                               toroidal_geo.toroidal_tangents[i])
        x_moved_collections.append(moved_points[0])
        y_moved_collections.append(moved_points[1])
        z_moved_collections.append(moved_points[2])

        x_moved_ctrl_collections.append(moved_ctrl_points[0])
        y_moved_ctrl_collections.append(moved_ctrl_points[1])
        z_moved_ctrl_collections.append(moved_ctrl_points[2])
    return x_collections, y_collections, \
        x_moved_collections ,y_moved_collections, z_moved_collections, \
            ctrl_x_collections, ctrl_y_collections, \
            toroidal_geo.toroidal_coordinates, toroidal_geo.toroidal_tangents, \
            x, y, z, ctrl_x, ctrl_y, ctrl_z,\
            x_moved_ctrl_collections, y_moved_ctrl_collections, z_moved_ctrl_collections

def geometry_construct(toroidal_sections, poloidal_sections, mode, init=False, plot=False, filename=None):
    """
    Construct the geometry based on the provided toroidal and poloidal sections.
    Args:
    toroidal_sections: A dictionary containing the parameters for the toroidal section.
    poloidal_sections: A list of dictionaries containing the parameters for each poloidal section.
    mode: An integer indicating the mode of operation (0 for reading from files, 
            1 for using provided data).
    """
    x_collections, y_collections, \
        x_moved_collections ,y_moved_collections, z_moved_collections, \
            ctrl_x_collections, ctrl_y_collections, \
            toroidal_coordinates, toroidal_tangents, \
                 x, y, z, ctrl_x, ctrl_y, ctrl_z, \
                    x_moved_ctrl_collections, y_moved_ctrl_collections, z_moved_ctrl_collections  = geometry_preprocess(toroidal_sections,
                                                                       poloidal_sections,
                                                                       mode)

    elongation_list, file_list, guide_vane_collections = geometry_elongation(poloidal_sections, \
                                                                [x_moved_collections,
                                                                y_moved_collections,
                                                                z_moved_collections], \
                                                                toroidal_tangents, 
                                                                toroidal_coordinates, plot=True)
    if plot:
        # merge_stls(file_list, "./typethree.stl")
        plot_typethree([x_collections, y_collections], [ctrl_x_collections, ctrl_y_collections],
                  [x, y, z], [ctrl_x, ctrl_y, ctrl_z],
                  [x_moved_collections, y_moved_collections, z_moved_collections],
                  [x_moved_ctrl_collections, y_moved_ctrl_collections, z_moved_ctrl_collections],
                  toroidal_coordinates,
                  guide_vane_collections, "./typethree.stl",
                  filename=filename)
    return 0

x0, data_format = gf.linearize_data("./inputs/toroidal_section.json")
gp.FORMAT = data_format
toroidal_sections, poloidal_sections = gf.delinearize_data(x0)
geometry_construct(toroidal_sections, poloidal_sections,
                          1, plot=True, filename="./typethree_geometry")

psi = []
radius = []
phi = []
for i in range(4):
    filename = "./inputs/poloidal_section_" + str(i + 1) + ".json"
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
        nval = data.get("N_s")
        psi.append(np.array(data.get("psi")) * nval / 360)
        radius.append(np.array(data.get("radius")))

filename = "./inputs/toroidal_section.json"
with open(filename, "r", encoding="utf-8") as f:
    data = json.load(f)
    phi.append(np.array(data.get("sections")))

plt.cla()
plt.close()
u_vals = np.linspace(0, 1, 200)
fig, ((ax_1, ax_2), (ax_3, ax_4)) = plt.subplots(2, 2, figsize=(12, 8))
x = np.linspace(0, 1, 4)
psi_nurbs = []
weights = [1.0, 1.0, 1.0, 1.0]  # Example weights for the control points
degree = 2
for i in range(len(psi[0])):
    psi_0 = [p[i] / (2 * np.pi) for p in psi]
    ax_3.plot(phi[0], psi_0, label=r"$\psi_" + str(i + 1) + "$", marker='o', 
              linewidth = 0.8)
    psi_nurbs.append([[phi[0][0],psi_0[0]],
                      [phi[0][1],psi_0[1]],
                           [phi[0][2],psi_0[2]],
                           [phi[0][3],psi_0[3]],
                           [1.0,psi_0[0]]])

    curve_points = np.array([nurbs_gen(psi_nurbs[i], 
                            [1.0, 5.0, 5.0, 5.0, 1.0], 2, u) for u in u_vals])
    ax_3.plot(curve_points[:, 0], curve_points[:, 1],
              linestyle='--', color = cfg.color[i], 
              linewidth = 0.8)
    
    radius_0 = [r[i] for r in radius]
    ax_4.plot(phi[0], radius_0, label=r"$r_" + str(i + 1) + "$", marker='o')


ax_3.legend(loc='upper right')
ax_3.set_xlabel(r"$\theta$ (normalized by 2$\pi$)")
ax_3.set_ylabel(r"$\psi$ (normalized by 2$\pi$)")
ax_3.set_title("Poloidal Angles")
ax_4.legend(loc='upper right')
ax_4.set_xlabel(r"$\theta$ (normalized by 2$\pi$)")
ax_4.set_ylabel(r"$r$ (normalized by $a$)")
ax_4.set_title("Poloidal Radii")

for i in range(len(psi[0])):
    psi_0 = [p[i] for p in psi]
    radius_0 = [r[i] for r in radius]

psi_deltas = []
psi_delta_nurbs = []
radius_deltas = []
bss = []
for i in range(1, len(psi[0])):
    psi_0 = [p[i] for p in psi]
    radius_0 = [r[i] for r in radius]
    psi_delta = [psi_0[j] - psi_0[j-1] for j in range(1, len(psi_0))]
    radius_delta = [radius_0[j] - radius_0[j-1] for j in range(1, len(radius_0))]
    psi_deltas.append(psi_delta)
    radius_deltas.append(radius_delta)

for j in range(len(psi[0])-1):
    psi_delta_nurbs.append([[phi[0][1],psi_deltas[j][0]],
                           [phi[0][2],psi_deltas[j][1]],
                           [phi[0][3],psi_deltas[j][2]]])
    curve_points = np.array([nurbs_gen(psi_delta_nurbs[j], [1.0, 5.0, 1.0], 2, u) for u in u_vals])
    ax_1.plot(curve_points[:,0], curve_points[:,1], linestyle='--', color = cfg.color[j], linewidth = 0.8)
    ax_1.plot(phi[0][1:], psi_deltas[j], 
              label=r"$\psi_" + str(j + 1) + "$", 
              marker='o', color = cfg.color[j],
              linewidth = 0.8)
    ax_2.plot(phi[0][1:], radius_deltas[j], 
              label=r"$r_" + str(j + 1) + "$", 
              marker='o', color = cfg.color[j],
              linewidth = 0.8)

ax_1.legend(loc='upper right')
ax_1.set_xlabel(r"$\theta$ (normalized by 2$\pi$)")
ax_1.set_ylabel(r"$\Delta  \psi$ (normalized by 2$\pi$)")
ax_1.set_title("Poloidal Angles")
ax_1.grid(True, linestyle='--', alpha=0.5)
ax_2.legend(loc='upper right')
ax_2.set_xlabel(r"$\theta$ (normalized by 2$\pi$)")
ax_2.set_ylabel(r"$\Delta r$ (normalized by $a$)")
ax_2.set_title("Poloidal Radii")
ax_2.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("./_paper/poloidal_delta_psi.pdf")