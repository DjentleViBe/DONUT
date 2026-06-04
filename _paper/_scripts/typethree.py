import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from geometry import geometry_process as gp
from geometry.geometry_operations import nurbs_gen_periodic, nurbs_gen
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
from classes_geometry import GeometryData, \
                            PoloidalGeometry, \
                            ToroidalGeometry
from geometry.geometry_operations import nurbs_gen, get_nurbs_y

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
    ax3.plot([0,ctrl_3d[0,0]], [0,ctrl_3d[0,1]], [0,ctrl_3d[0,2]],
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
    ind = 3
    ax3.plot(moved_points_3d[0][ind], moved_points_3d[1][ind], moved_points_3d[2][ind], color=cfg.color[3], label = 'poloidal section')
    ax3.scatter(moved_ctrl_points[0][ind], moved_ctrl_points[1][ind], moved_ctrl_points[2][ind], color=cfg.color[3], marker = '+',
                depthshade=False)
    ax3.scatter(toroidal_coordinates[3][0],
                toroidal_coordinates[3][1],
                toroidal_coordinates[3][2], color='k', depthshade=False)

    # original point
    x0 = moved_ctrl_points[0][ind]
    y0 = moved_ctrl_points[1][ind]
    z0 = moved_ctrl_points[2][ind]
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
    ax3.text(0.8, 0.0, 0.1, r'$\mathbf{T_1}$($r_{t_1}, \theta_{1}, \phi_{1}$)')
    ax3.text(-0.3, 1.0, 0.16, r'$P_1$($r_{p_1}, \psi_{1}$)', color=cfg.color[3])
    # ax3.text(-0.33, 1.0, -0.1, r'$\mathbf{Q}$')
    # ax3.text(0.0, -0.8, -0.1, r'$\alpha$')
    
    plt.legend()
    plt.savefig(f"./_paper/{filename}.pdf")

def geometry_preprocess(toroidal_sections, poloidal_sections):
    """
    Preprocess the geometry data by building the sketches of the toroidal and poloidal sections.
    This function prepares the data for further processing and optimization.
    """
    x_moved_ctrl_collections = []
    y_moved_ctrl_collections = []
    z_moved_ctrl_collections = []
    geom = GeometryData([], [], [], [], [], [], [])
    curvegeom = build_sketch_sector_toroidal(toroidal_sections['theta'],
                                            toroidal_sections['phi'],
                                            toroidal_sections['radius'],
                                            toroidal_sections['degree'],
                                            toroidal_sections['weights'],
                                            cfg.NUM_T)
    x, y, z = curvegeom.curve
    ctrl_x, ctrl_y, ctrl_z = curvegeom.ctrl
    z = toroidal_sections['deltas']
    expz = np.exp(z - np.max(z))
    deltas = expz / expz.sum()
    sections = np.cumsum(deltas)
    sections = np.concatenate([[0], sections[:-1]])
    toroidal_geom = get_toroidal_coordinates_tangent(sections, curvegeom.curve)
    twist_params = (0.0, 0.0, 0.0)
    psi_collect = []
    radius_collect = []
    for i, poloidal_file in enumerate(poloidal_sections):
        curvegeompol = build_sketch_sector(poloidal_file['psi'],
                                                    poloidal_file['radius'],
                                                    poloidal_file['degree'],
                                                    poloidal_file['weights'],
                                                    cfg.NUM_P)
        geom.ctrl_x_collections.append(curvegeompol.ctrl_xp)
        geom.ctrl_y_collections.append(curvegeompol.ctrl_yp)
        if cfg.STUDY_NAME == "Type3":
            psi_collect.append(np.array(poloidal_file['psi']))
            radius_collect.append(np.array(poloidal_file['radius']))
    if cfg.STUDY_NAME == "Type3":
        psi_nurbs = []
        radius_nurbs = []
        psi_points = []
        radius_points = []
        u_vals = np.linspace(0.0, 1.0, cfg.NUM_T)
        for i in range(len(psi_collect[0])):
            psi_0 = [p[i] / 360.0 for p in psi_collect]
            radius_0 = [r[i] for r in radius_collect]
            psi_nurbs.append([[sections[0], psi_0[0]],
                            [sections[1], psi_0[1]],
                            [sections[2], psi_0[2]],
                            [sections[3], psi_0[3]],
                            [1.0, 1.0 + psi_0[0]]])
            radius_nurbs.append([[sections[0], radius_0[0]],
                            [sections[1], radius_0[1]],
                            [sections[2], radius_0[2]],
                            [sections[3], radius_0[3]],
                            [1.0, radius_0[0]]])
            psi_points.append(get_nurbs_y(u_vals, 
                        psi_nurbs[i], 
                        [1.0, 5.0, 5.0, 5.0, 1.0],
                        2)[:-1])
            radius_points.append(get_nurbs_y(u_vals, 
                        radius_nurbs[i], 
                        [1.0, 5.0, 5.0, 5.0, 1.0],
                        2))
        toroid_geom = get_toroidal_coordinates_tangent(u_vals,
                                                    curvegeom.curve)
        for k in range(cfg.NUM_T):
            section_psi = np.array([p[k] * 360 for p in psi_points])
            section_radius = np.array([r[k] for r in radius_points])
            curvegeompol = build_sketch_sector(section_psi,
                                                section_radius,
                                                2,
                                                np.ones(len(psi_points)),
                                                cfg.NUM_P)
            geom.ctrl_x_collections.append(curvegeompol.ctrl_xp)
            geom.ctrl_y_collections.append(curvegeompol.ctrl_yp)
            geom.x_collections.append(curvegeompol.x_p)
            geom.y_collections.append(curvegeompol.y_p)
            moved_points = rotate_poloidal_section([curvegeompol.x_p, curvegeompol.y_p,
                                                    [0.0]*len(curvegeompol.x_p)],
                                                toroid_geom.toroidal_coordinates[k],
                                                toroid_geom.toroidal_tangents[k], 0.0)
            geom.x_moved_collections.append(moved_points[0])
            geom.y_moved_collections.append(moved_points[1])
            geom.z_moved_collections.append(moved_points[2])

            moved_ctrl_points = rotate_poloidal_section([curvegeompol.ctrl_xp, curvegeompol.ctrl_yp,
                                                     [0.0]*len(curvegeompol.ctrl_xp)],
                                               toroid_geom.toroidal_coordinates[i],
                                               toroid_geom.toroidal_tangents[i], 0.0)
            x_moved_ctrl_collections.append(moved_ctrl_points[0])
            y_moved_ctrl_collections.append(moved_ctrl_points[1])
            z_moved_ctrl_collections.append(moved_ctrl_points[2])
    return geom.x_collections, geom.y_collections, \
            geom.x_moved_collections ,geom.y_moved_collections, geom.z_moved_collections, \
            geom.ctrl_x_collections, geom.ctrl_y_collections, \
            toroidal_geom.toroidal_coordinates, toroidal_geom.toroidal_tangents, \
            *curvegeom.curve, *curvegeom.ctrl, \
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
    geom = GeometryData([], [], [], [], [], [], [])
    toroidgeom = ToroidalGeometry([],[])
    poloidgeom = PoloidalGeometry([],[],[],[],[],[])
    geom.x_collections, geom.y_collections, \
        geom.x_moved_collections ,geom.y_moved_collections, geom.z_moved_collections, \
            geom.ctrl_x_collections, geom.ctrl_y_collections, \
            toroidgeom.toroidal_coordinates, toroidgeom.toroidal_tangents, \
                 poloidgeom.x, poloidgeom.y, poloidgeom.z, \
                    poloidgeom.ctrl_x, poloidgeom.ctrl_y, poloidgeom.ctrl_z,\
                    x_moved_ctrl_collections, y_moved_ctrl_collections, z_moved_ctrl_collections\
                          = geometry_preprocess(toroidal_sections,
                                                poloidal_sections)
    elongation_list, file_list, guide_vane_collections = geometry_elongation(poloidal_sections,
                                                                [geom.x_moved_collections,
                                                                geom.y_moved_collections,
                                                                geom.z_moved_collections],
                                                                toroidgeom.toroidal_tangents,
                                                                toroidgeom.toroidal_coordinates,
                                                                plot=True)
    
    if plot:
        # merge_stls(file_list, "./typethree.stl")
        plot_typethree([geom.x_collections, geom.y_collections],
                        [geom.ctrl_x_collections, geom.ctrl_y_collections],
                        [poloidgeom.x, poloidgeom.y, poloidgeom.z],
                        [poloidgeom.ctrl_x, poloidgeom.ctrl_y, poloidgeom.ctrl_z],
                        [geom.x_moved_collections, geom.y_moved_collections,
                         geom.z_moved_collections], 
                         [x_moved_ctrl_collections, y_moved_ctrl_collections, z_moved_ctrl_collections],
                         toroidgeom.toroidal_coordinates,
                  guide_vane_collections, "./typethree.stl",
                  filename=filename)
    return 0

x0, data_format = gf.linearize_data("./inputs/toroidal_section.json")
gp.FORMAT = data_format
toroidal_sections, poloidal_sections = gf.delinearize_data(x0)
geometry_construct(toroidal_sections, poloidal_sections,
                          1, plot=True, filename="./typethree_geometry")

def check_clash(array):
    return np.all(np.diff(array, axis=1) >= 0)

psi = []
radius = []
phi = []
for i in range(4):
    filename = "./inputs/poloidal_section_" + str(i + 1) + ".json"
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
        nval = data.get("N_s")
        psi.append(np.array(data.get("psi")) / 360.0)
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
radius_nurbs = []
weights = [1.0, 1.0, 1.0, 1.0]  # Example weights for the control points
degree = 2
for i in range(len(psi[0])):
    psi_0 = [p[i] for p in psi]
    ax_3.plot(phi[0], psi_0, label=r"$\psi_" + str(i + 1) + "$", marker='o', 
              linewidth = 0.8)
    psi_nurbs.append([[phi[0][0],psi_0[0]],
                      [phi[0][1],psi_0[1]],
                           [phi[0][2],psi_0[2]],
                           [phi[0][3],psi_0[3]],
                           [1.0,psi_0[0]]])

    curve_points = np.array([nurbs_gen_periodic(psi_nurbs[i], 
                            [1.0, 5.0, 5.0, 5.0, 1.0], 2, u) for u in u_vals])
    ax_3.plot(curve_points[:, 0], curve_points[:, 1],
              linestyle='--', color = cfg.color[i], 
              linewidth = 0.8)
    
    radius_0 = [r[i] for r in radius]
    ax_4.plot(phi[0], radius_0, label=r"$r_" + str(i + 1) + "$", marker='o')
    radius_nurbs.append([[phi[0][0],radius_0[0]],
                      [phi[0][1],radius_0[1]],
                           [phi[0][2],radius_0[2]],
                           [phi[0][3],radius_0[3]],
                           [1.0,radius_0[0]]])

    radius_points = np.array([nurbs_gen(radius_nurbs[i], 
                            [1.0, 5.0, 5.0, 5.0, 1.0], 2, u) for u in u_vals])
    ax_4.plot(radius_points[:, 0][:-1], radius_points[:, 1][:-1],
              linestyle='--', color = cfg.color[i], 
              linewidth = 0.8)

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

print(check_clash(psi))
print(check_clash(psi_deltas))

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