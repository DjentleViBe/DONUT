import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from geometry import geometry_process as gp
from geometry.geometry_plotter import plot_geometry
from geometry.geometry_build import merge_stls
from launch_geometry import geometry_elongation
from geometry import geometry_fourier as gf
import matplotlib.pyplot as plt
import numpy as np
import config as cfg
from geometry.geometry_sector import build_sketch_sector, build_sketch_sector_toroidal, \
                    get_toroidal_coordinates_tangent, build_guide_vane
from geometry.geometry_reader import get_poloidal_sections_from_toroidal_file, \
                            get_geometry_parameters_from_poloidal_file,\
                            get_geometry_parameters_from_toroidal_file
from geometry.geometry_operations import rotate_poloidal_section

def plot_typetwo(points_2d, ctrl_2d,
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
    ax3.text(0.3, 1.0, 0, r'$\mathbf{T_1}$($r_{t_1}, \theta_{1}, \phi_{1}$)')
    ax3.text(0.9, 0.3, -0.08, r'$P_1$($r_{p_1}, \psi_{1}$)', color=cfg.color[3])
    ax3.text(0.7, 0.3, -0.1, r'$\mathbf{Q}$')
    ax3.text(0.0, -0.8, -0.1, r'$\alpha$')
    # loop around the line
    theta = np.linspace(0, 1.6*np.pi, 200)
    r = 0.1
    z0 = 1.1
    z_loop = r * np.cos(theta)
    y_loop = -0.7 + r * np.sin(theta)
    x_loop = -1.0 + np.full_like(theta, z0)

    ax3.plot(x_loop, y_loop, z_loop, color = 'k')

    # tangent direction at end
    p0 = np.array([x_loop[-2], y_loop[-2], z_loop[-2]])
    p1 = np.array([x_loop[-1], y_loop[-1], z_loop[-1]])
    d = p1 - p0
    d = d / np.linalg.norm(d)   # normalize
    # arrow length
    L = 0.05
    ax3.quiver(
        p0[0], p0[1], p0[2],
        d[0], d[1], d[2],
        length=L,
        normalize=True,
        arrow_length_ratio=1.0,
        color='k'
    )
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
    curve, ctrl = build_sketch_sector_toroidal(toroidal_sections['theta'],
                                            toroidal_sections['phi'],
                                            toroidal_sections['radius'],
                                            toroidal_sections['degree'],
                                            toroidal_sections['weights'])
    x, y, z = curve
    ctrl_x, ctrl_y, ctrl_z = ctrl
    toroidal_coordinates, toroidal_tangents = get_toroidal_coordinates_tangent(toroidal_sections['sections'],
                                                        [x, y, z])
    for i, poloidal_file in enumerate(poloidal_sections):
        if mode == 0:
            poloid_file = get_geometry_parameters_from_poloidal_file("./inputs/" +
                                                    poloidal_file + ".json")
            x_p, y_p, ctrl_xp, ctrl_yp = build_sketch_sector(poloid_file['theta'],
                                                    poloid_file['radius'],
                                                    poloid_file['degree'],
                                                    poloid_file['weights'])
        else:
            x_p, y_p, ctrl_xp, ctrl_yp = build_sketch_sector(poloidal_file['psi'],
                                                    poloidal_file['radius'],
                                                    poloidal_file['degree'],
                                                    poloidal_file['weights'])
        x_collections.append(x_p)
        y_collections.append(y_p)
        ctrl_x_collections.append(ctrl_xp)
        ctrl_y_collections.append(ctrl_yp)
        # moved_points = move_poloidal_section_origin([x, y], toroidal_coordinates[i])
        moved_points = rotate_poloidal_section([x_p, y_p, [0.0]*len(x_p)],
                                               toroidal_coordinates[i],
                                               toroidal_tangents[i])
        moved_ctrl_points = rotate_poloidal_section([ctrl_xp, ctrl_yp,
                                                     [0.0]*len(ctrl_xp)],
                                               toroidal_coordinates[i],
                                               toroidal_tangents[i])
        x_moved_collections.append(moved_points[0])
        y_moved_collections.append(moved_points[1])
        z_moved_collections.append(moved_points[2])

        x_moved_ctrl_collections.append(moved_ctrl_points[0])
        y_moved_ctrl_collections.append(moved_ctrl_points[1])
        z_moved_ctrl_collections.append(moved_ctrl_points[2])
    return x_collections, y_collections, \
        x_moved_collections ,y_moved_collections, z_moved_collections, \
            ctrl_x_collections, ctrl_y_collections, \
            toroidal_coordinates, toroidal_tangents, \
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
                                                                x_moved_collections,
                                                                y_moved_collections,
                                                                z_moved_collections, \
                                                                toroidal_tangents, plot=True)
    if plot:
        # merge_stls(file_list, "./typetwo.stl")
        plot_typetwo([x_collections, y_collections], [ctrl_x_collections, ctrl_y_collections],
                  [x, y, z], [ctrl_x, ctrl_y, ctrl_z],
                  [x_moved_collections, y_moved_collections, z_moved_collections],
                  [x_moved_ctrl_collections, y_moved_ctrl_collections, z_moved_ctrl_collections],
                  toroidal_coordinates,
                  guide_vane_collections, "./typetwo.stl",
                  filename=filename)
    return 0

x0, data_format = gf.linearize_data("./inputs/toroidal_section.json")
gp.FORMAT = data_format
toroidal_sections, poloidal_sections = gf.delinearize_data(x0)
geometry_construct(toroidal_sections, poloidal_sections,
                          1, plot=True, filename="./typetwo_geometry")
