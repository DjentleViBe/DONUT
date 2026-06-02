"""
This module serves as the main entry point for the geometry construction process. 
It reads the input geometry data, constructs the geometry based on the 
provided parameters, and generates the corresponding STL 
files for visualization and further processing. 
The module also computes the elongation of the guide vanes 
and plots the geometry for verification.
"""
import numpy as np
import config as cfg
import geometry.geometry_process as gp
from geometry.geometry_reader import get_poloidal_sections_from_toroidal_file, \
                            get_geometry_parameters_from_toroidal_file
from geometry.geometry_sector import build_sketch_sector, build_sketch_sector_toroidal, \
                    get_toroidal_coordinates_tangent, build_guide_vane
from geometry.geometry_plotter import plot_geometry
from geometry.geometry_operations import rotate_poloidal_section
from geometry.geometry_build import loft_revolved, write_stl, merge_stls
from objectives import compute_elongation_fit, compute_average_triangularity, \
                        compute_average_aspect_ratio
from classes_geometry import GeometryData, \
                            PoloidalGeometry, \
                            ToroidalGeometry
from geometry.geometry_operations import nurbs_gen
# from scipy.special import logsumexp

def softmax_max(x, beta=10.0):
    """Compute a smooth approximation of the maximum value in the 
    list x using the softmax function."""
    x = np.asarray(x)
    x = x - np.max(x)  # numerical stability
    return np.log(np.sum(np.exp(beta * x))) / beta

def geometry_init():
    """
    Initialize the geometry by reading the input files and extracting the necessary parameters.
    """
    poloidal_sections = get_poloidal_sections_from_toroidal_file("./inputs/toroidal_section.json")
    toroid_file = get_geometry_parameters_from_toroidal_file("./inputs/toroidal_section.json")
    return poloidal_sections, toroid_file

def geometry_preprocess(toroidal_sections, poloidal_sections):
    """
    Preprocess the geometry data by building the sketches of the toroidal and poloidal sections.
    This function prepares the data for further processing and optimization.
    """
    geom = GeometryData([], [], [], [], [], [], [])
    curvegeom = build_sketch_sector_toroidal(toroidal_sections['theta'],
                                            toroidal_sections['phi'],
                                            toroidal_sections['radius'],
                                            toroidal_sections['degree'],
                                            toroidal_sections['weights'],
                                            cfg.NUM_T)
    toroidal_geom = get_toroidal_coordinates_tangent(toroidal_sections['sections'],
                                                    curvegeom.curve)
    if cfg.STUDY_NAME == "Type2":
        twist_params = (
            toroidal_sections['N'],
            toroidal_sections['A'],
            toroidal_sections['k'])
    else:
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
        else:
            geom.x_collections.append(curvegeompol.x_p)
            geom.y_collections.append(curvegeompol.y_p)
            moved_points = rotate_poloidal_section([curvegeompol.x_p, curvegeompol.y_p,
                                                    [0.0]*len(curvegeompol.x_p)],
                                                toroidal_geom.toroidal_coordinates[i],
                                                toroidal_geom.toroidal_tangents[i],
                                                twist = twist_params[2] * i *
                                                (2 * np.pi / cfg.NUM_T) +
                                                twist_params[1] * np.sin(twist_params[0]
                                                    * i * (2 * np.pi / cfg.NUM_T)))
            geom.x_moved_collections.append(moved_points[0])
            geom.y_moved_collections.append(moved_points[1])
            geom.z_moved_collections.append(moved_points[2])
    if cfg.STUDY_NAME == "Type3":
        psi_nurbs = []
        radius_nurbs = []
        psi_points = []
        radius_points = []
        u_vals = np.linspace(0, 1, cfg.NUM_T + 1)
        for i in range(len(psi_collect[0])):
            psi_0 = [p[i] / 360.0 for p in psi_collect]
            radius_0 = [r[i] for r in radius_collect]
            psi_nurbs.append([[toroidal_sections['sections'][0], psi_0[0]],
                            [toroidal_sections['sections'][1], psi_0[1]],
                            [toroidal_sections['sections'][2], psi_0[2]],
                            [toroidal_sections['sections'][3], psi_0[3]],
                            [1.0, psi_0[0]]])
            radius_nurbs.append([[toroidal_sections['sections'][0], radius_0[0]],
                            [toroidal_sections['sections'][1], radius_0[1]],
                            [toroidal_sections['sections'][2], radius_0[2]],
                            [toroidal_sections['sections'][3], radius_0[3]],
                            [1.0, radius_0[0]]])
            curve_points_psi = np.array([nurbs_gen(psi_nurbs[i], 
                                [1.0, 5.0, 5.0, 5.0, 1.0], 2, u) for u in u_vals])
            psi_points.append(curve_points_psi[:, 1])
            curve_points_radius = np.array([nurbs_gen(radius_nurbs[i], 
                                [1.0, 5.0, 5.0, 5.0, 1.0], 2, u) for u in u_vals])
            radius_points.append(curve_points_radius[:, 1])
        toroid_geom = get_toroidal_coordinates_tangent(u_vals, curvegeom.curve)
        for k in range(cfg.NUM_T):
            section_psi = np.array([p[k] * 360 for p in psi_points])
            section_radius = np.array([r[k] for r in radius_points])
            curvegeompol = build_sketch_sector(section_psi,
                                                section_radius,
                                                poloidal_file['degree'],
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
    return geom.x_collections, geom.y_collections, \
            geom.x_moved_collections ,geom.y_moved_collections, geom.z_moved_collections, \
            geom.ctrl_x_collections, geom.ctrl_y_collections, \
            toroidal_geom.toroidal_coordinates, toroidal_geom.toroidal_tangents, \
            *curvegeom.curve, *curvegeom.ctrl

def geometry_elongation(poloidal_sections, moved_collections,
                        toroidal_tangents, toroidal_coordinates,
                        plot=False):
    """Computes elongation from coordinates"""
    guide_vane_collections = []
    file_list = []
    elongation_list = []
    if cfg.STUDY_NAME == "Type1":
        for i in range(len(poloidal_sections)):
            next_i = (i + 1) % len(poloidal_sections)
            guide_vane_collections.append(
                build_guide_vane(
                    [moved_collections[0][i],
                    moved_collections[1][i],
                    moved_collections[2][i]],

                    [moved_collections[0][next_i],
                    moved_collections[1][next_i],
                    moved_collections[2][next_i]],

                    toroidal_tangents[i],
                    toroidal_tangents[next_i],

                    toroidal_coordinates[i],
                    toroidal_coordinates[next_i],
                    cfg.NUM_GV
                )
            )
            if plot:
                vertices, faces = loft_revolved(np.asarray(guide_vane_collections[i]))
                write_stl(vertices, faces,
                f"./outputs/{cfg.STUDY_NAME}_{cfg.METHOD}_revolved_surface+{i}.stl")
                file_list.append(
                    f"./outputs/{cfg.STUDY_NAME}_{cfg.METHOD}_revolved_surface+{i}.stl")
            # epsilon_max = logsumexp(cfg.K_SMOOTH * np.array(vals)) / cfg.K_SMOOTH
            elongation_list.append(
                max(
                    compute_elongation_fit(np.asarray(guide_vane_collections[i])[:, j, :])
                    for j in range(cfg.NUM_GV)
                    )
                )
    elif cfg.STUDY_NAME == "Type2" or cfg.STUDY_NAME == "Type3":
        for i in range(cfg.NUM_T):
            next_i = (i + 1) % cfg.NUM_T
            current_section = np.stack([moved_collections[0][i],
                                        moved_collections[1][i],
                                        moved_collections[2][i]], axis=-1)
            next_section = np.stack([moved_collections[0][next_i],
                                     moved_collections[1][next_i],
                                     moved_collections[2][next_i]], axis=-1)

            section_pair = np.stack([current_section, next_section], axis=1)

            guide_vane_collections.append(section_pair)
            if plot:
                vertices, faces = loft_revolved(np.asarray(guide_vane_collections[i]))
                write_stl(vertices, faces,
                f"./outputs/{cfg.STUDY_NAME}_{cfg.METHOD}_revolved_surface+{i}.stl")
                file_list.append(
                    f"./outputs/{cfg.STUDY_NAME}_{cfg.METHOD}_revolved_surface+{i}.stl")
            # epsilon_max = logsumexp(cfg.K_SMOOTH * np.array(vals)) / cfg.K_SMOOTH
            elongation_list.append(
                max(
                    compute_elongation_fit(np.asarray(guide_vane_collections[i])[:, j, :])
                    for j in range(2)
                    )
                )
    return elongation_list, file_list, guide_vane_collections

def geometry_construct(toroidal_sections, poloidal_sections,
                       init=False, plot=False, filename=None):
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
                    poloidgeom.ctrl_x, poloidgeom.ctrl_y, poloidgeom.ctrl_z\
                          = geometry_preprocess(toroidal_sections,
                                                poloidal_sections)
    elongation_list, file_list, guide_vane_collections = geometry_elongation(poloidal_sections,
                                                                [geom.x_moved_collections,
                                                                geom.y_moved_collections,
                                                                geom.z_moved_collections],
                                                                toroidgeom.toroidal_tangents,
                                                                toroidgeom.toroidal_coordinates,
                                                                plot=True)
    # gp.TRIAL_ELONGATION = np.percentile(elongation_list, 95)
    ceval = max(elongation_list)
    ctval = compute_average_triangularity(geom.x_collections, geom.y_collections)
    arval = compute_average_aspect_ratio(geom.x_moved_collections,
                                      geom.y_moved_collections,
                                      geom.z_moved_collections)
    if init:
        gp.TRIAL_ELONGATION = ceval
        gp.CURRENT_AR = arval
        gp.CURRENT_TRIANGULARITY = ctval
    if plot:
        merge_stls(file_list, "./outputs/" + cfg.STUDY_NAME + "_" +
                    cfg.METHOD + "_" + filename + ".stl")
        plot_geometry([geom.x_collections, geom.y_collections],
                        [geom.ctrl_x_collections, geom.ctrl_y_collections],
                        [poloidgeom.x, poloidgeom.y, poloidgeom.z],
                        [poloidgeom.ctrl_x, poloidgeom.ctrl_y, poloidgeom.ctrl_z],
                        [geom.x_moved_collections, geom.y_moved_collections,
                         geom.z_moved_collections],
                        guide_vane_collections, "./outputs/" + cfg.STUDY_NAME +
                        "_" + cfg.METHOD + "_" + filename + ".stl",
                        filename=filename)
    return ceval, ctval, arval

def geometry_optimise(toroidal_sections, poloidal_sections):
    """
    Optimize the geometry by adjusting the parameters of the toroidal and poloidal sections.
    This function can be used to minimize the elongation or other objective 
    functions related to the geometry.
    """
    _, _, \
        x_moved_collections ,y_moved_collections, z_moved_collections,\
            _, _, toroidal_coordinates, toroidal_tangents,\
                _, _, _, _, _, _ = geometry_preprocess(toroidal_sections,
                                                        poloidal_sections)

    elongation_list, _, _ = geometry_elongation(poloidal_sections,
                                            [x_moved_collections,
                                            y_moved_collections,
                                            z_moved_collections],
                                            toroidal_tangents, toroidal_coordinates,
                                            plot=False)
    # gp.TRIAL_ELONGATION = np.percentile(elongation_list, 95)
    gp.TRIAL_ELONGATION = max(elongation_list)
    return gp.TRIAL_ELONGATION

def geometry_constraint(toroidal_sections, poloidal_sections):
    """
    Compute the constraints for the geometry optimization problem.
    This function can be used to enforce certain geometric properties or
    limitations during the optimization process.
    """
    x_collections, y_collections, \
        x_moved_collections ,y_moved_collections, z_moved_collections, \
            _, _, _, _, _, _, _, _, _, _ = geometry_preprocess(toroidal_sections,
                                                                               poloidal_sections)

    gp.CURRENT_TRIANGULARITY = compute_average_triangularity(x_collections, y_collections)
    gp.CURRENT_AR = compute_average_aspect_ratio(x_moved_collections,
                                                 y_moved_collections,
                                                 z_moved_collections)
    return gp.CURRENT_TRIANGULARITY, gp.CURRENT_AR

def geometry_pipeline(toroidal_sections, poloidal_sections):
    """
    Construct the geometry based on the provided toroidal and poloidal sections.
    Args:
    toroidal_sections: A dictionary containing the parameters for the toroidal section.
    poloidal_sections: A list of dictionaries containing the parameters for each poloidal section.
    mode: An integer indicating the mode of operation (0 for reading from files, 
            1 for using provided data).
    """
    trial_elongation = geometry_optimise(toroidal_sections, poloidal_sections)
    current_triangularity, current_ar = geometry_constraint(toroidal_sections, poloidal_sections)

    return trial_elongation, current_triangularity, current_ar
