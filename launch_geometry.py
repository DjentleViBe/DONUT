"""
This module serves as the main entry point for the geometry construction process. 
It reads the input geometry data, constructs the geometry based on the 
provided parameters, and generates the corresponding STL 
files for visualization and further processing. 
The module also computes the elongation of the guide vanes 
and plots the geometry for verification.
"""
import numpy as np
from scipy.special import logsumexp
import config as cfg
import geometry.geometry_process as gp
from geometry.geometry_reader import get_poloidal_sections_from_toroidal_file, \
                            get_geometry_parameters_from_poloidal_file,\
                            get_geometry_parameters_from_toroidal_file
from geometry.geometry_sector import build_sketch_sector, build_sketch_sector_toroidal, \
                    get_toroidal_coordinates_tangent, build_guide_vane
from geometry.geometry_plotter import plot_geometry
from geometry.geometry_operations import rotate_poloidal_section
from geometry.geometry_build import loft_revolved, write_stl, merge_stls
from objectives import compute_elongation_fit, compute_average_triangularity, \
                        compute_average_aspect_ratio

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
    ctrl_x_collections = []
    ctrl_y_collections = []
    curve, ctrl = build_sketch_sector_toroidal(toroidal_sections['theta'],
                                            toroidal_sections['phi'],
                                            toroidal_sections['radius'],
                                            toroidal_sections['degree'],
                                            toroidal_sections['weights'])
    x, y, z = curve
    ctrl_x, ctrl_y, ctrl_z = ctrl
    toroidal_coordinates, toroidal_tangents = get_toroidal_coordinates_tangent(
                                                        toroidal_sections['sections'],
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
        x_moved_collections.append(moved_points[0])
        y_moved_collections.append(moved_points[1])
        z_moved_collections.append(moved_points[2])
    return x_collections, y_collections, \
        x_moved_collections ,y_moved_collections, z_moved_collections, \
            ctrl_x_collections, ctrl_y_collections, \
            toroidal_coordinates, toroidal_tangents, \
            x, y, z, ctrl_x, ctrl_y, ctrl_z

def geometry_elongation(poloidal_sections, x_moved_collections,
                        y_moved_collections, z_moved_collections,
                        toroidal_tangents, plot=False):
    guide_vane_collections = []
    file_list = []
    elongation_list = []
    nval = len(poloidal_sections)
    for i in range(nval):
        next_i = (i + 1) % nval
        guide_vane_collections.append(
            build_guide_vane(
                [x_moved_collections[i],
                y_moved_collections[i],
                z_moved_collections[i]],

                [x_moved_collections[next_i],
                y_moved_collections[next_i],
                z_moved_collections[next_i]],

                toroidal_tangents[i],
                toroidal_tangents[next_i]
            )
        )
        if plot:
            vertices, faces = loft_revolved(np.asarray(guide_vane_collections[i]))
            write_stl(vertices, faces,
             f"./outputs/{cfg.STUDY_NAME}_{cfg.METHOD}_revolved_surface+{i}.stl")
            file_list.append(f"./outputs/{cfg.STUDY_NAME}_{cfg.METHOD}_revolved_surface+{i}.stl")
        vals = [compute_elongation_fit(np.asarray(guide_vane_collections[i])[:, j, :])
                for j in range(100)]
        # epsilon_max = logsumexp(cfg.K_SMOOTH * np.array(vals)) / cfg.K_SMOOTH
        epsilon_max = max(vals)
        elongation_list.append(epsilon_max)
    return elongation_list, file_list, guide_vane_collections
    
def geometry_construct(toroidal_sections, poloidal_sections, mode, plot=False, filename=None):
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
                 x, y, z, ctrl_x, ctrl_y, ctrl_z = geometry_preprocess(toroidal_sections, poloidal_sections, mode)

    elongation_list, file_list, guide_vane_collections = geometry_elongation(poloidal_sections, \
                                          x_moved_collections, y_moved_collections, z_moved_collections, \
                                            toroidal_tangents, plot=True)
    # gp.CURRENT_ELONGATION = np.percentile(elongation_list, 95)
    gp.CURRENT_ELONGATION = max(elongation_list)
    gp.CURRENT_TRIANGULARITY = compute_average_triangularity(x_collections, y_collections)
    gp.CURRENT_AR = compute_average_aspect_ratio(x_moved_collections, y_moved_collections, z_moved_collections)

    if plot:
        merge_stls(file_list, "./outputs/" + cfg.STUDY_NAME + "_" + cfg.METHOD + "_" + filename + ".stl")
        plot_geometry([x_collections, y_collections], [ctrl_x_collections, ctrl_y_collections],
                  [x, y, z], [ctrl_x, ctrl_y, ctrl_z],
                  [x_moved_collections, y_moved_collections, z_moved_collections],
                  guide_vane_collections, "./outputs/" + cfg.STUDY_NAME + "_" + cfg.METHOD + "_" + filename + ".stl",
                  filename=filename)
    return gp.CURRENT_ELONGATION

def geometry_optimise(toroidal_sections, poloidal_sections):
    """
    Optimize the geometry by adjusting the parameters of the toroidal and poloidal sections.
    This function can be used to minimize the elongation or other objective functions related to the geometry.
    """
    x_collections, y_collections, \
        x_moved_collections ,y_moved_collections, z_moved_collections, \
            _, _, _, toroidal_tangents, _, _, _, _, _, _ = geometry_preprocess(toroidal_sections, poloidal_sections, mode=1)

    elongation_list, _, _ = geometry_elongation(poloidal_sections, \
                                          x_moved_collections, y_moved_collections, z_moved_collections, \
                                            toroidal_tangents, plot=False)
    # gp.CURRENT_ELONGATION = np.percentile(elongation_list, 95)
    gp.CURRENT_ELONGATION = max(elongation_list)
    return gp.CURRENT_ELONGATION

def geometry_constraint(toroidal_sections, poloidal_sections):
    """
    Compute the constraints for the geometry optimization problem.
    This function can be used to enforce certain geometric properties or limitations during the optimization process.
    """
    x_collections, y_collections, \
        x_moved_collections ,y_moved_collections, z_moved_collections, \
            _, _, _, toroidal_tangents, _, _, _, _, _, _ = geometry_preprocess(toroidal_sections, poloidal_sections, mode=1)

    elongation_list, _, _ = geometry_elongation(poloidal_sections, \
                                          x_moved_collections, y_moved_collections, z_moved_collections, \
                                            toroidal_tangents, plot=False)
    # gp.CURRENT_ELONGATION = np.percentile(elongation_list, 95)
    gp.CURRENT_TRIANGULARITY = compute_average_triangularity(x_collections, y_collections)
    gp.CURRENT_AR = compute_average_aspect_ratio(x_moved_collections, y_moved_collections, z_moved_collections)
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
    CURRENT_ELONGATION = geometry_optimise(toroidal_sections, poloidal_sections)
    CURRENT_TRIANGULARITY, CURRENT_AR = geometry_constraint(toroidal_sections, poloidal_sections)

    return CURRENT_ELONGATION, CURRENT_TRIANGULARITY, CURRENT_AR
