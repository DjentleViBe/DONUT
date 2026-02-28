"""Main module for DONUT geometry processing and visualization.
This module serves as the entry point for the DONUT project. 
It orchestrates the reading of geometry parameters from input files, 
building the sketches of the sectors, and plotting the poloidal cross 
section of the geometry.
"""
import numpy as np
from geometry.geometry_reader import get_poloidal_sections_from_toroidal_file, \
                            get_geometry_parameters_from_poloidal_file,\
                            get_geometry_parameters_from_toroidal_file
from geometry.geometry_sector import build_sketch_sector, build_sketch_sector_toroidal, \
                    get_toroidal_coordinates_tangent, build_guide_vane
from geometry.geometry_plotter import plot_geometry
from geometry.geometry_operations import rotate_poloidal_section
from geometry.geometry_build import loft_revolved, write_stl, merge_stls

if __name__ == "__main__":
    poloidal_sections = get_poloidal_sections_from_toroidal_file("./inputs/toroidal_section.json")
    x_collections = []
    y_collections = []
    x_moved_collections = []
    y_moved_collections = []
    z_moved_collections = []
    ctrl_x_collections = []
    ctrl_y_collections = []
    toroid_file = get_geometry_parameters_from_toroidal_file("./inputs/toroidal_section.json")
    (x, y, z), (ctrl_x, ctrl_y, ctrl_z) = build_sketch_sector_toroidal(
                                                        toroid_file['theta'],
                                                        toroid_file['phi'],
                                                        toroid_file['radius'],
                                                        toroid_file['degree'],
                                                        toroid_file['weights'])
    toroidal_coordinates, toroidal_tangents = get_toroidal_coordinates_tangent(
                                                        toroid_file['sections'],
                                                        [x, y, z])
    for i, poloidal_file in enumerate(poloidal_sections):
        poloid_file = get_geometry_parameters_from_poloidal_file("./inputs/" +
                                                    poloidal_file + ".json")
        x_p, y_p, ctrl_xp, ctrl_yp = build_sketch_sector(poloid_file['theta'],
                                                   poloid_file['radius'],
                                                   poloid_file['degree'],
                                                   poloid_file['weights'])
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
    guide_vane_collections = []
    file_list = []
    N = len(poloidal_sections)
    for i in range(N):
        next_i = (i + 1) % N
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
        vertices, faces = loft_revolved(np.asarray(guide_vane_collections[i]))
        write_stl(vertices, faces, f"./outputs/revolved_surface+{i}.stl")
        file_list.append(f"./outputs/revolved_surface+{i}.stl")
    merge_stls(file_list, "./outputs/combined.stl")
    plot_geometry([x_collections, y_collections], [ctrl_x_collections, ctrl_y_collections],
                  [x, y, z], [ctrl_x, ctrl_y, ctrl_z],
                  [x_moved_collections, y_moved_collections, z_moved_collections],
                  guide_vane_collections, "./outputs/combined.stl")
