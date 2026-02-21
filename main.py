"""Main module for DONUT geometry processing and visualization.
This module serves as the entry point for the DONUT project. 
It orchestrates the reading of geometry parameters from input files, 
building the sketches of the sectors, and plotting the poloidal cross 
section of the geometry.
"""
from geometry_reader import get_poloidal_sections_from_toroidal_file, \
                            get_geometry_parameters_from_poloidal_file,\
                            get_geometry_parameters_from_toroidal_file
from sector import build_sketch_sector, build_sketch_sector_toroidal
from geometry_plotter import plot_geometry

if __name__ == "__main__":
    poloidal_sections = get_poloidal_sections_from_toroidal_file("./inputs/toroidal_section.json")
    x_collections = []
    y_collections = []
    ctrl_x_collections = []
    ctrl_y_collections = []
    for poloidal_file in poloidal_sections:
        poloid_file = get_geometry_parameters_from_poloidal_file("./inputs/" +
                                                    poloidal_file + ".json")
        x, y, ctrl_x, ctrl_y = build_sketch_sector(poloid_file['theta'],
                                                   poloid_file['radius'],
                                                   poloid_file['degree'],
                                                   poloid_file['weights'])
        x_collections.append(x)
        y_collections.append(y)
        ctrl_x_collections.append(ctrl_x)
        ctrl_y_collections.append(ctrl_y)
    toroid_file = get_geometry_parameters_from_toroidal_file("./inputs/toroidal_section.json")
    (x, y, z), (ctrl_x, ctrl_y, ctrl_z) = build_sketch_sector_toroidal(toroid_file['theta'],
                                                                   toroid_file['phi'],
                                                                   toroid_file['radius'],
                                                                    toroid_file['degree'],
                                                                    toroid_file['weights'])
    plot_geometry([x_collections, y_collections], [ctrl_x_collections, ctrl_y_collections],
                  [x, y, z], [ctrl_x, ctrl_y, ctrl_z])
