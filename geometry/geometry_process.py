from launch_geometry import geometry_construct, geometry_calculate
from geometry.geometry_reader import linearize_data, delinearize_data
import numpy as np

def geometry_process():
    #poloidal_sections, toroidal_sections = geometry_init()
    x0, format = linearize_data("./inputs/toroidal_section.json")
    toroidal_sections, poloidal_sections = delinearize_data(x0, format)
    elongation = geometry_construct(toroidal_sections, poloidal_sections, 1)

def geometry_process_optimization(x0, format):
    toroidal_sections, poloidal_sections = delinearize_data(x0, format)
    elongation = geometry_calculate(toroidal_sections, poloidal_sections)
    # print("RAW x:", x0[:5])
    # print("objective:", elongation)
    return elongation
    