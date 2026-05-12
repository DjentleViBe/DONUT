import numpy as np
from launch_geometry import geometry_construct, geometry_calculate
from geometry.geometry_reader import linearize_data, delinearize_data

CURRENT_ELONGATION = None
BEST_ELONGATION = np.inf

def geometry_process():
    global CURRENT_ELONGATION
    #poloidal_sections, toroidal_sections = geometry_init()
    x0, format = linearize_data("./inputs/toroidal_section.json")
    toroidal_sections, poloidal_sections = delinearize_data(x0, format)
    CURRENT_ELONGATION = geometry_construct(toroidal_sections, poloidal_sections, 1)

def geometry_process_optimization(x0, format):
    global CURRENT_ELONGATION
    global BEST_ELONGATION
    toroidal_sections, poloidal_sections = delinearize_data(x0, format)
    CURRENT_ELONGATION = geometry_calculate(toroidal_sections, poloidal_sections)
    # track best value seen so far
    if BEST_ELONGATION is None or CURRENT_ELONGATION < BEST_ELONGATION:
        BEST_ELONGATION = CURRENT_ELONGATION
    print(f"\t\tRAW x: {x0[:5]}", f"Objective: {CURRENT_ELONGATION}")
    return CURRENT_ELONGATION
    