"""Main module for DONUT geometry processing and visualization.
This module serves as the entry point for the DONUT project. 
It orchestrates the reading of geometry parameters from input files, 
building the sketches of the sectors, and plotting the poloidal cross 
section of the geometry.
"""
import numpy as np
import config as cfg
from scipy.optimize import minimize
from geometry.geometry_process import geometry_process_optimization
from geometry.geometry_reader import delinearize_data, linearize_data
from launch_geometry import geometry_calculate, geometry_construct

ITERATION = 0  # external counter
ELONGATION = 0.0

def callback(xk):
    """callback function to be called after each optimization iteration. 
    It logs the current iteration number."""
    global ITERATION
    ITERATION += 1
    print(f"\nIteration {ITERATION}: Max elongation = {ELONGATION}, xk norm = {np.linalg.norm(xk)}")

if __name__ == "__main__":
    x0, data_format = linearize_data("./inputs/toroidal_section.json")
    toroidal_sections, poloidal_sections = delinearize_data(x0, data_format)
    geometry_construct(toroidal_sections, poloidal_sections, 1, plot=True,
                       filename="initial_geometry")

    result = minimize(geometry_process_optimization, x0,
                      args=(data_format,), method=cfg.METHOD,
                      options={'maxiter': cfg.MAX_ITER},
                      callback=callback)
    print("Optimization result:", result)
    toroidal_sections, poloidal_sections = delinearize_data(result.x, data_format)
    geometry_construct(toroidal_sections, poloidal_sections, 1, plot=True,
                       filename="optimized_geometry")
