"""Main module for DONUT geometry processing and visualization.
This module serves as the entry point for the DONUT project. 
It orchestrates the reading of geometry parameters from input files, 
building the sketches of the sectors, and plotting the poloidal cross 
section of the geometry.
"""
import numpy as np
from geometry.geometry_process import geometry_process_optimization
from geometry.geometry_reader import delinearize_data, linearize_data
from scipy.optimize import minimize
from launch_geometry import geometry_calculate, geometry_construct

iteration = 0  # external counter

def callback(xk):
    global iteration
    iteration += 1
    # recompute objective for logging
    toroidal_sections, poloidal_sections = delinearize_data(xk, format)
    elongation = geometry_calculate(toroidal_sections, poloidal_sections)
    print(f"Iteration {iteration}: Max elongation = {elongation}, xk norm = {np.linalg.norm(xk)}")

if __name__ == "__main__":
    # geometry_process()
    # initial guess
    x0, format = linearize_data("./inputs/toroidal_section.json")
    toroidal_sections, poloidal_sections = delinearize_data(x0, format)
    geometry_construct(toroidal_sections, poloidal_sections, 1, plot=True, filename="initial_geometry")

    result = minimize(geometry_process_optimization, x0, 
                      args=(format,), method='L-BFGS-B', 
                      options={'maxiter': 5},
                      callback=callback)
    print("Optimization result:", result)
    toroidal_sections, poloidal_sections = delinearize_data(result.x, format)
    geometry_construct(toroidal_sections, poloidal_sections, 1, plot=True, filename="optimized_geometry")
