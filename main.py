"""Main module for DONUT geometry processing and visualization.
This module serves as the entry point for the DONUT project. 
It orchestrates the reading of geometry parameters from input files, 
building the sketches of the sectors, and plotting the poloidal cross 
section of the geometry.
"""
import numpy as np
from scipy.optimize import minimize
import config as cfg
import geometry.geometry_process as gp
from geometry.geometry_process import geometry_process_optimization
import geometry.geometry_fourier as gf
from launch_geometry import geometry_construct
from geometry.geometry_writer import write_geometry_parameters_to_file

ITERATION = 0  # external counter

def callback(*args):
    """callback function to be called after each optimization iteration. 
    It logs the current iteration number."""
    global ITERATION
    ITERATION += 1
    xk = args[0]
    print(f"\nIteration {ITERATION}: Max elongation = {gp.BEST_ELONGATION: .4f}," 
          f"Triangularity = {gp.CURRENT_TRIANGULARITY: .4f},"
          f"Aspect Ratio = {gp.CURRENT_AR: .4f},"
          f"xk norm = {np.linalg.norm(xk): .4f}")

if __name__ == "__main__":
    if cfg.STUDY_NAME == "Spherical":
        print("Running Spherical optimization...")
        x0, data_format = gf.linearize_data("./inputs/toroidal_section.json")
        toroidal_sections, poloidal_sections = gf.delinearize_data(x0, data_format)
        write_geometry_parameters_to_file(toroidal_sections, poloidal_sections, data_format, "./results/"+ cfg.STUDY_NAME + "_" + cfg.METHOD + "_initial_geometry.json")
    gp.CURRENT_ELONGATION = geometry_construct(toroidal_sections, poloidal_sections, 1, plot=True,
                       filename=cfg.STUDY_NAME + "_" + cfg.METHOD +"_initial_geometry")

    if cfg.METHOD == 'COBYLA':
        result = minimize(geometry_process_optimization, x0,
                      args=(data_format,), method=cfg.METHOD,
                      options={'maxiter': len(x0) * 2},
                      callback=callback)
    else:
        result = minimize(geometry_process_optimization, x0,
                      args=(data_format,), method=cfg.METHOD,
                      options={'maxiter': cfg.MAX_ITER},
                      callback=callback)

    print("Optimization result:", result)
    """
    constraints = [
        {
            'type': 'ineq',
            'fun': lambda x: compute_triangularity_from_x(x) - cfg.DELTA_MIN
        },
        {
            'type': 'ineq',
            'fun': lambda x: cfg.DELTA_MAX - compute_triangularity_from_x(x)
        }
    ]
    """
    if cfg.STUDY_NAME == "Spherical":
        toroidal_sections, poloidal_sections = gf.delinearize_data(result.x, data_format)
        write_geometry_parameters_to_file(toroidal_sections, poloidal_sections, data_format, "./results/"+ cfg.STUDY_NAME + "_" + cfg.METHOD + "_optimized_geometry.json")
    gp.CURRENT_ELONGATION = geometry_construct(toroidal_sections, poloidal_sections, 1, plot=True,
                    filename=cfg.STUDY_NAME + "_" + cfg.METHOD +"_optimized_geometry")
