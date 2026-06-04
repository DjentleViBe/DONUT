import numpy as np
import config as cfg
from geometry.geometry_process import geometry_process_optimization, geometry_process_constraint
from scipy.optimize import minimize, NonlinearConstraint
from file_operations import write_to_csv
from geometry.geometry_writer import write_geometry_parameters_to_file
from launch_geometry import geometry_construct
import geometry.geometry_process as gp
import geometry.geometry_fourier as gf

ITERATION = 0  # external counter
elongation_current_iteration = []
triangularity_current_iteration = []
ar_current_iteration = []
elongation_best_iteration = []
triangularity_best_iteration = []
ar_best_iteration = []

def callback(*args):
    """callback function to be called after each optimization iteration. 
    It logs the current iteration number."""
    global ITERATION
    ITERATION += 1
    xk = args[0]
    elongation_current_iteration.append(gp.TRIAL_ELONGATION)
    triangularity_current_iteration.append(gp.CURRENT_TRIANGULARITY)
    ar_current_iteration.append(gp.CURRENT_AR)
    elongation_best_iteration.append(gp.BEST_ELONGATION)
    triangularity_best_iteration.append(gp.BEST_TRIANGULARITY)
    ar_best_iteration.append(gp.BEST_AR)
    print(f"\nIteration {ITERATION}: Current elongation = {gp.TRIAL_ELONGATION:.4f},"
          f"Triangularity = {gp.CURRENT_TRIANGULARITY:.4f},"
          f"Aspect Ratio = {gp.CURRENT_AR:.4f},"
          f"xk norm = {np.linalg.norm(xk):.4f}")
    
def optimisation_block(x0, data_format):
    """
    main optimisation block for gradient based methods"""
    elongation_current_iteration.append(gp.TRIAL_ELONGATION)
    triangularity_current_iteration.append(gp.CURRENT_TRIANGULARITY)
    ar_current_iteration.append(gp.CURRENT_AR)
    if cfg.METHOD == 'COBYLA':
        result = minimize(geometry_process_optimization, x0,
                    method=cfg.METHOD,
                    options={'maxiter': len(x0) * 2},
                    callback=callback)
    elif cfg.METHOD == 'Powell':
        result = minimize(geometry_process_optimization, x0,
                    method=cfg.METHOD,
                    options={'maxfev': cfg.MAX_ITER * len(x0)},
                    callback=callback)
    elif cfg.METHOD == 'trust-constr':
        nonlinear_constraint = NonlinearConstraint(geometry_process_constraint,
                                                [cfg.DELTA_MIN, cfg.AR_MIN],
                                                [cfg.DELTA_MAX, cfg.AR_MAX],
                                                finite_diff_rel_step=cfg.FDRS)
        result = minimize(geometry_process_optimization, x0,
                    method=cfg.METHOD,
                    options={'maxiter': cfg.MAX_ITER, 'finite_diff_rel_step': cfg.FDRS},
                    constraints=[nonlinear_constraint],
                    callback=callback)
    elif cfg.METHOD == 'SLSQP':
        nonlinear_constraint = NonlinearConstraint(geometry_process_constraint,
                                                [cfg.DELTA_MIN, cfg.AR_MIN],
                                                [cfg.DELTA_MAX, cfg.AR_MAX])
        result = minimize(geometry_process_optimization, x0,
                    method=cfg.METHOD,
                    options={'maxiter': cfg.MAX_ITER, "eps": cfg.FDRS},
                    constraints=[nonlinear_constraint],
                    bounds = [(-1, 1)] * len(x0),
                    callback=callback)
    else:
        result = minimize(geometry_process_optimization, x0,
                    method=cfg.METHOD,
                    options={'maxiter': cfg.MAX_ITER},
                    callback=callback)

    print("Optimization result:", result)

    toroidal_sections, poloidal_sections = gf.delinearize_data(gp.CURRENT_X)
    write_geometry_parameters_to_file(toroidal_sections, poloidal_sections,
                                        data_format, "./results/"+
                                        cfg.STUDY_NAME + "_" + cfg.METHOD +
                                        "_optimized_geometry.json")
    print(f"Current elongation : {gp.TRIAL_ELONGATION}")
    geometry_construct(toroidal_sections, poloidal_sections,
                        plot=True, filename=cfg.STUDY_NAME +
                        "_" + cfg.METHOD +"_optimized_geometry")
    toroidal_sections, poloidal_sections = gf.delinearize_data(gp.BEST_X)
    write_geometry_parameters_to_file(toroidal_sections, poloidal_sections,
                                        data_format, "./results/"+ cfg.STUDY_NAME +
                                        "_" + cfg.METHOD + "_best_geometry.json")
    print(f"Max elongation : {gp.BEST_ELONGATION}")
    geometry_construct(toroidal_sections, poloidal_sections,
                        plot=True, filename=cfg.STUDY_NAME +
                        "_" + cfg.METHOD +"_best_geometry")   
    elongation_best_iteration.append(gp.BEST_ELONGATION)
    triangularity_best_iteration.append(gp.BEST_TRIANGULARITY)
    ar_best_iteration.append(gp.BEST_AR)
    write_to_csv(gp.elongation_history, gp.triangularity_history, gp.ar_history,
                filename="./results/" + cfg.STUDY_NAME + "_" + cfg.METHOD +"_history.csv")
    write_to_csv(elongation_current_iteration, triangularity_current_iteration,
                ar_current_iteration, filename="./results/" + cfg.STUDY_NAME +
                    "_" + cfg.METHOD +"_current_iteration.csv")
    write_to_csv(elongation_best_iteration, triangularity_best_iteration,
                ar_best_iteration, filename="./results/" + cfg.STUDY_NAME +
                    "_" + cfg.METHOD +"_best_iteration.csv")
    
def geometry_init(plot):
    """
    Initialise geoemtry
    """
    print(f"Running {cfg.STUDY_NAME} optimization...")
    x0, data_format = gf.linearize_data("./inputs/toroidal_section.json")
    gp.FORMAT = data_format
    toroidal_sections, poloidal_sections = gf.delinearize_data(x0)
    write_geometry_parameters_to_file(toroidal_sections, poloidal_sections,
                                        data_format, "./results/"+ cfg.STUDY_NAME
                                        + "_" + cfg.METHOD + "_initial_geometry.json")

    gp.TRIAL_ELONGATION, gp.CURRENT_TRIANGULARITY, gp.CURRENT_AR = geometry_construct(
                                                            toroidal_sections,
                                                            poloidal_sections,
                                                            init=True, plot=plot,
                                                            filename=cfg.STUDY_NAME +
                                                            "_" + cfg.METHOD +
                                                            "_initial_geometry")
    return x0, data_format