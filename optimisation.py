"""
Optimisation logic for DONUT
"""
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
import config as cfg
from file_operations import write_to_csv
from geometry.geometry_process import geometry_process_optimization, geometry_process_constraint
from geometry.geometry_writer import write_geometry_parameters_to_file
from geometry.geometry_fourier import genetic_data
from launch_geometry import geometry_construct
import geometry.geometry_process as gp
import geometry.geometry_fourier as gf
import random

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

def optimisation_block(x0):
    """
    main optimisation block for gradient based methods"""
    print(f"Running {cfg.STUDY_NAME} optimization...")
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
                                         "./results/"+
                                        cfg.STUDY_NAME + "_" + cfg.METHOD +
                                        "_optimized_geometry.json")
    print(f"Current elongation : {gp.TRIAL_ELONGATION}")
    geometry_construct(toroidal_sections, poloidal_sections,
                        plot=True, filename=cfg.STUDY_NAME +
                        "_" + cfg.METHOD +"_optimized_geometry")
    toroidal_sections, poloidal_sections = gf.delinearize_data(gp.BEST_X)
    write_geometry_parameters_to_file(toroidal_sections, poloidal_sections,
                                         "./results/"+ cfg.STUDY_NAME +
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
    x0, data_format = gf.linearize_data("./inputs/toroidal_section.json")
    gp.FORMAT = data_format
    toroidal_sections, poloidal_sections = gf.delinearize_data(x0)
    write_geometry_parameters_to_file(toroidal_sections, poloidal_sections,
                                         "./results/"+ cfg.STUDY_NAME
                                        + "_" + cfg.METHOD + "_initial_geometry.json")

    gp.TRIAL_ELONGATION, gp.CURRENT_TRIANGULARITY, gp.CURRENT_AR = geometry_construct(
                                                            toroidal_sections,
                                                            poloidal_sections,
                                                            init=True, plot=plot,
                                                            filename=cfg.STUDY_NAME +
                                                            "_" + cfg.METHOD +
                                                            "_initial_geometry")
    return x0, data_format

def genetic_init(toroidal_sections, poloidal_sections, plot):
    """
    Initialise geoemtry
    """
    write_geometry_parameters_to_file(toroidal_sections, poloidal_sections,
                                         "./results/"+ cfg.STUDY_NAME
                                        + "_" + cfg.METHOD + "_initial_geometry.json")

    gp.TRIAL_ELONGATION, gp.CURRENT_TRIANGULARITY, gp.CURRENT_AR = geometry_construct(
                                                            toroidal_sections,
                                                            poloidal_sections,
                                                            init=True, plot=plot,
                                                            filename=cfg.STUDY_NAME +
                                                            "_" + cfg.METHOD +
                                                            "_initial_geometry")
    return 0

def genetic_objective(te, ct, ar):
    return te + ct + ar

def tournament_selection(fitness, k=3):
    candidates = random.sample(fitness, k)
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]

def crossover(p1, p2):
    cut = np.random.randint(1, len(p1))
    return np.concatenate([p1[:cut], p2[cut:]])

def mutate(genome, mutation_rate=0.1, sigma=0.05):
    child = genome.copy()
    mask = np.random.rand(len(child)) < mutation_rate
    child[mask] += np.random.normal(loc=0.0, scale=sigma,size=np.sum(mask))
    child = np.clip(child, 0.0, 1.0)
    return child

def evaluate_genome(do_gen):
    toroids, poloids = gf.decode_genome(do_gen, 4)
    genetic_init(
        toroids,
        poloids,
        plot=False
    )

    score = genetic_objective(
        gp.TRIAL_ELONGATION,
        gp.CURRENT_TRIANGULARITY,
        gp.CURRENT_AR
    )
    return score

def genetic_block(pop_size = 50,
                  generations=100,
                  elite_fraction=0.1):
    # --------------------------------------------------
    # Initial population
    # --------------------------------------------------
    population = [genetic_data(4) for _ in range(pop_size)]
    best_genome = None
    best_score = np.inf
    # --------------------------------------------------
    # Evolution loop
    # --------------------------------------------------
    for generation in range(generations):
        fitness_list = []

        # ----------------------------------------------
        # Evaluate
        # ----------------------------------------------
        for do_gen in population:
            score = evaluate_genome(do_gen)
            fitness_list.append(
                (score, do_gen)
            )
        fitness_list.sort(
            key=lambda x: x[0]
        )
        generation_best_score = fitness_list[0][0]
        generation_best_genome = fitness_list[0][1]
        if generation_best_score < best_score:
            best_score = generation_best_score
            best_genome = generation_best_genome.copy()
        print(
            f"Gen {generation:4d} | "
            f"Best = {generation_best_score:.6f} | "
            f"Global Best = {best_score:.6f}"
        )
        # ----------------------------------------------
        # Elitism
        # ----------------------------------------------
        elite_count = max(1, int(elite_fraction * pop_size))
        new_population = [genome.copy() for _, genome in fitness_list[:elite_count]]
        # ----------------------------------------------
        # Create offspring
        # ----------------------------------------------
        while len(new_population) < pop_size:
            parent1 = tournament_selection(fitness_list)
            parent2 = tournament_selection(fitness_list)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate=0.1, sigma=0.05)
            new_population.append(child)
        population = new_population
    # --------------------------------------------------
    # Decode final best solution
    # --------------------------------------------------
    best_toroids, best_poloids = gf.decode_genome(
        best_genome,
        4
    )
    return (
        best_genome,
        best_score,
        best_toroids,
        best_poloids
    )
