import numpy as np
import sys
from launch_geometry import geometry_construct, geometry_calculate
import geometry.geometry_fourier as gf
import config as cfg

CURRENT_ELONGATION = None
CURRENT_TRIANGULARITY = None
BEST_ELONGATION = np.inf
FUNC_EVAL = 0

def geometry_process_optimization(x0, format):
    global CURRENT_ELONGATION
    global BEST_ELONGATION
    global CURRENT_TRIANGULARITY
    global FUNC_EVAL
    FUNC_EVAL += 1
    if np.any(np.isnan(x0)) or np.any(np.isinf(x0)):
        print("BAD INPUT x DETECTED")
        sys.exit(1)
        
    if cfg.STUDY_NAME == "Fourier":
        toroidal_sections, poloidal_sections = gf.delinearize_data(x0, format)
    CURRENT_ELONGATION, CURRENT_TRIANGULARITY = geometry_calculate(toroidal_sections, poloidal_sections)
    if cfg.METHOD != 'trust-constr' or cfg.METHOD != 'SLSQP:':
        penalty = 0.0
        delta_min = 0.25
        delta_max = 0.55
        penalty_weight = 1000.0

        if CURRENT_TRIANGULARITY < delta_min:
            penalty += penalty_weight * (delta_min - CURRENT_TRIANGULARITY)**2

        if CURRENT_TRIANGULARITY > delta_max:
            penalty += penalty_weight * (CURRENT_TRIANGULARITY - delta_max)**2
            # track best value seen so far
        if BEST_ELONGATION is None or CURRENT_ELONGATION < BEST_ELONGATION:
            BEST_ELONGATION = CURRENT_ELONGATION
        print(f"Func eval {FUNC_EVAL}:", f"Objective: {CURRENT_ELONGATION:.6f}, Triangularity: {CURRENT_TRIANGULARITY:.6f}, Penalty: {penalty:.6f}")
        return CURRENT_ELONGATION + penalty
    else:
        if BEST_ELONGATION is None or CURRENT_ELONGATION < BEST_ELONGATION:
            BEST_ELONGATION = CURRENT_ELONGATION
        print(f"Func eval {FUNC_EVAL}:", f"Objective: {CURRENT_ELONGATION:.6f}, Triangularity: {CURRENT_TRIANGULARITY:.6f}")
        return CURRENT_ELONGATION