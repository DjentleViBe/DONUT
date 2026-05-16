import numpy as np
import sys
from launch_geometry import geometry_construct, geometry_calculate
import geometry.geometry_fourier as gf
import config as cfg

CURRENT_ELONGATION = None
CURRENT_TRIANGULARITY = None
CURRENT_AR = None
BEST_ELONGATION = np.inf
FUNC_EVAL = 0

def geometry_process_optimization(x0, format):
    global CURRENT_ELONGATION
    global BEST_ELONGATION
    global CURRENT_TRIANGULARITY
    global CURRENT_AR
    global FUNC_EVAL
    FUNC_EVAL += 1
    if np.any(np.isnan(x0)) or np.any(np.isinf(x0)):
        print("BAD INPUT x DETECTED")
        sys.exit(1)
        
    if cfg.STUDY_NAME == "Fourier":
        toroidal_sections, poloidal_sections = gf.delinearize_data(x0, format)
    CURRENT_ELONGATION, CURRENT_TRIANGULARITY = geometry_calculate(toroidal_sections, poloidal_sections)
    if cfg.METHOD != 'trust-constr' and cfg.METHOD != 'SLSQP':
        penalty_tri = 0.0
        penalty_AR = 0.0
        if CURRENT_TRIANGULARITY < cfg.DELTA_MIN:
            penalty_tri += cfg.PENALTY_WEIGHT * (cfg.DELTA_MIN - CURRENT_TRIANGULARITY)**2

        if CURRENT_TRIANGULARITY > cfg.DELTA_MAX:
            penalty_tri += cfg.PENALTY_WEIGHT * (CURRENT_TRIANGULARITY - cfg.DELTA_MAX)**2
        
        if CURRENT_AR < cfg.AR_MIN:
            penalty_AR += cfg.PENALTY_WEIGHT * (cfg.AR_MIN - CURRENT_AR)**2
        if CURRENT_AR > cfg.AR_MAX:
            penalty_AR += cfg.PENALTY_WEIGHT * (CURRENT_AR - cfg.AR_MAX)**2

        if BEST_ELONGATION is None or CURRENT_ELONGATION < BEST_ELONGATION:
            BEST_ELONGATION = CURRENT_ELONGATION
        print(f"Func eval {FUNC_EVAL}:", f"Objective: {CURRENT_ELONGATION:.6f}, Triangularity: {CURRENT_TRIANGULARITY:.6f}, Aspect Ratio: {CURRENT_AR:.6f}, Penalty: {penalty_tri:.6f}")
        return CURRENT_ELONGATION + penalty_tri + penalty_AR
    else:
        if BEST_ELONGATION is None or CURRENT_ELONGATION < BEST_ELONGATION:
            BEST_ELONGATION = CURRENT_ELONGATION
        print(f"Func eval {FUNC_EVAL}:", f"Objective: {CURRENT_ELONGATION:.6f}, Triangularity: {CURRENT_TRIANGULARITY:.6f}, Aspect Ratio: 0.000000, Penalty: 0.000000")
        return CURRENT_ELONGATION