import numpy as np
import sys
from launch_geometry import geometry_construct, geometry_calculate
import geometry.geometry_fourier as gf
import config as cfg

CURRENT_ELONGATION = None
BEST_ELONGATION = np.inf
FUNC_EVAL = 0

def geometry_process_optimization(x0, format):
    global CURRENT_ELONGATION
    global BEST_ELONGATION
    global FUNC_EVAL
    if np.any(np.isnan(x0)) or np.any(np.isinf(x0)):
        print("BAD INPUT x DETECTED")
        sys.exit(1)
        
    if cfg.STUDY_NAME == "Fourier":
        toroidal_sections, poloidal_sections = gf.delinearize_data(x0, format)
    CURRENT_ELONGATION = geometry_calculate(toroidal_sections, poloidal_sections)
    # track best value seen so far
    if BEST_ELONGATION is None or CURRENT_ELONGATION < BEST_ELONGATION:
        BEST_ELONGATION = CURRENT_ELONGATION
    FUNC_EVAL += 1
    print(f"Func eval {FUNC_EVAL}:", f"Objective: {CURRENT_ELONGATION:.6f}")
    return CURRENT_ELONGATION
    