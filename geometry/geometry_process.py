import numpy as np
from launch_geometry import geometry_pipeline
import geometry.geometry_fourier as gf
import config as cfg

TRIAL_ELONGATION = None
CURRENT_TRIANGULARITY = None
CURRENT_AR = None
BEST_ELONGATION = 0.0
BEST_TRIANGULARITY = 0.0
BEST_AR = 0.0
BEST_X = None
CURRENT_X = None
FUNC_EVAL = 0
CONSTR_EVAL = 0
FORMAT = None
best_elongation_history = []
elongation_history = []
triangularity_history = []
ar_history = []

class Evaluator:
    def __init__(self):
        self.last_x = None
        self.result = None

    def evaluate(self, x):
        
        if cfg.STUDY_NAME == "Spherical":
            toroidal_sections, poloidal_sections = gf.delinearize_data(x)
        elongation, triangularity, ar = geometry_pipeline(toroidal_sections, poloidal_sections)
        self.result = {
            "elongation": elongation,
            "triangularity": triangularity,
            "ar": ar
        }
        return self.result

class Tracker:
    def __init__(self):
        self.best_x = None
        self.best_value = np.inf
        self.history = []

    def update(self, x, value, constraints_ok=True):
        self.history.append((np.copy(x), value))
        if constraints_ok and value < self.best_value:
            self.best_value = value
            self.best_x = np.copy(x)

evaluator = Evaluator()

def geometry_process_optimization(x):
    global elongation_history, triangularity_history, ar_history
    global FUNC_EVAL
    global CURRENT_TRIANGULARITY, CURRENT_AR, TRIAL_ELONGATION
    global BEST_TRIANGULARITY, BEST_AR, BEST_ELONGATION
    global BEST_X, CURRENT_X
    FUNC_EVAL += 1
    penalty_tri = 0.0
    penalty_AR = 0.0
    total_penalty = 0.0
    r = evaluator.evaluate(x)
    if cfg.METHOD != 'trust-constr' and cfg.METHOD != 'SLSQP':
        if r["triangularity"] < cfg.DELTA_MIN:
            penalty_tri += cfg.PENALTY_WEIGHT * (cfg.DELTA_MIN - r["triangularity"])**2
        if r["triangularity"] > cfg.DELTA_MAX:
            penalty_tri += cfg.PENALTY_WEIGHT * (r["triangularity"] - cfg.DELTA_MAX)**2
        if r["ar"] < cfg.AR_MIN:
            penalty_AR += cfg.AR_WEIGHT * (cfg.AR_MIN - r["ar"])**2
        if r["ar"] > cfg.AR_MAX:
            penalty_AR += cfg.AR_WEIGHT * (r["ar"] - cfg.AR_MAX)**2
        total_penalty = penalty_tri + penalty_AR
        CURRENT_AR = r["ar"]
        CURRENT_TRIANGULARITY = r["triangularity"]
        constraints_ok = (cfg.DELTA_MIN <= r["triangularity"] <= cfg.DELTA_MAX
                            and cfg.AR_MIN <= r["ar"] <= cfg.AR_MAX)
        if constraints_ok:
            if BEST_ELONGATION == 0.0 or r["elongation"] < BEST_ELONGATION:
                BEST_ELONGATION = r["elongation"]
                BEST_X = np.copy(x)
                BEST_TRIANGULARITY = r["triangularity"]
                BEST_AR = r["ar"]
            else:
                CURRENT_X = np.copy(x)
        print(
        f"Func eval {FUNC_EVAL}: "
        f"Objective: {r['elongation']:.4f}, "
        f"Triangularity: {r['triangularity']:.4f}, "
        f"Aspect Ratio: {r['ar']:.4f}, "
        f"Penalty: {total_penalty:.4f}")
        best_elongation_history.append(BEST_ELONGATION)
        elongation_history.append(r["elongation"])
        triangularity_history.append(r['triangularity'])
        ar_history.append(r['ar'])
        return r["elongation"] + total_penalty
    else:
        # print("x:", x)
        CURRENT_AR = r["ar"]
        CURRENT_TRIANGULARITY = r["triangularity"]
        constraints_ok = (cfg.DELTA_MIN <= r["triangularity"] <= cfg.DELTA_MAX
                            and cfg.AR_MIN <= r["ar"] <= cfg.AR_MAX)
        if constraints_ok:
            if BEST_ELONGATION == 0.0 or r["elongation"] < BEST_ELONGATION:
                BEST_ELONGATION = r["elongation"]
                BEST_X = np.copy(x)
                BEST_TRIANGULARITY = r["triangularity"]
                BEST_AR = r["ar"]
            else:
                CURRENT_X = np.copy(x)
        print(
        f"Func eval {FUNC_EVAL}: "
        f"Objective: {r['elongation']:.4f}, "
        f"Triangularity: {r['triangularity']:.4f}, "
        f"Aspect Ratio: {r['ar']:.4f}")
        best_elongation_history.append(BEST_ELONGATION)
        elongation_history.append(r["elongation"])
        triangularity_history.append(r['triangularity'])
        ar_history.append(r['ar'])
        return r["elongation"]

def geometry_process_constraint(x):
    global CONSTR_EVAL
    CONSTR_EVAL += 1
    r = evaluator.evaluate(x)
    # print("x:", x)
    print(
        f"Constr. eval {CONSTR_EVAL}: Triangularity: {r['triangularity']:.4f}, "
        f"Aspect Ratio: {r['ar']:.4f}")

    return [r["triangularity"], r["ar"]]