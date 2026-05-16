"""Configuration file for DONUT project.
This file contains global configuration parameters and settings 
that can be used across the project."""

# optimsation parameters
MAX_ITER = 5
TOLERANCE = 1e-6
STUDY_NAME = "Fourier"
# 'COBYLA' 
# 'Nelder-Mead'
# 'L-BFGS-B'
# 'trust-constr'
METHOD = 'COBYLA'  
K_SMOOTH = 20.0
PHI_T = 2
THETA_T = 2
THETA_P = 2
# constriant parameters
# Triangularity constraint parameters
DELTA_MIN = 0.0
DELTA_MAX = 0.25
PENALTY_WEIGHT = 1000.0
# plotting parameters
color = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
          "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
            "#bcbd22", "#17becf"]
