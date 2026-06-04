"""Main module for DONUT geometry processing and visualization.
This module serves as the entry point for the DONUT project. 
It orchestrates the reading of geometry parameters from input files, 
building the sketches of the sectors, and plotting the poloidal cross 
section of the geometry.
"""
import argparse
import config as cfg
import geometry.geometry_process as gp
from file_operations import write_to_csv
from optimisation import geometry_init, optimisation_block

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DONUT")
    parser.add_argument("--type", required=False, help="Optimisation type")
    args = parser.parse_args()

    if args.type == "genetic":
        print("Genetic alogorithm study")
        x0, data_format = geometry_init(plot=False)
        print(gp.TRIAL_ELONGATION, gp.CURRENT_TRIANGULARITY, gp.CURRENT_AR)
    else:
        x0, data_format = geometry_init(plot=True)
        optimisation_block(x0, data_format)
