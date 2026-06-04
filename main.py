"""Main module for DONUT geometry processing and visualization.
This module serves as the entry point for the DONUT project. 
It orchestrates the reading of geometry parameters from input files, 
building the sketches of the sectors, and plotting the poloidal cross 
section of the geometry.
"""
import argparse
import config as cfg
from optimisation import geometry_init, optimisation_block, genetic_block

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DONUT")
    parser.add_argument("--type", required=False, help="Optimisation type")
    args = parser.parse_args()

    if args.type == "genetic":
        print("Genetic alogorithm study")
        genetic_block(3)
    else:
        x0, data_format = geometry_init(plot=True)
        optimisation_block(x0)
