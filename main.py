"""Main module for DONUT geometry processing and visualization.
This module serves as the entry point for the DONUT project. 
It orchestrates the reading of geometry parameters from input files, 
building the sketches of the sectors, and plotting the poloidal cross 
section of the geometry.
"""
import argparse
import config as cfg
import geometry.geometry_process as gp
from geometry.geometry_fourier import genetic_data
from file_operations import write_to_csv
from optimisation import geometry_init, genetic_init, optimisation_block
from geometry.geometry_writer import write_geometry_parameters_to_file
import geometry.geometry_fourier as gf

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DONUT")
    parser.add_argument("--type", required=False, help="Optimisation type")
    args = parser.parse_args()

    if args.type == "genetic":
        print("Genetic alogorithm study")
        toroid, poloid, do_gen = genetic_data(4)
        write_geometry_parameters_to_file(toroid, poloid,
                                         "./results/"+ cfg.STUDY_NAME
                                        + "_" + cfg.METHOD + "_initial_geometry.json")
        genetic_init(toroid, poloid, plot=False)
        print(gp.TRIAL_ELONGATION, gp.CURRENT_TRIANGULARITY, gp.CURRENT_AR)
    else:
        x0, data_format = geometry_init(plot=True)
        optimisation_block(x0, data_format)
