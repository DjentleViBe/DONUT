"""Main module for DONUT geometry processing and visualization.
This module serves as the entry point for the DONUT project. 
It orchestrates the reading of geometry parameters from input files, 
building the sketches of the sectors, and plotting the poloidal cross 
section of the geometry.
"""
from launch_geometry import geometry_init, geometry_construct

if __name__ == "__main__":
    poloidal_sections, toroid_file = geometry_init()
    geometry_construct(poloidal_sections, toroid_file)
