"""Geometry reader for DONUT geometry.
This module contains functions to read the geometry parameters 
from the input files, including the toroidal section file and the 
poloidal section files. It uses the file operations to list the files 
in the input directory and read the JSON files to extract the necessary 
parameters for building the sketches of the sectors.
"""
import json
from geometry.geometry_check import param_num_sector, param_num_sector_toroidal

def get_poloidal_sections_from_toroidal_file(filename):
    """Read the toroidal section file to get the list of poloidal sections to process.
    Returns:
    List of poloidal section file names (without extension) to process.
    """
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
        # get the list of poloidal sections
        poloidal_sections = data.get("main_sections", [])
        print("Poloidal sections to process:", poloidal_sections)
        return poloidal_sections

def get_geometry_parameters_from_poloidal_file(poloidal_file):
    """Read the poloidal section file to get the geometry parameters for building the sketch.
    Args:
        poloidal_file (str): The path to the poloidal section file.
    Returns:
        dict: A dictionary containing the geometry parameters.
    """
    with open(poloidal_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        nval = data.get("N_s")
        theta = data.get("theta")
        radius = data.get("radius")
        weights = data.get("weights")
        degree = data.get("degree")
        param_num_sector(nval, theta, radius)
        return {"N_s": nval, "theta": theta, "radius": radius, "weights": weights, "degree": degree}

def get_geometry_parameters_from_toroidal_file(toroidal_file):
    """Read the toroidal section file to get the geometry parameters for building the sketch.
    Args:
        toroidal_file (str): The path to the toroidal section file.
    Returns:
        dict: A dictionary containing the geometry parameters.
    """
    with open(toroidal_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        nval = data.get("N_t")
        theta = data.get("theta")
        phi = data.get("phi")
        radius = data.get("radius")
        weights = data.get("weights")
        degree = data.get("degree")
        sections = data.get("sections")
        param_num_sector_toroidal(nval, theta, phi, radius)
        return {"N_t": nval, "theta": theta, "phi": phi,
                "radius": radius, "weights": weights,
                "degree": degree, "sections": sections}

def get_geometry_parameters_from_twist_file(twist_file):
    """Read the twist section file to get the geometry parameters for building the sketch.
    Args:
        twist_file (str): The path to the twist section file.
    Returns:
        dict: A dictionary containing the geometry parameters.
    """
    with open(twist_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        N = data.get("N")
        A = data.get("A")
        k = data.get("k")
        
        # aram_num_sector_toroidal(nval, theta, phi, radius)
        return {"N": N, "A": A, "k": k}