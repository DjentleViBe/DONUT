"""Geometry reader for DONUT geometry.
This module contains functions to read the geometry parameters 
from the input files, including the toroidal section file and the 
poloidal section files. It uses the file operations to list the files 
in the input directory and read the JSON files to extract the necessary 
parameters for building the sketches of the sectors.
"""
import json
import numpy as np
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

def linearize_data(toroidal_file):
    """
    Output : 
    Toroidal - [phi, theta, radius, weights, sections] * N_t
    Poloidal - [theta, radius, weights] * N_s
    """
    print("linearizing data")
    toroidal_array = []
    poloidal_array = []
    format = []
    toroidal_prop = get_geometry_parameters_from_toroidal_file(toroidal_file)
    format.append(toroidal_prop.get("N_t"))
    phi = np.array(toroidal_prop.get("phi"))
    theta = np.array(toroidal_prop.get("theta"))
    radius = np.array(toroidal_prop.get("radius"))
    weights = np.array(toroidal_prop.get("weights"))
    sections = np.array(toroidal_prop.get("sections"))
    combined = np.concatenate([phi, theta, radius, weights, sections])
    toroidal_array.append(combined)
    # loop through the poloidal files
    for i in range (0, toroidal_prop['N_t']):
        with open("./inputs/poloidal_section_" + str(i + 1) + ".json", "r", encoding="utf-8") as f:
            data = json.load(f)
            nval = data.get("N_s")
            theta = np.array(data.get("theta"))
            radius = np.array(data.get("radius"))
            weights = np.array(data.get("weights"))
            degree = data.get("degree")

            combined = np.concatenate([theta, radius, weights])
            poloidal_array.append(combined)
            format.append(nval)
    poloidal_flat = np.concatenate(poloidal_array) if poloidal_array else np.array([])
    toroidal_flat = np.concatenate(toroidal_array) if toroidal_array else np.array([])
    x0 = np.concatenate([poloidal_flat, toroidal_flat])
    print(f"Total number of elements to optimize: {len(x0)}")
    print(format)
    return x0, format

def delinearize_data(x0, format):
    nval = format[0]
    phi = x0[-nval*5 : -nval*4]
    theta = x0[-nval*4 : -nval*3]
    radius = x0[-nval*3 : -nval*2]
    weights = x0[-nval*2 : -nval]
    sections = x0[-nval : ]
    toroidal_sections = {"N_t": nval, "theta": theta, "phi": phi,
                "radius": radius, "weights": weights,
                "degree": 3, "sections": sections}
    poloidal_sections= []
    idx = 0

    for i in range(format[0]):
        N_s = format[i + 1]
        
        start = idx
        end = idx + 3 * N_s
        theta  = x0[start : start + N_s]
        radius = x0[start + N_s : start + 2 * N_s]
        weights = x0[start + 2 * N_s : start + 3 * N_s]
        poloidal_section = {
            "N_s": N_s,
            "theta": theta,
            "radius": radius,
            "weights": weights,
            "degree": 3
        }
        poloidal_sections.append(poloidal_section)
        idx += 3 * N_s
    return toroidal_sections, poloidal_sections
