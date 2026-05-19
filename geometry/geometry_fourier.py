import numpy as np
import json
import config as cfg
from geometry.geometry_reader import get_geometry_parameters_from_toroidal_file
from geometry.geometry_constraints import f_to_u, u_to_f
from helper.fourier_helper import fourier_encode, fourier_decode
import geometry.geometry_process as gp

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
    theta0 = np.deg2rad(90.0)
    theta_delta = np.deg2rad(45.0)
    toroidal_prop = get_geometry_parameters_from_toroidal_file(toroidal_file)
    format.append(toroidal_prop.get("N_t"))
    phi = np.radians(np.array(toroidal_prop.get("phi")))
    theta = ((np.array(toroidal_prop.get("theta")) * np.pi / 180.0) - theta0) / theta_delta
    radius = np.array(toroidal_prop.get("radius"), dtype=np.float64)
    weights = np.array(toroidal_prop.get("weights"), dtype=np.float64)
    sections = np.array(toroidal_prop.get("sections"), dtype=np.float64)

    phi_to_u = f_to_u(phi)
    combined = np.concatenate([phi_to_u, theta, radius, weights, sections])
    toroidal_array.append(combined)
    # loop through the poloidal files
    for i in range (0, toroidal_prop['N_t']):
        with open("./inputs/poloidal_section_" + str(i + 1) + ".json", "r", encoding="utf-8") as f:
            data = json.load(f)
            nval = data.get("N_s")
            psi = np.array(data.get("psi")) * np.pi / 180.0
            radius = np.array(data.get("radius"), dtype=np.float64)
            weights = np.array(data.get("weights"), dtype=np.float64)
            degree = data.get("degree")

            psi_to_u = f_to_u(psi)
            combined = np.concatenate([psi_to_u, radius, weights])
            poloidal_array.append(combined)
            format.append(nval)
    poloidal_flat = np.concatenate(poloidal_array) if poloidal_array else np.array([])
    toroidal_flat = np.concatenate(toroidal_array) if toroidal_array else np.array([])
    x0 = np.concatenate([poloidal_flat, toroidal_flat])
    print(f"Total number of elements to optimize: {len(x0)}")
    
    return x0, format

def delinearize_data(x0):
    """
    Reconstruct toroidal + poloidal from flat vector.
    """
    idx = 0
    theta0 = np.deg2rad(90.0)
    theta_delta = np.deg2rad(45.0)
    # -----------------------
    # Poloidal reconstruction
    # -----------------------
    format_list = gp.FORMAT
    N_t = format_list[0]
    poloidal_sections = []
    for i in range(N_t):
        N_s = format_list[i + 1]

        psi_to_u = x0[idx : idx + N_s]
        idx += N_s

        radius = x0[idx : idx + N_s]
        idx += N_s

        weights = x0[idx : idx + N_s]
        idx += N_s

        psi = np.degrees(u_to_f(psi_to_u, 0))

        poloidal_sections.append({
            "N_s": N_s,
            "psi": psi,
            "radius": radius,
            "weights": weights,
            "degree": 3
        })

    # -----------------------
    # Toroidal reconstruction
    # -----------------------
    N_t = format_list[0]

    phi_to_u = x0[idx : idx + N_t]
    idx += N_t

    theta_to_x = x0[idx : idx + N_t]
    idx += N_t

    radius = x0[idx : idx + N_t]
    idx += N_t

    weights = x0[idx : idx + N_t]
    idx += N_t

    sections = x0[idx : idx + N_t]
    idx += N_t

    phi = np.degrees(u_to_f(phi_to_u, 0))
    theta = np.degrees(theta0 + theta_to_x * theta_delta)  # direct decoding for theta
    toroidal_sections = {
        "N_t": N_t,
        "phi": phi,
        "theta": theta,
        "radius": radius,
        "weights": weights,
        "degree": 3,
        "sections": sections
    }
    return toroidal_sections, poloidal_sections
