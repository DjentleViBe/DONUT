import numpy as np
import json
import config as cfg
from geometry.geometry_reader import get_geometry_parameters_from_toroidal_file
from helper.fourier_helper import fourier_encode, fourier_decode

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
    phi = np.array(toroidal_prop.get("phi")) * np.pi / 180.0
    theta = np.array(toroidal_prop.get("theta")) * np.pi / 180.0
    radius = np.array(toroidal_prop.get("radius"))
    weights = np.array(toroidal_prop.get("weights"))
    sections = np.array(toroidal_prop.get("sections"))

    sin_phi_coeffs   = fourier_encode(np.sin(phi), cfg.PHI_T)
    cos_phi_coeffs   = fourier_encode(np.cos(phi), cfg.PHI_T)
    
    sin_theta_coeffs = fourier_encode(np.sin(theta), cfg.THETA_T)
    cos_theta_coeffs = fourier_encode(np.cos(theta), cfg.THETA_T)

    combined = np.concatenate([sin_phi_coeffs, cos_phi_coeffs, sin_theta_coeffs, cos_theta_coeffs, radius, weights, sections])
    toroidal_array.append(combined)
    # loop through the poloidal files
    for i in range (0, toroidal_prop['N_t']):
        with open("./inputs/poloidal_section_" + str(i + 1) + ".json", "r", encoding="utf-8") as f:
            data = json.load(f)
            nval = data.get("N_s")
            theta = np.array(data.get("theta")) * np.pi / 180.0
            radius = np.array(data.get("radius"))
            weights = np.array(data.get("weights"))
            degree = data.get("degree")

            sin_theta_coeffs = fourier_encode(np.sin(theta), cfg.THETA_P)
            cos_theta_coeffs = fourier_encode(np.cos(theta), cfg.THETA_P)
            combined = np.concatenate([sin_theta_coeffs, cos_theta_coeffs, radius, weights])
            poloidal_array.append(combined)
            format.append(nval)
    poloidal_flat = np.concatenate(poloidal_array) if poloidal_array else np.array([])
    toroidal_flat = np.concatenate(toroidal_array) if toroidal_array else np.array([])
    x0 = np.concatenate([poloidal_flat, toroidal_flat])
    print(f"Total number of elements to optimize: {len(x0)}")
    return x0, format

def delinearize_data(x0, format_list):
    """
    Reconstruct toroidal + poloidal from flat vector.
    """

    idx = 0

    # -----------------------
    # Poloidal reconstruction
    # -----------------------
    N_t = format_list[0]
    poloidal_sections = []
    ncoeff_p = 2 * cfg.THETA_P + 1
    for i in range(N_t):
        N_s = format_list[i + 1]

        sin_theta_coeffs = x0[idx : idx + ncoeff_p]
        idx += ncoeff_p

        cos_theta_coeffs = x0[idx : idx + ncoeff_p]
        idx += ncoeff_p

        radius = x0[idx : idx + N_s]
        idx += N_s

        weights = x0[idx : idx + N_s]
        idx += N_s

        sin_theta = fourier_decode(sin_theta_coeffs, N_s, cfg.THETA_P)
        cos_theta = fourier_decode(cos_theta_coeffs, N_s, cfg.THETA_P)
        norm = np.sqrt(sin_theta**2 + cos_theta**2)
        sin_theta /= np.maximum(norm, 1e-12)
        cos_theta /= np.maximum(norm, 1e-12)
        theta = np.mod(np.degrees(np.arctan2(sin_theta, cos_theta)), 360.0)

        poloidal_sections.append({
            "N_s": N_s,
            "theta": theta,
            "radius": radius,
            "weights": weights,
            "degree": 3
        })

    # -----------------------
    # Toroidal reconstruction
    # -----------------------
    n_coeff_phi_t = 2 * cfg.PHI_T + 1
    n_coeff_theta_t = 2 * cfg.THETA_T + 1
    N_t = format_list[0]

    sin_phi_coeffs = x0[idx : idx + n_coeff_phi_t]
    idx += n_coeff_phi_t
    cos_phi_coeffs = x0[idx : idx + n_coeff_phi_t]
    idx += n_coeff_phi_t

    sin_theta_coeffs = x0[idx : idx + n_coeff_theta_t]
    idx += n_coeff_theta_t
    cos_theta_coeffs = x0[idx : idx + n_coeff_theta_t]
    idx += n_coeff_theta_t

    radius = x0[idx : idx + N_t]
    idx += N_t

    weights = x0[idx : idx + N_t]
    idx += N_t

    sections = x0[idx : idx + N_t]
    idx += N_t

    sin_phi = fourier_decode(sin_phi_coeffs, N_t, cfg.PHI_T)
    cos_phi = fourier_decode(cos_phi_coeffs, N_t, cfg.PHI_T)
    norm = np.sqrt(sin_phi**2 + cos_phi**2)
    sin_phi /= np.maximum(norm, 1e-12)
    cos_phi /= np.maximum(norm, 1e-12)

    sin_theta = fourier_decode(sin_theta_coeffs, N_t, cfg.THETA_T)
    cos_theta = fourier_decode(cos_theta_coeffs, N_t, cfg.THETA_T)
    norm = np.sqrt(sin_theta**2 + cos_theta**2)
    sin_theta /= np.maximum(norm, 1e-12)
    cos_theta /= np.maximum(norm, 1e-12)

    phi = np.mod(np.degrees(np.arctan2(sin_phi, cos_phi)), 360.0)
    theta = np.mod(np.degrees(np.arctan2(sin_theta, cos_theta)), 360.0)

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
