import numpy as np
import json
import config as cfg
from geometry.geometry_reader import get_geometry_parameters_from_toroidal_file, get_geometry_parameters_from_twist_file
from geometry.geometry_constraints import f_to_u, u_to_f
from helper.fourier_helper import fourier_encode, fourier_decode
import geometry.geometry_process as gp
import random
from classes_geometry import DonutGenetic

def linearize_data(toroidal_file):
    """
    Output : 
    Toroidal - [phi, theta, radius, weights, sections] * N_t
    Poloidal - [theta, radius, weights] * N_s
    """
    print("Linearizing data")
    toroidal_array = []
    poloidal_array = []
    format = []
    theta0 = np.deg2rad(90.0)
    theta_delta = np.deg2rad(45.0)
    toroidal_prop = get_geometry_parameters_from_toroidal_file(toroidal_file)
    twist_prop = get_geometry_parameters_from_twist_file("inputs/twist_section.json")
    format.append(toroidal_prop.get("N_t"))
    phi = np.radians(np.array(toroidal_prop.get("phi")))
    theta = ((np.array(toroidal_prop.get("theta")) * np.pi / 180.0) - theta0) / theta_delta
    radius = np.array(toroidal_prop.get("radius"), dtype=np.float64)
    weights = np.array(toroidal_prop.get("weights"), dtype=np.float64)
    phi_to_u = f_to_u(phi)

    if cfg.STUDY_NAME == "Type1":
        sections = np.array(toroidal_prop.get("sections"), dtype=np.float64)
        combined = np.concatenate([phi_to_u, theta, radius, weights, sections])
    elif cfg.STUDY_NAME == "Type3":
        sections = np.array(toroidal_prop.get("sections"), dtype=np.float64)
        deltas = np.zeros(len(sections))
        # deltas = np.diff(np.r_[sections, sections[0] + len(sections)])
        combined = np.concatenate([phi_to_u, theta, radius, weights, deltas])

    elif cfg.STUDY_NAME == "Type2":
        N = np.array([twist_prop.get("N")], dtype=np.float64)
        A = np.array([twist_prop.get("A")], dtype=np.float64)
        k = np.array([twist_prop.get("k")], dtype=np.float64)
        combined = np.concatenate([phi_to_u, theta, radius, weights, N, A, k])
    
    toroidal_array.append(combined)
    # loop through the poloidal files
    for i in range (0, toroidal_prop['N_t']):
        if cfg.STUDY_NAME == "Type1":
            filename = "./inputs/poloidal_section_" + str(i + 1) + ".json"
            with open(filename, "r", encoding="utf-8") as f:
                data = json.load(f)
                psi = np.array(data.get("psi")) * np.pi / 180.0
            psi = f_to_u(psi)
        elif cfg.STUDY_NAME == "Type2":
            filename = "./inputs/poloidal_section.json"
            with open(filename, "r", encoding="utf-8") as f:
                data = json.load(f)
                psi = np.array(data.get("psi")) * np.pi / 180.0
            psi = f_to_u(psi)     
        elif cfg.STUDY_NAME == "Type3":
            filename = "./inputs/poloidal_section_" + str(i + 1) + ".json"
            with open(filename, "r", encoding="utf-8") as f:
                data = json.load(f)
                psi = np.array(data.get("psi")) * np.pi / 180.0
        nval = data.get("N_s")  
        radius = np.array(data.get("radius"), dtype=np.float64)
        weights = np.array(data.get("weights"), dtype=np.float64)
        degree = data.get("degree")
        combined = np.concatenate([psi, radius, weights])
        poloidal_array.append(combined)
        format.append(nval)
        if cfg.STUDY_NAME == "Type2":
            break
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
    poloidal_sections = []
    if cfg.STUDY_NAME == "Type1":
        N_t = format_list[0]
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
    elif cfg.STUDY_NAME == "Type2":
        N_t = cfg.NUM_T
        for i in range(N_t):
            idx = 0
            N_s = format_list[1]

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
    elif cfg.STUDY_NAME == "Type3":
        N_t = format_list[0]
        for i in range(N_t):
            N_s = format_list[i + 1]

            psi_to_u = x0[idx : idx + N_s]
            idx += N_s

            radius = x0[idx : idx + N_s]
            idx += N_s

            weights = x0[idx : idx + N_s]
            idx += N_s

            psi = np.degrees(psi_to_u)

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

    phi = np.degrees(u_to_f(phi_to_u, 0))
    theta = np.degrees(theta0 + theta_to_x * theta_delta)  # direct decoding for theta
    
    if cfg.STUDY_NAME == "Type1":
        sections = x0[idx : idx + N_t]
        toroidal_sections = {
        "N_t": N_t,
        "phi": phi,
        "theta": theta,
        "radius": radius,
        "weights": weights,
        "degree": 3,
        "deltas": sections
        }
    elif cfg.STUDY_NAME == "Type3":
        deltas = x0[idx : idx + N_t]
        toroidal_sections = {
        "N_t": N_t,
        "phi": phi,
        "theta": theta,
        "radius": radius,
        "weights": weights,
        "degree": 3,
        "deltas": deltas
        }
    elif cfg.STUDY_NAME == "Type2":
        sections = np.array(np.linspace(start=0.0, stop = 1.0 - (1 / cfg.NUM_T), num=cfg.NUM_T), dtype=np.float64)
        N = x0[idx : idx + 1]
        idx += 1
        gp.N = N
        A = x0[idx : idx + 1]
        idx += 1
        gp.A = A
        k = x0[idx : idx + 1]
        gp.k = k
        idx += 1
        toroidal_sections = {
        "N_t": N_t,
        "phi": phi,
        "theta": theta,
        "radius": radius,
        "weights": weights,
        "degree": 3,
        "sections": sections,
        "N" : N,
        "A" : A,
        "k" : k
        }
    
    return toroidal_sections, poloidal_sections

def genetic_data(N_s = 4, 
                 phi_limits = [-30, 30], 
                 N_limits = [1, 6],
                 A_limits = [0.1, 1],
                 k_limits = [1, 3]):
    weights = np.ones(N_s)
    poloidal_sections = []

    phi_collect = np.random.uniform(0, 1, N_s)
    toroid_radius_collect = np.random.uniform(0, 1, N_s)
    poloid_radius_collect = np.random.uniform(0, 1, N_s)
    ncollect = np.random.uniform(0, 1, 1)
    acollect = np.random.uniform(0, 1, 1)
    kcollect = np.random.uniform(0, 1, 1)

    theta_collect = np.random.uniform(0, 1, N_s)
    psi_collect = np.random.uniform(0, 1, N_s)

    theta_collect = theta_collect / theta_collect.sum()
    psi_collect = psi_collect / psi_collect.sum()
    do_gen = np.concatenate([toroid_radius_collect,
                            poloid_radius_collect,
                            phi_collect,
                            theta_collect, 
                            psi_collect,
                            ncollect,
                            acollect,
                            kcollect,
                            ])
    N_actual = int(N_limits[0] + ncollect[0] * (N_limits[1] - N_limits[0]))
    A_actual = A_limits[0] + acollect[0] * (A_limits[1] - A_limits[0])
    k_actual = k_limits[0] + kcollect[0] * (k_limits[1] - k_limits[0])
    sections = np.array(np.linspace(start=0.0, stop = 1.0 - (1 / cfg.NUM_T), num=cfg.NUM_T), dtype=np.float64)
    theta_actual = np.concatenate((
    [0.0],
    np.cumsum(theta_collect) * 360.0
    ))
    psi_actual = np.concatenate((
        [0.0],
        np.cumsum(psi_collect) * 360.0
    ))
    phi_actual = [90 + phi_limits[0] + x * (phi_limits[1] - phi_limits[0]) for x in phi_collect]
    """
    toroidal_sections = {
        "N_t": N_s,
        "phi": phi_actual,
        "theta": theta_actual[:-1],
        "radius": toroid_radius_collect,
        "weights": weights,
        "degree": 3,
        "sections": sections,
        "N" : N_actual,
        "A" : A_actual,
        "k" : k_actual
        }
    N_t = cfg.NUM_T
    for i in range(N_t):
        poloidal_sections.append({
                "N_s": N_s,
                "psi": psi_actual[:-1],
                "radius": poloid_radius_collect,
                "weights": weights,
                "degree": 3
            })
    """
    return do_gen

def decode_genome(do_gen, N_s,
                  phi_limits = [-30, 30], 
                 N_limits = [1, 6],
                 A_limits = [0.1, 1],
                 k_limits = [1, 3]):
    poloidal_sections = []
    weights = np.ones(N_s)
    idx = 0
    toroid_radius_collect = do_gen[idx : idx + N_s]
    idx += N_s

    poloid_radius_collect = do_gen[idx : idx + N_s]
    idx += N_s
    
    phi_collect = do_gen[idx : idx + N_s]
    idx += N_s
    
    theta_collect= do_gen[idx : idx + N_s]
    idx += N_s
    
    psi_collect = do_gen[idx : idx + N_s]
    idx += N_s
                            
    ncollect = do_gen[idx : idx + 1]
    idx += 1
    
    acollect = do_gen[idx : idx + 1]
    idx += 1
    
    kcollect= do_gen[idx : idx + 1]

    N_actual = int(N_limits[0] + ncollect[0] * (N_limits[1] - N_limits[0]))
    A_actual = A_limits[0] + acollect[0] * (A_limits[1] - A_limits[0])
    k_actual = k_limits[0] + kcollect[0] * (k_limits[1] - k_limits[0])
    sections = np.array(np.linspace(start=0.0, stop = 1.0 - (1 / cfg.NUM_T), num=cfg.NUM_T), dtype=np.float64)
    theta_actual = np.concatenate((
    [0.0],
    np.cumsum(theta_collect) * 360.0
    ))

    psi_actual = np.concatenate((
        [0.0],
        np.cumsum(psi_collect) * 360.0
    ))

    phi_actual = (
        90.0
        + phi_limits[0]
        + phi_collect * (phi_limits[1] - phi_limits[0])
    )
    toroidal_sections = {
        "N_t": N_s,
        "phi": phi_actual,
        "theta": theta_actual[:-1],
        "radius": toroid_radius_collect,
        "weights": weights,
        "degree": 3,
        "sections": sections,
        "N" : N_actual,
        "A" : A_actual,
        "k" : k_actual
        }

    for i in range(cfg.NUM_T):
        poloidal_sections.append({
                    "N_s": N_s,
                    "psi": psi_actual[:-1],
                    "radius": poloid_radius_collect,
                    "weights": weights,
                    "degree": 3
                })
    
    return toroidal_sections, poloidal_sections
        