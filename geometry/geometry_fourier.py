"""
Data formatting
"""
import json
import numpy as np
import config as cfg
from geometry.geometry_reader import get_geometry_parameters_from_toroidal_file,\
                                     get_geometry_parameters_from_twist_file
from geometry.geometry_constraints import f_to_u, u_to_f
import geometry.geometry_process as gp

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
        pmval = np.array([twist_prop.get("Pm")], dtype=np.float64)
        bval = np.array([twist_prop.get("B")], dtype=np.float64)
        kval = np.array([twist_prop.get("k")], dtype=np.float64)
        combined = np.concatenate([phi_to_u, theta, radius, weights, pmval, bval, kval])

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
        ntval = format_list[0]
        for i in range(ntval):
            nsval = format_list[i + 1]

            psi_to_u = x0[idx : idx + nsval]
            idx += nsval

            radius = x0[idx : idx + nsval]
            idx += nsval

            weights = x0[idx : idx + nsval]
            idx += nsval

            psi = np.degrees(u_to_f(psi_to_u, 0))

            poloidal_sections.append({
                "N_s": nsval,
                "psi": psi,
                "radius": radius,
                "weights": weights,
                "degree": 3
            })
    elif cfg.STUDY_NAME == "Type2":
        ntval = cfg.NUM_T
        for i in range(ntval):
            idx = 0
            nsval = format_list[1]

            psi_to_u = x0[idx : idx + nsval]
            idx += nsval

            radius = x0[idx : idx + nsval]
            idx += nsval

            weights = x0[idx : idx + nsval]
            idx += nsval

            psi = np.degrees(u_to_f(psi_to_u, 0))

            poloidal_sections.append({
                "N_s": nsval,
                "psi": psi,
                "radius": radius,
                "weights": weights,
                "degree": 3
            })
    elif cfg.STUDY_NAME == "Type3":
        ntval = format_list[0]
        for i in range(ntval):
            nsval = format_list[i + 1]

            psi_to_u = x0[idx : idx + nsval]
            idx += nsval

            radius = x0[idx : idx + nsval]
            idx += nsval

            weights = x0[idx : idx + nsval]
            idx += nsval

            psi = np.degrees(psi_to_u)

            poloidal_sections.append({
                "N_s": nsval,
                "psi": psi,
                "radius": radius,
                "weights": weights,
                "degree": 3
            })
    # -----------------------
    # Toroidal reconstruction
    # -----------------------
    ntval = format_list[0]

    phi_to_u = x0[idx : idx + ntval]
    idx += ntval

    theta_to_x = x0[idx : idx + ntval]
    idx += ntval

    radius = x0[idx : idx + ntval]
    idx += ntval

    weights = x0[idx : idx + ntval]
    idx += ntval

    phi = np.degrees(u_to_f(phi_to_u, 0))
    theta = np.degrees(theta0 + theta_to_x * theta_delta)
    if cfg.STUDY_NAME == "Type1":
        sections = x0[idx : idx + ntval]
        toroidal_sections = {
        "N_t": ntval,
        "phi": phi,
        "theta": theta,
        "radius": radius,
        "weights": weights,
        "degree": 3,
        "deltas": sections
        }
    elif cfg.STUDY_NAME == "Type3":
        deltas = x0[idx : idx + ntval]
        toroidal_sections = {
        "N_t": ntval,
        "phi": phi,
        "theta": theta,
        "radius": radius,
        "weights": weights,
        "degree": 3,
        "deltas": deltas
        }
    elif cfg.STUDY_NAME == "Type2":
        sections = np.array(np.linspace(start=0.0,
                        stop = 1.0 - (1 / cfg.NUM_T),
                        num=cfg.NUM_T), dtype=np.float64)
        pmval = x0[idx : idx + 1]
        idx += 1
        gp.PM = pmval
        bval = x0[idx : idx + 1]
        idx += 1
        gp.B = bval
        kval = x0[idx : idx + 1]
        gp.K = kval
        idx += 1
        toroidal_sections = {
        "N_t": ntval,
        "phi": phi,
        "theta": theta,
        "radius": radius,
        "weights": weights,
        "degree": 3,
        "sections": sections,
        "Pm" : pmval,
        "B" : bval,
        "k" : kval
        }
    return toroidal_sections, poloidal_sections

def genetic_data(nsval = cfg.NUM_S):
    """
    Generate geometry data for GA
    """
    phi_collect = np.random.uniform(0, 1, nsval)
    toroid_radius_collect = np.random.uniform(0, 1, nsval)
    poloid_radius_collect = np.random.uniform(0, 1, nsval)
    ncollect = np.random.uniform(0, 1, 1)
    acollect = np.random.uniform(0, 1, 1)
    kcollect = np.random.uniform(0, 1, 1)

    theta_collect = np.random.uniform(0, 1, nsval)
    psi_collect = np.random.uniform(0, 1, nsval)

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
    return do_gen

def decode_genome(do_gen, nsval = cfg.NUM_S,
                  theta_limits = cfg.THETA_LIMITS,
                 pm_limits = cfg.N_LIMITS,
                 b_limits = cfg.B_LIMITS,
                 k_limits = cfg.K_LIMITS,
                 radius_plimits = cfg.RADIUS_P,
                 radius_tlimits = cfg.RADIUS_T):
    """
    decode genome information into poloid and toroid sections
    """
    poloidal_sections = []
    weights = np.ones(nsval)
    idx = 0
    toroid_radius_collect = do_gen[idx : idx + nsval]
    idx += nsval

    poloid_radius_collect = do_gen[idx : idx + nsval]
    idx += nsval

    phi_collect = do_gen[idx : idx + nsval]
    idx += nsval

    theta_collect= do_gen[idx : idx + nsval]
    idx += nsval

    psi_collect = do_gen[idx : idx + nsval]
    idx += nsval

    ncollect = do_gen[idx : idx + 1]
    idx += 1

    acollect = do_gen[idx : idx + 1]
    idx += 1

    kcollect= do_gen[idx : idx + 1]

    pm_actual = pm_limits[0] + ncollect[0] * (pm_limits[1] - pm_limits[0])
    b_actual = b_limits[0] + acollect[0] * (b_limits[1] - b_limits[0])
    k_actual = int(k_limits[0] + kcollect[0] * (k_limits[1] - k_limits[0]))
    sections = np.array(np.linspace(start=0.0,
                        stop = 1.0 - (1 / cfg.NUM_T),
                        num=cfg.NUM_T), dtype=np.float64)
    phi_collect = phi_collect / phi_collect.sum()
    phi_actual = np.concatenate((
    [0.0],
    np.cumsum(phi_collect) * 360.0
    ))
    psi_collect = psi_collect / psi_collect.sum()
    psi_actual = np.concatenate((
        [0.0],
        np.cumsum(psi_collect) * 360.0
    ))

    theta_actual = (
        90.0
        + theta_limits[0]
        + theta_collect * (theta_limits[1] - theta_limits[0])
    )

    poloid_radius_actual = (
        radius_plimits[0]
        + poloid_radius_collect * (radius_plimits[1] - radius_plimits[0])
    )

    toroid_radius_actual = (
        radius_tlimits[0]
        + toroid_radius_collect * (radius_tlimits[1] - radius_tlimits[0])
    )
    toroidal_sections = {
        "N_t": nsval,
        "phi": phi_actual[:-1],
        "theta": theta_actual,
        "radius": toroid_radius_actual,
        "weights": weights,
        "degree": 3,
        "sections": sections,
        "Pm" : pm_actual,
        "B" : b_actual,
        "k" : k_actual
        }
    gp.PM = [pm_actual]
    gp.B = [b_actual]
    gp.K = [k_actual]
    for _ in range(cfg.NUM_T):
        poloidal_sections.append({
                    "N_s": nsval,
                    "psi": psi_actual[:-1],
                    "radius": poloid_radius_actual,
                    "weights": weights,
                    "degree": 3
                })
    return toroidal_sections, poloidal_sections
