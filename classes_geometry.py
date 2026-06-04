"""
Contains data classes required for geometry construction
"""
from dataclasses import dataclass

@dataclass
class DonutGenetic:
    """
    Genetic data for geoemtry"""
    phi_collect:list
    theta_collect:list
    toroid_radius_collect:list
    sections_collect:list
    ncollect:float
    acollect:float
    kcollect:float

@dataclass
class GeometryData:
    """
    Geometry data for construction
    """
    x_collections: list
    y_collections: list
    x_moved_collections: list
    y_moved_collections: list
    z_moved_collections: list
    ctrl_x_collections: list
    ctrl_y_collections: list

@dataclass
class ToroidalGeometry:
    """
    Toroid geometry content
    """
    toroidal_coordinates: list
    toroidal_tangents: list

@dataclass
class PoloidalGeometry:
    """
    Poloid geometry content
    """
    x: list
    y: list
    z : list
    ctrl_x: list
    ctrl_y: list
    ctrl_z: list

@dataclass
class SketchGeometryToroidal:
    """
    2D sketch
    """
    curve: list
    ctrl: list

@dataclass
class SketchGeometryPoloidal:
    """
    2D sketch
    """
    x_p: list
    y_p: list
    ctrl_xp:list
    ctrl_yp:list
