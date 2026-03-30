from launch_geometry import geometry_init, geometry_construct
from geometry.geometry_reader import linearize_data, delinearize_data

def geometry_process():
    #poloidal_sections, toroidal_sections = geometry_init()
    x0, format = linearize_data("./inputs/toroidal_section.json")
    toroidal_sections, poloidal_sections = delinearize_data(x0, format)
    geometry_construct(toroidal_sections, poloidal_sections, 1)
    