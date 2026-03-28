from launch_geometry import geometry_init, geometry_construct
from geometry.geometry_reader import linearize_data

def geometry_process():
    poloidal_sections, toroid_file = geometry_init()
    geometry_construct(poloidal_sections, toroid_file)
    linearize_data("./inputs/toroidal_section.json")