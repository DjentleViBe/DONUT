import json

def write_geometry_parameters_to_file(toroidal_sections, poloidal_sections, filename):
    """Write the geometry parameters to an output file in JSON format.
    Args:
        toroidal_sections (list): List of toroidal section parameters.
        poloidal_sections (list): List of poloidal section parameters.
        data_format (dict): The data format dictionary containing the parameter names and values.
    """

    output = {
        "toroidal": toroidal_sections,
        "poloidal": poloidal_sections
    }

    with open(filename, "w") as f:
        json.dump(output, f, indent=4, default=str)