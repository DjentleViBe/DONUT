"""
Build geometry
"""
import numpy as np
from stl import mesh

def loft_revolved(curves):
    """
    curves: list of arrays [(N,3), (N,3), ...]
    periodic in curve direction
    """
    ncval = len(curves)
    nval = curves[0].shape[0]

    # stack vertices
    vertices = np.vstack(curves)

    faces = []

    for j in range(ncval):
        j_next = (j + 1) % ncval

        for i in range(nval - 1):
            a = j * nval + i
            b = j * nval + i + 1
            c = j_next * nval + i
            d = j_next * nval + i + 1

            # quad → 2 triangles
            faces.append([a, b, c])
            faces.append([b, d, c])

    return vertices, np.array(faces)

def write_stl(vertices, faces, filename="revolve.stl"):
    """
    Write .stl to file
    """
    m = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        m.vectors[i] = vertices[f]
    m.save(filename)

def merge_stls(file_list, output="merged.stl"):
    """
    Merge given .stl files
    """
    meshes = [mesh.Mesh.from_file(f) for f in file_list]

    combined_data = np.concatenate([m.data for m in meshes])

    merged = mesh.Mesh(combined_data.copy())
    merged.save(output)
