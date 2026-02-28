import numpy as np
from stl import mesh

def loft_revolved(curves):
    """
    curves: list of arrays [(N,3), (N,3), ...]
    periodic in curve direction
    """
    Nc = len(curves)
    N = curves[0].shape[0]

    # stack vertices
    vertices = np.vstack(curves)

    faces = []

    for j in range(Nc):
        j_next = (j + 1) % Nc

        for i in range(N - 1):
            a = j * N + i
            b = j * N + i + 1
            c = j_next * N + i
            d = j_next * N + i + 1

            # quad → 2 triangles
            faces.append([a, b, c])
            faces.append([b, d, c])

    return vertices, np.array(faces)

def write_stl(vertices, faces, filename="revolve.stl"):
    m = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        m.vectors[i] = vertices[f]
    m.save(filename)

def merge_stls(file_list, output="merged.stl"):
    meshes = [mesh.Mesh.from_file(f) for f in file_list]

    combined_data = np.concatenate([m.data for m in meshes])

    merged = mesh.Mesh(combined_data.copy())
    merged.save(output)
