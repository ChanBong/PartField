import open3d as o3d
import numpy as np
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "exp_results/clustering/objaverse_test/ply/002e462c8bfa4267a9c9f038c7966f3b_0_05.ply"

mesh = o3d.io.read_triangle_mesh(path)
print(f"Vertices: {len(mesh.vertices)}")
print(f"Triangles: {len(mesh.triangles)}")
print(f"Has vertex colors: {mesh.has_vertex_colors()}")
print(f"Has vertex normals: {mesh.has_vertex_normals()}")

colors = np.asarray(mesh.vertex_colors)
print(f"Vertex color shape: {colors.shape}")
if len(colors) > 0:
    print(f"Color range: {colors.min():.3f} - {colors.max():.3f}")
    unique = np.unique(colors, axis=0)
    print(f"Unique colors: {len(unique)}")
    for i, c in enumerate(unique[:10]):
        print(f"  Color {i}: RGB({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f})")
else:
    print("NO vertex colors!")

# Also check the raw PLY header
with open(path, 'rb') as f:
    header = b""
    while True:
        line = f.readline()
        header += line
        if b"end_header" in line:
            break
    print(f"\nPLY Header:\n{header.decode('ascii', errors='replace')}")
