"""Quick visualizer for PartField results.

Usage:
    python visualize_results.py --mode pca       # View PCA feature coloring
    python visualize_results.py --mode cluster    # View part segmentation (default: 5 clusters)
    python visualize_results.py --mode cluster --k 10  # View with 10 clusters
    python visualize_results.py --mode compare    # Side-by-side: input vs PCA vs segmentation
    python visualize_results.py --mode all        # Show all models sequentially
"""
import argparse
import glob
import os
import open3d as o3d
import numpy as np
from plyfile import PlyData


FEAT_DIR = "exp_results/partfield_features/objaverse_test"
CLUSTER_DIR = "exp_results/clustering/objaverse_test/ply"


def get_model_ids():
    """Find all model IDs from the feature directory."""
    npy_files = glob.glob(os.path.join(FEAT_DIR, "part_feat_*_batch.npy"))
    ids = []
    for f in npy_files:
        basename = os.path.basename(f)
        model_id = basename.replace("part_feat_", "").replace("_0_batch.npy", "")
        ids.append(model_id)
    return sorted(ids)


def load_mesh_with_face_colors(path):
    """Load a PLY mesh and convert per-face colors to per-vertex colors for visualization."""
    plydata = PlyData.read(path)

    # Extract vertices
    vx = plydata['vertex']['x']
    vy = plydata['vertex']['y']
    vz = plydata['vertex']['z']
    vertices = np.column_stack([vx, vy, vz])

    # Extract faces
    face_data = plydata['face']
    faces = np.vstack(face_data['vertex_indices'])

    # Extract per-face colors
    has_face_color = 'red' in face_data.data.dtype.names
    face_colors = None
    if has_face_color:
        fr = face_data['red'].astype(np.float64) / 255.0
        fg = face_data['green'].astype(np.float64) / 255.0
        fb = face_data['blue'].astype(np.float64) / 255.0
        face_colors = np.column_stack([fr, fg, fb])

    # Build Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()

    if face_colors is not None:
        # Convert face colors to vertex colors by averaging incident face colors
        vertex_colors = np.zeros((len(vertices), 3), dtype=np.float64)
        vertex_counts = np.zeros(len(vertices), dtype=np.float64)
        for fi in range(len(faces)):
            for vi in faces[fi]:
                vertex_colors[vi] += face_colors[fi]
                vertex_counts[vi] += 1.0
        vertex_counts[vertex_counts == 0] = 1.0
        vertex_colors /= vertex_counts[:, None]
        mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

    return mesh


def show_mesh(mesh, title="PartField"):
    """Display mesh with colors visible (no default lighting override)."""
    o3d.visualization.draw_geometries(
        [mesh],
        window_name=title,
        width=1024,
        height=768,
        mesh_show_back_face=True,
    )


def show_pca(model_id):
    """Show PCA-colored feature visualization."""
    pca_path = os.path.join(FEAT_DIR, f"feat_pca_{model_id}_0.ply")
    if not os.path.exists(pca_path):
        print(f"PCA file not found: {pca_path}")
        print("(First model may be missing PCA due to caching. Re-run inference to regenerate.)")
        return
    mesh = load_mesh_with_face_colors(pca_path)
    print(f"\n--- PCA Feature Visualization: {model_id[:8]}... ---")
    print("Colors = first 3 PCA components of the 448-dim feature vectors mapped to RGB.")
    print("Similar colors = similar features. Close the window to continue.")
    show_mesh(mesh, f"PCA Features - {model_id[:8]}")


def show_cluster(model_id, k=5):
    """Show part segmentation at k clusters."""
    cluster_path = os.path.join(CLUSTER_DIR, f"{model_id}_0_{k:02d}.ply")
    if not os.path.exists(cluster_path):
        print(f"Cluster file not found: {cluster_path}")
        return
    mesh = load_mesh_with_face_colors(cluster_path)
    print(f"\n--- Part Segmentation ({k} clusters): {model_id[:8]}... ---")
    print("Each color = one detected part. Close the window to continue.")
    show_mesh(mesh, f"Segmentation (k={k}) - {model_id[:8]}")


def show_input(model_id):
    """Show original input mesh."""
    input_path = os.path.join(FEAT_DIR, f"input_{model_id}_0.ply")
    if not os.path.exists(input_path):
        print(f"Input file not found: {input_path}")
        return
    pcd = o3d.io.read_point_cloud(input_path)
    print(f"\n--- Input Point Cloud: {model_id[:8]}... ---")
    print(f"Points: {len(pcd.points)}. Close the window to continue.")
    o3d.visualization.draw_geometries([pcd], window_name=f"Input - {model_id[:8]}", width=1024, height=768)


def show_cluster_progression(model_id):
    """Show segmentation at multiple granularities."""
    for k in [2, 5, 10, 15, 20]:
        cluster_path = os.path.join(CLUSTER_DIR, f"{model_id}_0_{k:02d}.ply")
        if os.path.exists(cluster_path):
            mesh = load_mesh_with_face_colors(cluster_path)
            print(f"\n--- {k} clusters: {model_id[:8]}... --- (close window for next)")
            show_mesh(mesh, f"k={k} - {model_id[:8]}")


def main():
    parser = argparse.ArgumentParser(description="Visualize PartField results")
    parser.add_argument("--mode", choices=["pca", "cluster", "compare", "progression", "all"],
                        default="cluster", help="Visualization mode")
    parser.add_argument("--k", type=int, default=5, help="Number of clusters for cluster mode")
    parser.add_argument("--model", type=int, default=None, help="Model index (0, 1, ...)")
    args = parser.parse_args()

    model_ids = get_model_ids()
    if not model_ids:
        print("No results found. Run inference first.")
        return

    print(f"Found {len(model_ids)} models:")
    for i, mid in enumerate(model_ids):
        print(f"  [{i}] {mid}")

    if args.model is not None:
        model_ids = [model_ids[args.model]]

    for mid in model_ids:
        if args.mode == "pca":
            show_pca(mid)
        elif args.mode == "cluster":
            show_cluster(mid, args.k)
        elif args.mode == "compare":
            show_input(mid)
            show_pca(mid)
            show_cluster(mid, args.k)
        elif args.mode == "progression":
            show_cluster_progression(mid)
        elif args.mode == "all":
            show_input(mid)
            show_pca(mid)
            show_cluster_progression(mid)


if __name__ == "__main__":
    main()
