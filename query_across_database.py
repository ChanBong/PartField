"""1-to-Many Feature Query across a database of parts.

Polyscope GUI that loads ALL parts' PartField features, lets you click
faces on a query part, and ranks every face across ALL other parts by
cosine similarity.

Usage (from PartField repo root):
    python query_across_database.py --data_root exp_results/partfield_features/step200

Requirements: numpy, trimesh, polyscope, torch
"""

import os
import sys
import glob
import argparse
import numpy as np
import trimesh
import polyscope as ps
import polyscope.imgui as psim

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_mesh_util(input_fname):
    """Load a mesh with trimesh (same as partfield.utils)."""
    return trimesh.load(input_fname, force='mesh', process=False)


def discover_parts(data_root):
    """Scan data_root for PartField output files and return a list of part names.

    PartField saves:
      - part_feat_{name}_0.npy  OR  part_feat_{name}_0_batch.npy
      - feat_pca_{name}_0.ply
    We detect unique part names from feat_pca_*_0.ply files.
    """
    ply_files = sorted(glob.glob(os.path.join(data_root, "feat_pca_*_0.ply")))
    names = []
    for p in ply_files:
        basename = os.path.basename(p)  # feat_pca_{name}_0.ply
        # Strip prefix "feat_pca_" and suffix "_0.ply"
        name = basename[len("feat_pca_"):-len("_0.ply")]
        names.append(name)
    return names


def feature_path(data_root, name):
    """Return the feature .npy path for a part name (handles _batch suffix)."""
    p = os.path.join(data_root, f"part_feat_{name}_0.npy")
    if os.path.exists(p):
        return p
    return os.path.join(data_root, f"part_feat_{name}_0_batch.npy")


def mesh_path(data_root, name):
    """Return the PCA-colored PLY path for a part name."""
    return os.path.join(data_root, f"feat_pca_{name}_0.ply")


# ---------------------------------------------------------------------------
# Database: pre-load all features into one big matrix
# ---------------------------------------------------------------------------

class PartDatabase:
    """Holds L2-normalized face features for every part in one contiguous array."""

    def __init__(self, data_root):
        self.data_root = data_root
        self.part_names = discover_parts(data_root)
        assert len(self.part_names) > 0, f"No parts found in {data_root}"

        all_feats = []       # list of (N_i, D) arrays
        self.part_offsets = []  # (start, end) index into the big array
        self.part_nfaces = {}   # name -> num faces

        offset = 0
        print(f"Loading {len(self.part_names)} parts from {data_root} ...")
        for i, name in enumerate(self.part_names):
            feat = np.load(feature_path(data_root, name), allow_pickle=True).astype(np.float32)
            n = feat.shape[0]
            all_feats.append(feat)
            self.part_offsets.append((offset, offset + n))
            self.part_nfaces[name] = n
            offset += n
            if (i + 1) % 50 == 0 or (i + 1) == len(self.part_names):
                print(f"  loaded {i + 1}/{len(self.part_names)} parts, {offset} total faces")

        # Build big matrix and L2-normalize rows
        self.all_features = np.vstack(all_feats)  # (total_faces, D)
        norms = np.linalg.norm(self.all_features, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        self.all_features = self.all_features / norms

        self.total_faces = self.all_features.shape[0]
        self.feat_dim = self.all_features.shape[1]
        print(f"Database ready: {len(self.part_names)} parts, "
              f"{self.total_faces} faces, {self.feat_dim}D features, "
              f"{self.all_features.nbytes / 1e6:.1f} MB")

    def query(self, query_feat):
        """Compute cosine similarity of query_feat against all faces.

        Args:
            query_feat: (D,) L2-normalized feature vector
        Returns:
            scores: (total_faces,) cosine similarity in [-1, 1]
        """
        return self.all_features @ query_feat

    def rank_parts(self, query_feat, exclude_name=None):
        """Rank all parts by how well they match a query feature.

        Returns a list of dicts sorted by best_sim descending:
            [{'name': str, 'best_sim': float, 'avg_sim': float,
              'n_good': int, 'scores': np.array}, ...]
        """
        all_scores = self.query(query_feat)

        results = []
        for i, name in enumerate(self.part_names):
            if name == exclude_name:
                continue
            start, end = self.part_offsets[i]
            scores = all_scores[start:end]
            results.append({
                'name': name,
                'best_sim': float(np.max(scores)),
                'avg_sim': float(np.mean(scores)),
                'n_good': int(np.sum(scores > 0.5)),
                'scores': scores,
            })

        results.sort(key=lambda r: r['best_sim'], reverse=True)
        return results


# ---------------------------------------------------------------------------
# GUI State
# ---------------------------------------------------------------------------

class AppState:
    def __init__(self, db: PartDatabase):
        self.db = db

        # Part selector
        self.part_names = db.part_names
        self.query_idx = 0         # index into part_names
        self.query_name = None     # currently loaded query part name
        self.query_mesh = None     # dict with V, F, feat_np, etc.

        # Selection
        self.selected_faces = set()

        # Search results
        self.results = []          # output of db.rank_parts()
        self.result_scroll_idx = 0
        self.result_view_idx = -1  # which result is being visualized

        # Visualization
        self.feature_range = 0.3   # heatmap range for distance coloring
        self.result_mesh = None    # loaded result mesh dict
        self.auto_search = False

        # Filter
        self.filter_text = ""


def load_part_mesh(db, name):
    """Load mesh + features for a part, return dict compatible with shape_pair.py."""
    feat = np.load(feature_path(db.data_root, name), allow_pickle=True).astype(np.float32)
    tm = load_mesh_util(mesh_path(db.data_root, name))
    V = np.array(tm.vertices, dtype=np.float32)
    F = np.array(tm.faces)
    pca_colors = np.array(tm.visual.face_colors, dtype=np.float32)[:, :3] / 255.0

    # L2-normalize features
    norms = np.linalg.norm(feat, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    feat_normed = feat / norms

    return {
        'V': V,
        'F': F,
        'pca_colors': pca_colors,
        'feat_np': feat,
        'feat_normed': feat_normed,
        'trimesh': tm,
    }


def load_query_part(state):
    """Load selected query part into polyscope."""
    name = state.part_names[state.query_idx]

    # Remove old meshes
    if state.query_name is not None:
        ps.remove_all_structures()

    state.query_name = name
    state.query_mesh = load_part_mesh(state.db, name)
    state.selected_faces = set()
    state.results = []
    state.result_view_idx = -1
    state.result_mesh = None

    m = state.query_mesh
    ps_mesh = ps.register_surface_mesh("query", m['V'], m['F'])
    ps_mesh.set_selection_mode('faces_only')
    ps_mesh.add_color_quantity('pca_colors', m['pca_colors'], defined_on='faces', enabled=True)
    state.query_mesh['ps_mesh'] = ps_mesh

    print(f"Loaded query part: {name} ({m['F'].shape[0]} faces)")


def compute_query_feature(state):
    """Average the L2-normed features of selected faces to produce query vector."""
    if not state.selected_faces:
        return None
    m = state.query_mesh
    indices = list(state.selected_faces)
    avg = np.mean(m['feat_normed'][indices], axis=0)
    # Re-normalize
    norm = np.linalg.norm(avg)
    if norm < 1e-8:
        return None
    return avg / norm


def run_search(state):
    """Execute the search and populate results."""
    qfeat = compute_query_feature(state)
    if qfeat is None:
        print("No faces selected!")
        return
    state.results = state.db.rank_parts(qfeat, exclude_name=state.query_name)
    state.result_scroll_idx = 0
    state.result_view_idx = -1
    print(f"Search done. Top match: {state.results[0]['name']} "
          f"(best_sim={state.results[0]['best_sim']:.4f})")


def show_result(state, result_idx):
    """Load a result part side-by-side and show heatmap."""
    if result_idx < 0 or result_idx >= len(state.results):
        return

    result = state.results[result_idx]
    name = result['name']

    # Remove previous result mesh
    if state.result_mesh is not None:
        try:
            ps.remove_surface_mesh("result")
        except Exception:
            pass

    state.result_view_idx = result_idx
    state.result_mesh = load_part_mesh(state.db, name)

    m = state.result_mesh
    ps_mesh = ps.register_surface_mesh("result", m['V'], m['F'])

    # Position to the right of query
    query_bounds = state.query_mesh['trimesh'].bounds
    query_width = query_bounds[1][0] - query_bounds[0][0]
    result_bounds = m['trimesh'].bounds
    result_width = result_bounds[1][0] - result_bounds[0][0]
    gap = max(query_width, result_width) * 0.3
    offset = query_width / 2 + gap + result_width / 2
    ps_mesh.translate((offset, 0, 0))

    # Compute heatmap: cosine distance = (1 - cos_sim) / 2
    scores = result['scores']
    distances = (1.0 - scores) / 2.0
    ps_mesh.add_scalar_quantity("match_distance", distances, cmap='blues',
                                vminmax=(0, state.feature_range),
                                defined_on='faces', enabled=True)

    state.result_mesh['ps_mesh'] = ps_mesh
    print(f"Showing result: {name} (best_sim={result['best_sim']:.4f})")


# ---------------------------------------------------------------------------
# Polyscope callback
# ---------------------------------------------------------------------------

def ps_callback(state: AppState):
    psim.SetNextWindowSize((420, 0))
    opened, _ = psim.Begin("Query Across Database", True)
    if not opened:
        psim.End()
        return

    # ---- Part selector ----
    psim.TextUnformatted("=== Query Part ===")

    changed, state.query_idx = psim.Combo("Part", state.query_idx, state.part_names)
    if changed:
        load_query_part(state)

    if state.query_mesh is not None:
        n_faces = state.query_mesh['F'].shape[0]
        psim.TextUnformatted(f"Faces: {n_faces}, Selected: {len(state.selected_faces)}")

    # ---- Face selection via click ----
    psim.Separator()
    psim.TextUnformatted("=== Face Selection ===")
    psim.TextUnformatted("Click faces on query mesh to select.")
    psim.TextUnformatted("Shift+click to deselect.")

    io = psim.GetIO()
    if io.MouseClicked[0] and state.query_mesh is not None:
        screen_coords = io.MousePos

        # Check if mouse is over the ImGui window — if so, skip picking
        if not io.WantCaptureMouse:
            pick_result = ps.pick(screen_coords=screen_coords)

            if pick_result.is_hit and pick_result.structure_name == "query":
                if pick_result.structure_data['element_type'] == "face":
                    f_hit = pick_result.structure_data['index']
                    shift_held = io.KeyShift

                    if shift_held and f_hit in state.selected_faces:
                        state.selected_faces.discard(f_hit)
                    else:
                        state.selected_faces.add(f_hit)

                    # Update selection visualization
                    sel_color = np.copy(state.query_mesh['pca_colors'])
                    for fi in state.selected_faces:
                        sel_color[fi] = [1.0, 0.0, 0.0]  # red

                    state.query_mesh['ps_mesh'].add_color_quantity(
                        'pca_colors', sel_color, defined_on='faces', enabled=True)

                    # Also show distance heatmap on query mesh
                    qfeat = compute_query_feature(state)
                    if qfeat is not None:
                        dists = (1.0 - state.query_mesh['feat_normed'] @ qfeat) / 2.0
                        state.query_mesh['ps_mesh'].add_scalar_quantity(
                            "self_distance", dists, cmap='blues',
                            vminmax=(0, state.feature_range),
                            defined_on='faces', enabled=False)

                    if state.auto_search:
                        run_search(state)
                        if state.results:
                            show_result(state, 0)

    if psim.Button("Clear Selection"):
        state.selected_faces = set()
        if state.query_mesh is not None:
            state.query_mesh['ps_mesh'].add_color_quantity(
                'pca_colors', state.query_mesh['pca_colors'],
                defined_on='faces', enabled=True)

    psim.SameLine()
    _, state.auto_search = psim.Checkbox("Auto-search", state.auto_search)

    # ---- Search ----
    psim.Separator()
    psim.TextUnformatted("=== Search ===")

    if psim.Button("Search All Parts") and state.selected_faces:
        run_search(state)
        if state.results:
            show_result(state, 0)

    changed, state.feature_range = psim.SliderFloat("Distance range",
                                                     state.feature_range,
                                                     v_min=0.01, v_max=0.5)
    if changed and state.result_mesh is not None and state.result_view_idx >= 0:
        # Update heatmap range on result
        result = state.results[state.result_view_idx]
        distances = (1.0 - result['scores']) / 2.0
        state.result_mesh['ps_mesh'].add_scalar_quantity(
            "match_distance", distances, cmap='blues',
            vminmax=(0, state.feature_range),
            defined_on='faces', enabled=True)

    # ---- Results list ----
    if state.results:
        psim.Separator()
        psim.TextUnformatted(f"=== Results ({len(state.results)} parts) ===")

        # Show top results in a scrollable list
        max_show = min(20, len(state.results))
        for i in range(max_show):
            r = state.results[i]
            marker = ">>>" if i == state.result_view_idx else "   "
            label = f"{marker} #{i+1} {r['name'][:25]}  best={r['best_sim']:.3f}  good={r['n_good']}"
            if psim.Button(label):
                show_result(state, i)

        if len(state.results) > max_show:
            psim.TextUnformatted(f"  ... and {len(state.results) - max_show} more")

    psim.End()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Query features across a part database")
    parser.add_argument('--data_root', required=True,
                        help='Path to PartField feature directory (e.g. exp_results/partfield_features/step200)')
    args = parser.parse_args()

    db = PartDatabase(args.data_root)
    state = AppState(db)

    ps.init()
    ps.set_ground_plane_mode("none")

    # Load first part
    load_query_part(state)

    ps.set_user_callback(lambda: ps_callback(state))
    ps.show()


if __name__ == "__main__":
    main()
