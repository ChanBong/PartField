"""1-to-Many Feature Query with B-Rep face selection.

Same as query_across_database.py but uses face_map.npy files to let you
click once and select an entire B-Rep face (hole, fillet, planar surface)
instead of individual triangles.

Usage (from PartField repo root):
    python query_across_database_brep.py \
        --data_root exp_results/step200_brep \
        --obj_dir data/step-200-obj-brep

Requirements: numpy, trimesh, polyscope
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
    return trimesh.load(input_fname, force='mesh', process=False)


def discover_parts(data_root):
    ply_files = sorted(glob.glob(os.path.join(data_root, "feat_pca_*_0.ply")))
    names = []
    for p in ply_files:
        basename = os.path.basename(p)
        name = basename[len("feat_pca_"):-len("_0.ply")]
        names.append(name)
    return names


def feature_path(data_root, name):
    p = os.path.join(data_root, f"part_feat_{name}_0.npy")
    if os.path.exists(p):
        return p
    return os.path.join(data_root, f"part_feat_{name}_0_batch.npy")


def mesh_path(data_root, name):
    return os.path.join(data_root, f"feat_pca_{name}_0.ply")


def face_map_path(obj_dir, name):
    return os.path.join(obj_dir, f"{name}_face_map.npy")


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

class PartDatabase:
    """Lazy-loading database. Features are read per-part during search
    to avoid loading 3M+ triangles into one giant matrix."""

    def __init__(self, data_root, obj_dir):
        self.data_root = data_root
        self.obj_dir = obj_dir
        self.part_names = discover_parts(data_root)
        assert len(self.part_names) > 0, f"No parts found in {data_root}"

        # Check face maps exist
        missing = [n for n in self.part_names
                   if not os.path.exists(face_map_path(obj_dir, n))]
        if missing:
            print(f"WARNING: {len(missing)} parts missing face_map.npy in {obj_dir}")
            print(f"  First few: {missing[:5]}")
            print(f"  These parts will not have B-Rep face selection.")

        print(f"Database ready: {len(self.part_names)} parts (lazy loading)")

    def _load_normed_features(self, name):
        """Load and L2-normalize features for a single part."""
        feat = np.load(feature_path(self.data_root, name), allow_pickle=True).astype(np.float32)
        norms = np.linalg.norm(feat, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        return feat / norms

    def rank_parts(self, query_feat, exclude_name=None):
        results = []
        for i, name in enumerate(self.part_names):
            if name == exclude_name:
                continue
            feat = self._load_normed_features(name)
            scores = feat @ query_feat
            results.append({
                'name': name,
                'best_sim': float(np.max(scores)),
                'avg_sim': float(np.mean(scores)),
                'n_good': int(np.sum(scores > 0.5)),
                'scores': scores,
            })
            if (i + 1) % 50 == 0:
                print(f"  searched {i + 1}/{len(self.part_names)} parts...")
        results.sort(key=lambda r: r['best_sim'], reverse=True)
        return results


# ---------------------------------------------------------------------------
# GUI State
# ---------------------------------------------------------------------------

class AppState:
    def __init__(self, db: PartDatabase):
        self.db = db

        self.part_names = db.part_names
        self.query_idx = 0
        self.query_name = None
        self.query_mesh = None

        # B-Rep face selection (set of B-Rep face IDs, not triangle indices)
        self.selected_brep_faces = set()

        # Search results
        self.results = []
        self.result_view_idx = -1

        # Visualization
        self.feature_range = 0.3
        self.result_mesh = None
        self.auto_search = False


# ---------------------------------------------------------------------------
# Mesh loading
# ---------------------------------------------------------------------------

def load_part_mesh(db, name):
    feat = np.load(feature_path(db.data_root, name), allow_pickle=True).astype(np.float32)
    tm = load_mesh_util(mesh_path(db.data_root, name))
    V = np.array(tm.vertices, dtype=np.float32)
    F = np.array(tm.faces)
    pca_colors = np.array(tm.visual.face_colors, dtype=np.float32)[:, :3] / 255.0

    norms = np.linalg.norm(feat, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    feat_normed = feat / norms

    # Load B-Rep face map if available
    fmap_path = face_map_path(db.obj_dir, name)
    if os.path.exists(fmap_path):
        fmap = np.load(fmap_path)
        n_brep = int(fmap.max()) + 1
    else:
        fmap = None
        n_brep = 0

    return {
        'V': V,
        'F': F,
        'pca_colors': pca_colors,
        'feat_np': feat,
        'feat_normed': feat_normed,
        'trimesh': tm,
        'face_map': fmap,       # int32 array, length = num triangles
        'n_brep_faces': n_brep,
    }


def normalize_vertices(V, offset=np.zeros(3)):
    """Center at origin, scale to unit bounding box, then shift by offset."""
    centroid = V.mean(axis=0)
    V_centered = V - centroid
    scale = np.abs(V_centered).max()
    if scale > 1e-8:
        V_centered = V_centered / scale
    return V_centered + offset


def get_triangles_for_brep_faces(face_map, brep_face_ids):
    """Return set of triangle indices belonging to any of the given B-Rep face IDs."""
    if face_map is None or not brep_face_ids:
        return set()
    mask = np.isin(face_map, list(brep_face_ids))
    return set(np.where(mask)[0])


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

def load_query_part(state):
    name = state.part_names[state.query_idx]

    if state.query_name is not None:
        ps.remove_all_structures()

    state.query_name = name
    state.query_mesh = load_part_mesh(state.db, name)
    state.selected_brep_faces = set()
    state.results = []
    state.result_view_idx = -1
    state.result_mesh = None

    m = state.query_mesh
    V_query = normalize_vertices(m['V'], offset=np.array([-1.25, 0., 0.]))
    ps_mesh = ps.register_surface_mesh("query", V_query, m['F'])
    ps_mesh.set_selection_mode('faces_only')
    ps_mesh.add_color_quantity('pca_colors', m['pca_colors'], defined_on='faces', enabled=True)
    state.query_mesh['ps_mesh'] = ps_mesh

    fmap_status = f", {m['n_brep_faces']} B-Rep faces" if m['face_map'] is not None else ", NO face map"
    print(f"Loaded query part: {name} ({m['F'].shape[0]} triangles{fmap_status})")


def compute_query_feature(state):
    if not state.selected_brep_faces:
        return None
    m = state.query_mesh
    tri_indices = get_triangles_for_brep_faces(m['face_map'], state.selected_brep_faces)
    if not tri_indices:
        return None
    indices = list(tri_indices)
    avg = np.mean(m['feat_normed'][indices], axis=0)
    norm = np.linalg.norm(avg)
    if norm < 1e-8:
        return None
    return avg / norm


def run_search(state):
    qfeat = compute_query_feature(state)
    if qfeat is None:
        print("No faces selected!")
        return
    state.results = state.db.rank_parts(qfeat, exclude_name=state.query_name)
    state.result_view_idx = -1
    print(f"Search done. Top match: {state.results[0]['name']} "
          f"(best_sim={state.results[0]['best_sim']:.4f})")


def show_result(state, result_idx):
    if result_idx < 0 or result_idx >= len(state.results):
        return

    result = state.results[result_idx]
    name = result['name']

    if state.result_mesh is not None:
        try:
            ps.remove_surface_mesh("result")
        except Exception:
            pass

    state.result_view_idx = result_idx
    state.result_mesh = load_part_mesh(state.db, name)

    m = state.result_mesh
    V_result = normalize_vertices(m['V'], offset=np.array([1.25, 0., 0.]))
    ps_mesh = ps.register_surface_mesh("result", V_result, m['F'])

    # Similarity heatmap: red/hot = good match, blue/cold = no match
    scores = result['scores']
    ps_mesh.add_scalar_quantity("similarity", scores, cmap='turbo',
                                vminmax=(1.0 - state.feature_range, 1.0),
                                defined_on='faces', enabled=True)

    state.result_mesh['ps_mesh'] = ps_mesh
    print(f"Showing result: {name} (best_sim={result['best_sim']:.4f})")


def update_selection_viz(state):
    """Redraw the query mesh colors to show selected B-Rep faces in red."""
    m = state.query_mesh
    sel_tris = get_triangles_for_brep_faces(m['face_map'], state.selected_brep_faces)

    sel_color = np.copy(m['pca_colors'])
    for ti in sel_tris:
        sel_color[ti] = [1.0, 0.0, 0.0]

    m['ps_mesh'].add_color_quantity('pca_colors', sel_color, defined_on='faces', enabled=True)

    # Self-similarity heatmap (disabled by default, user can toggle)
    qfeat = compute_query_feature(state)
    if qfeat is not None:
        self_sim = m['feat_normed'] @ qfeat
        m['ps_mesh'].add_scalar_quantity(
            "self_similarity", self_sim, cmap='turbo',
            vminmax=(1.0 - state.feature_range, 1.0),
            defined_on='faces', enabled=False)


# ---------------------------------------------------------------------------
# Polyscope callback
# ---------------------------------------------------------------------------

def ps_callback(state: AppState):
    psim.SetNextWindowSize((440, 0))
    opened, _ = psim.Begin("Query Across Database (B-Rep)", True)
    if not opened:
        psim.End()
        return

    # ---- Part selector ----
    psim.TextUnformatted("=== Query Part ===")
    changed, state.query_idx = psim.Combo("Part", state.query_idx, state.part_names)
    if changed:
        load_query_part(state)

    if state.query_mesh is not None:
        m = state.query_mesh
        n_tris = m['F'].shape[0]
        n_brep = m['n_brep_faces']
        n_sel_brep = len(state.selected_brep_faces)
        n_sel_tris = len(get_triangles_for_brep_faces(m['face_map'], state.selected_brep_faces))

        if m['face_map'] is not None:
            psim.TextUnformatted(f"{n_tris} triangles, {n_brep} B-Rep faces")
            psim.TextUnformatted(f"Selected: {n_sel_brep} B-Rep faces ({n_sel_tris} triangles)")
        else:
            psim.TextUnformatted(f"{n_tris} triangles (no face map found)")

    # ---- Face selection via click ----
    psim.Separator()
    psim.TextUnformatted("=== Face Selection ===")
    psim.TextUnformatted("Click to select a B-Rep face.")
    psim.TextUnformatted("Shift+click to deselect.")

    io = psim.GetIO()
    if io.MouseClicked[0] and state.query_mesh is not None:
        if not io.WantCaptureMouse:
            pick_result = ps.pick(screen_coords=io.MousePos)

            if (pick_result.is_hit
                    and pick_result.structure_name == "query"
                    and pick_result.structure_data['element_type'] == "face"):

                f_hit = pick_result.structure_data['index']
                m = state.query_mesh

                if m['face_map'] is not None:
                    brep_id = int(m['face_map'][f_hit])
                    shift_held = io.KeyShift

                    if shift_held:
                        state.selected_brep_faces.discard(brep_id)
                    else:
                        state.selected_brep_faces.add(brep_id)

                    update_selection_viz(state)

                    if state.auto_search and state.selected_brep_faces:
                        run_search(state)
                        if state.results:
                            show_result(state, 0)

    if psim.Button("Clear Selection"):
        state.selected_brep_faces = set()
        if state.query_mesh is not None:
            state.query_mesh['ps_mesh'].add_color_quantity(
                'pca_colors', state.query_mesh['pca_colors'],
                defined_on='faces', enabled=True)

    psim.SameLine()
    _, state.auto_search = psim.Checkbox("Auto-search", state.auto_search)

    # Show selected B-Rep face IDs
    if state.selected_brep_faces:
        ids_str = ", ".join(str(x) for x in sorted(state.selected_brep_faces))
        psim.TextUnformatted(f"B-Rep IDs: {ids_str}")

    # ---- Search ----
    psim.Separator()
    psim.TextUnformatted("=== Search ===")

    if psim.Button("Search All Parts") and state.selected_brep_faces:
        run_search(state)
        if state.results:
            show_result(state, 0)

    changed, state.feature_range = psim.SliderFloat("Distance range",
                                                     state.feature_range,
                                                     v_min=0.01, v_max=0.5)
    if changed and state.result_mesh is not None and state.result_view_idx >= 0:
        result = state.results[state.result_view_idx]
        state.result_mesh['ps_mesh'].add_scalar_quantity(
            "similarity", result['scores'], cmap='turbo',
            vminmax=(1.0 - state.feature_range, 1.0),
            defined_on='faces', enabled=True)

    # ---- Results list ----
    if state.results:
        psim.Separator()
        psim.TextUnformatted(f"=== Results ({len(state.results)} parts) ===")

        max_show = min(10, len(state.results))
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
    parser = argparse.ArgumentParser(description="Query features across a part database (B-Rep face selection)")
    parser.add_argument('--data_root', required=True,
                        help='Path to PartField feature directory')
    parser.add_argument('--obj_dir', required=True,
                        help='Path to OBJ directory containing face_map.npy files')
    args = parser.parse_args()

    db = PartDatabase(args.data_root, args.obj_dir)
    state = AppState(db)

    ps.init()
    ps.set_ground_plane_mode("none")

    load_query_part(state)

    ps.set_user_callback(lambda: ps_callback(state))
    ps.show()


if __name__ == "__main__":
    main()
