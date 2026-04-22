import numpy as np
import torch
import polyscope as ps
import polyscope.imgui as psim
import potpourri3d as pp3d 
import trimesh
import igl
from dataclasses import dataclass
from simple_parsing import ArgumentParser
from arrgh import arrgh

### For clustering
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from scipy.sparse import coo_matrix, csr_matrix
from scipy.spatial import KDTree
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors
import networkx as nx

from scipy.optimize import linear_sum_assignment

import os, sys
sys.path.append("..")
from partfield.utils import *

@dataclass
class Options:
    
    """ Basic Options """
    filename: str

    """System Options"""
    device: str = "cuda"  #  Device
    debug: bool = False  #  enable debug checks
    extras: bool = False # include extra output for viz/debugging

    """ State """
    mode: str = 'pca'
    m: dict = None          # mesh

    # pca mode

    # feature explore mode
    i_feature: int = 0

    i_cluster: int = 1 

    i_eps: int = 0.6 

    ### For mixing in clustering
    weight_dist = 1.0
    weight_feat = 1.0
    
    ### For clustering visualization
    feature_range: float = 0.1
    continuous_explore: bool = False

    viz_mode: str = "faces"

    output_fol: str = "results_single"

    ### For adj_matrix
    adj_mode: str = "Vanilla"
    add_knn_edges: bool = False

    ### counter for screenshot
    counter: int = 0

modes_list = ['pca', 'feature_viz', 'cluster_agglo', 'cluster_kmeans']
adj_mode_list = ["Vanilla", "Face_MST", "CC_MST"]

#### For clustering
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        
        if rootX != rootY:
            if self.rank[rootX] > self.rank[rootY]:
                self.parent[rootY] = rootX
            elif self.rank[rootX] < self.rank[rootY]:
                self.parent[rootX] = rootY
            else:
                self.parent[rootY] = rootX
                self.rank[rootX] += 1

#####################################
## Face adjacency computation options
#####################################
def construct_face_adjacency_matrix_ccmst(face_list, vertices, k=10, with_knn=True):
    """
    Given a list of faces (each face is a 3-tuple of vertex indices),
    construct a face-based adjacency matrix of shape (num_faces, num_faces).

    Two faces are adjacent if they share an edge (the "mesh adjacency").
    If multiple connected components remain, we:
      1) Compute the centroid of each connected component as the mean of all face centroids.
      2) Use a KNN graph (k=10) based on centroid distances on each connected component.
      3) Compute MST of that KNN graph.
      4) Add MST edges that connect different components as "dummy" edges
         in the face adjacency matrix, ensuring one connected component. The selected face for 
         each connected component is the face closest to the component centroid.

    Parameters
    ----------
    face_list : list of tuples
        List of faces, each face is a tuple (v0, v1, v2) of vertex indices.
    vertices : np.ndarray of shape (num_vertices, 3)
        Array of vertex coordinates.
    k : int, optional
        Number of neighbors to use in centroid KNN. Default is 10.

    Returns
    -------
    face_adjacency : scipy.sparse.csr_matrix
        A CSR sparse matrix of shape (num_faces, num_faces),
        containing 1s for adjacent faces (shared-edge adjacency)
        plus dummy edges ensuring a single connected component.
    """
    num_faces = len(face_list)
    if num_faces == 0:
        # Return an empty matrix if no faces
        return csr_matrix((0, 0))

    #--------------------------------------------------------------------------
    # 1) Build adjacency based on shared edges.
    #    (Same logic as the original code, plus import statements.)
    #--------------------------------------------------------------------------
    edge_to_faces = defaultdict(list)
    uf = UnionFind(num_faces)
    for f_idx, (v0, v1, v2) in enumerate(face_list):
        # Sort each edge’s endpoints so (i, j) == (j, i)
        edges = [
            tuple(sorted((v0, v1))),
            tuple(sorted((v1, v2))),
            tuple(sorted((v2, v0)))
        ]
        for e in edges:
            edge_to_faces[e].append(f_idx)

    row = []
    col = []
    for edge, face_indices in edge_to_faces.items():
        unique_faces = list(set(face_indices))
        if len(unique_faces) > 1:
            # For every pair of distinct faces that share this edge,
            # mark them as mutually adjacent
            for i in range(len(unique_faces)):
                for j in range(i + 1, len(unique_faces)):
                    fi = unique_faces[i]
                    fj = unique_faces[j]
                    row.append(fi)
                    col.append(fj)
                    row.append(fj)
                    col.append(fi)
                    uf.union(fi, fj)

    data = np.ones(len(row), dtype=np.int8)
    face_adjacency = coo_matrix(
        (data, (row, col)), shape=(num_faces, num_faces)
    ).tocsr()

    #--------------------------------------------------------------------------
    # 2) Check if the graph from shared edges is already connected.
    #--------------------------------------------------------------------------
    n_components = 0
    for i in range(num_faces):
        if uf.find(i) == i:
            n_components += 1
    print("n_components", n_components)

    if n_components == 1:
        # Already a single connected component, no need for dummy edges
        return face_adjacency

    #--------------------------------------------------------------------------
    # 3) Compute centroids of each face for building a KNN graph.
    #--------------------------------------------------------------------------
    face_centroids = []
    for (v0, v1, v2) in face_list:
        centroid = (vertices[v0] + vertices[v1] + vertices[v2]) / 3.0
        face_centroids.append(centroid)
    face_centroids = np.array(face_centroids)

    #--------------------------------------------------------------------------
    # 4b) Build a KNN graph on connected components
    #--------------------------------------------------------------------------
    # Group faces by their root representative in the Union-Find structure
    component_dict = {}
    for face_idx in range(num_faces):
        root = uf.find(face_idx)
        if root not in component_dict:
            component_dict[root] = set()
        component_dict[root].add(face_idx)

    connected_components = list(component_dict.values())
    
    print("Using connected component MST.")
    component_centroid_face_idx = []
    connected_component_centroids = []
    knn = NearestNeighbors(n_neighbors=1, algorithm='auto')
    for component in connected_components:
        curr_component_faces = list(component)
        curr_component_face_centroids = face_centroids[curr_component_faces]
        component_centroid = np.mean(curr_component_face_centroids, axis=0)

        ### Assign a face closest to the centroid
        face_idx = curr_component_faces[np.argmin(np.linalg.norm(curr_component_face_centroids-component_centroid, axis=-1))]

        connected_component_centroids.append(component_centroid)
        component_centroid_face_idx.append(face_idx)

    component_centroid_face_idx = np.array(component_centroid_face_idx)
    connected_component_centroids = np.array(connected_component_centroids)

    if n_components < k:
        knn = NearestNeighbors(n_neighbors=n_components, algorithm='auto')
    else:
        knn = NearestNeighbors(n_neighbors=k, algorithm='auto')
    knn.fit(connected_component_centroids)
    distances, indices = knn.kneighbors(connected_component_centroids)    

    #--------------------------------------------------------------------------
    # 5) Build a weighted graph in NetworkX using centroid-distances as edges
    #--------------------------------------------------------------------------
    G = nx.Graph()
    # Add each face as a node in the graph
    G.add_nodes_from(range(num_faces))

    # For each face i, add edges (i -> j) for each neighbor j in the KNN
    for idx1 in range(n_components):
        i = component_centroid_face_idx[idx1]
        for idx2, dist in zip(indices[idx1], distances[idx1]):
            j = component_centroid_face_idx[idx2]
            if i == j:
                continue  # skip self-loop
            # Add an undirected edge with 'weight' = distance
            # NetworkX handles parallel edges gracefully via last add_edge,
            # but it typically overwrites the weight if (i, j) already exists.
            G.add_edge(i, j, weight=dist)

    #--------------------------------------------------------------------------
    # 6) Compute MST on that KNN graph
    #--------------------------------------------------------------------------
    mst = nx.minimum_spanning_tree(G, weight='weight')
    # Sort MST edges by ascending weight, so we add the shortest edges first
    mst_edges_sorted = sorted(
        mst.edges(data=True), key=lambda e: e[2]['weight']
    )
    print("mst edges sorted", len(mst_edges_sorted))
    #--------------------------------------------------------------------------
    # 7) Use a union-find structure to add MST edges only if they
    #    connect two currently disconnected components of the adjacency matrix
    #--------------------------------------------------------------------------

    # Convert face_adjacency to LIL format for efficient edge addition
    adjacency_lil = face_adjacency.tolil()

    # Now, step through MST edges in ascending order
    for (u, v, attr) in mst_edges_sorted:
        if uf.find(u) != uf.find(v):
            # These belong to different components, so unify them
            uf.union(u, v)
            # And add a "dummy" edge to our adjacency matrix
            adjacency_lil[u, v] = 1
            adjacency_lil[v, u] = 1

    # Convert back to CSR format and return
    face_adjacency = adjacency_lil.tocsr()

    if with_knn:
        print("Adding KNN edges.")
        ### Add KNN edges graph too
        dummy_row = []
        dummy_col = []
        for idx1 in range(n_components):
            i = component_centroid_face_idx[idx1]
            for idx2 in indices[idx1]:
                j = component_centroid_face_idx[idx2]     
                dummy_row.extend([i, j])
                dummy_col.extend([j, i]) ### duplicates are handled by coo

        dummy_data = np.ones(len(dummy_row), dtype=np.int16)
        dummy_mat = coo_matrix(
            (dummy_data, (dummy_row, dummy_col)),
            shape=(num_faces, num_faces)
        ).tocsr()
        face_adjacency = face_adjacency + dummy_mat
        ###########################

    return face_adjacency
#########################

def construct_face_adjacency_matrix_facemst(face_list, vertices, k=10, with_knn=True):
    """
    Given a list of faces (each face is a 3-tuple of vertex indices),
    construct a face-based adjacency matrix of shape (num_faces, num_faces).

    Two faces are adjacent if they share an edge (the "mesh adjacency").
    If multiple connected components remain, we:
      1) Compute the centroid of each face.
      2) Use a KNN graph (k=10) based on centroid distances.
      3) Compute MST of that KNN graph.
      4) Add MST edges that connect different components as "dummy" edges
         in the face adjacency matrix, ensuring one connected component.

    Parameters
    ----------
    face_list : list of tuples
        List of faces, each face is a tuple (v0, v1, v2) of vertex indices.
    vertices : np.ndarray of shape (num_vertices, 3)
        Array of vertex coordinates.
    k : int, optional
        Number of neighbors to use in centroid KNN. Default is 10.

    Returns
    -------
    face_adjacency : scipy.sparse.csr_matrix
        A CSR sparse matrix of shape (num_faces, num_faces),
        containing 1s for adjacent faces (shared-edge adjacency)
        plus dummy edges ensuring a single connected component.
    """
    num_faces = len(face_list)
    if num_faces == 0:
        # Return an empty matrix if no faces
        return csr_matrix((0, 0))

    #--------------------------------------------------------------------------
    # 1) Build adjacency based on shared edges.
    #    (Same logic as the original code, plus import statements.)
    #--------------------------------------------------------------------------
    edge_to_faces = defaultdict(list)
    uf = UnionFind(num_faces)
    for f_idx, (v0, v1, v2) in enumerate(face_list):
        # Sort each edge’s endpoints so (i, j) == (j, i)
        edges = [
            tuple(sorted((v0, v1))),
            tuple(sorted((v1, v2))),
            tuple(sorted((v2, v0)))
        ]
        for e in edges:
            edge_to_faces[e].append(f_idx)

    row = []
    col = []
    for edge, face_indices in edge_to_faces.items():
        unique_faces = list(set(face_indices))
        if len(unique_faces) > 1:
            # For every pair of distinct faces that share this edge,
            # mark them as mutually adjacent
            for i in range(len(unique_faces)):
                for j in range(i + 1, len(unique_faces)):
                    fi = unique_faces[i]
                    fj = unique_faces[j]
                    row.append(fi)
                    col.append(fj)
                    row.append(fj)
                    col.append(fi)
                    uf.union(fi, fj)

    data = np.ones(len(row), dtype=np.int8)
    face_adjacency = coo_matrix(
        (data, (row, col)), shape=(num_faces, num_faces)
    ).tocsr()

    #--------------------------------------------------------------------------
    # 2) Check if the graph from shared edges is already connected.
    #--------------------------------------------------------------------------
    n_components = 0
    for i in range(num_faces):
        if uf.find(i) == i:
            n_components += 1
    print("n_components", n_components)

    if n_components == 1:
        # Already a single connected component, no need for dummy edges
        return face_adjacency
    #--------------------------------------------------------------------------
    # 3) Compute centroids of each face for building a KNN graph.
    #--------------------------------------------------------------------------
    face_centroids = []
    for (v0, v1, v2) in face_list:
        centroid = (vertices[v0] + vertices[v1] + vertices[v2]) / 3.0
        face_centroids.append(centroid)
    face_centroids = np.array(face_centroids)

    #--------------------------------------------------------------------------
    # 4) Build a KNN graph (k=10) over face centroids using scikit‐learn
    #--------------------------------------------------------------------------
    knn = NearestNeighbors(n_neighbors=k, algorithm='auto')
    knn.fit(face_centroids)
    distances, indices = knn.kneighbors(face_centroids)
    # 'distances[i]' are the distances from face i to each of its 'k' neighbors
    # 'indices[i]' are the face indices of those neighbors

    #--------------------------------------------------------------------------
    # 5) Build a weighted graph in NetworkX using centroid-distances as edges
    #--------------------------------------------------------------------------
    G = nx.Graph()
    # Add each face as a node in the graph
    G.add_nodes_from(range(num_faces))

    # For each face i, add edges (i -> j) for each neighbor j in the KNN
    for i in range(num_faces):
        for j, dist in zip(indices[i], distances[i]):
            if i == j:
                continue  # skip self-loop
            # Add an undirected edge with 'weight' = distance
            # NetworkX handles parallel edges gracefully via last add_edge,
            # but it typically overwrites the weight if (i, j) already exists.
            G.add_edge(i, j, weight=dist)

    #--------------------------------------------------------------------------
    # 6) Compute MST on that KNN graph
    #--------------------------------------------------------------------------
    mst = nx.minimum_spanning_tree(G, weight='weight')
    # Sort MST edges by ascending weight, so we add the shortest edges first
    mst_edges_sorted = sorted(
        mst.edges(data=True), key=lambda e: e[2]['weight']
    )
    print("mst edges sorted", len(mst_edges_sorted))
    #--------------------------------------------------------------------------
    # 7) Use a union-find structure to add MST edges only if they
    #    connect two currently disconnected components of the adjacency matrix
    #--------------------------------------------------------------------------

    # Convert face_adjacency to LIL format for efficient edge addition
    adjacency_lil = face_adjacency.tolil()

    # Now, step through MST edges in ascending order
    for (u, v, attr) in mst_edges_sorted:
        if uf.find(u) != uf.find(v):
            # These belong to different components, so unify them
            uf.union(u, v)
            # And add a "dummy" edge to our adjacency matrix
            adjacency_lil[u, v] = 1
            adjacency_lil[v, u] = 1

    # Convert back to CSR format and return
    face_adjacency = adjacency_lil.tocsr()

    if with_knn:
        print("Adding KNN edges.")
        ### Add KNN edges graph too
        dummy_row = []
        dummy_col = []
        for i in range(num_faces):
            for j in indices[i]:        
                dummy_row.extend([i, j])
                dummy_col.extend([j, i]) ### duplicates are handled by coo

        dummy_data = np.ones(len(dummy_row), dtype=np.int16)
        dummy_mat = coo_matrix(
            (dummy_data, (dummy_row, dummy_col)),
            shape=(num_faces, num_faces)
        ).tocsr()
        face_adjacency = face_adjacency + dummy_mat
        ###########################

    return face_adjacency

def construct_face_adjacency_matrix_naive(face_list):
    """
    Given a list of faces (each face is a 3-tuple of vertex indices),
    construct a face-based adjacency matrix of shape (num_faces, num_faces).
    Two faces are adjacent if they share an edge.

    If multiple connected components exist, dummy edges are added to 
    turn them into a single connected component. Edges are added naively by
    randomly selecting a face and connecting consecutive components -- (comp_i, comp_i+1) ...

    Parameters
    ----------
    face_list : list of tuples
        List of faces, each face is a tuple (v0, v1, v2) of vertex indices.

    Returns
    -------
    face_adjacency : scipy.sparse.csr_matrix
        A CSR sparse matrix of shape (num_faces, num_faces), 
        containing 1s for adjacent faces and 0s otherwise. 
        Additional edges are added if the faces are in multiple components.
    """

    num_faces = len(face_list)
    if num_faces == 0:
        # Return an empty matrix if no faces
        return csr_matrix((0, 0))

    # Step 1: Map each undirected edge -> list of face indices that contain that edge
    edge_to_faces = defaultdict(list)

    # Populate the edge_to_faces dictionary
    for f_idx, (v0, v1, v2) in enumerate(face_list):
        # For an edge, we always store its endpoints in sorted order
        # to avoid duplication (e.g. edge (2,5) is the same as (5,2)).
        edges = [
            tuple(sorted((v0, v1))),
            tuple(sorted((v1, v2))),
            tuple(sorted((v2, v0)))
        ]
        for e in edges:
            edge_to_faces[e].append(f_idx)

    # Step 2: Build the adjacency (row, col) lists among faces
    row = []
    col = []
    for e, faces_sharing_e in edge_to_faces.items():
        # If an edge is shared by multiple faces, make each pair of those faces adjacent
        f_indices = list(set(faces_sharing_e))  # unique face indices for this edge
        if len(f_indices) > 1:
            # For each pair of faces, mark them as adjacent
            for i in range(len(f_indices)):
                for j in range(i + 1, len(f_indices)):
                    f_i = f_indices[i]
                    f_j = f_indices[j]
                    row.append(f_i)
                    col.append(f_j)
                    row.append(f_j)
                    col.append(f_i)

    # Create a COO matrix, then convert it to CSR
    data = np.ones(len(row), dtype=np.int8)
    face_adjacency = coo_matrix(
        (data, (row, col)),
        shape=(num_faces, num_faces)
    ).tocsr()

    # Step 3: Ensure single connected component
    # Use connected_components to see how many components exist
    n_components, labels = connected_components(face_adjacency, directed=False)

    if n_components > 1:
        # We have multiple components; let's "connect" them via dummy edges
        # The simplest approach is to pick one face from each component
        # and connect them sequentially to enforce a single component.
        component_representatives = []

        for comp_id in range(n_components):
            # indices of faces in this component
            faces_in_comp = np.where(labels == comp_id)[0]
            if len(faces_in_comp) > 0:
                # take the first face in this component as a representative
                component_representatives.append(faces_in_comp[0])

        # Now, add edges between consecutive representatives
        dummy_row = []
        dummy_col = []
        for i in range(len(component_representatives) - 1):
            f_i = component_representatives[i]
            f_j = component_representatives[i + 1]
            dummy_row.extend([f_i, f_j])
            dummy_col.extend([f_j, f_i])

        if dummy_row:
            dummy_data = np.ones(len(dummy_row), dtype=np.int8)
            dummy_mat = coo_matrix(
                (dummy_data, (dummy_row, dummy_col)),
                shape=(num_faces, num_faces)
            ).tocsr()
            face_adjacency = face_adjacency + dummy_mat

    return face_adjacency
#####################################

def load_features(feature_filename, mesh_filename, viz_mode):
    
    print("Reading features:")
    print(f"  Feature filename: {feature_filename}")
    print(f"  Mesh filename: {mesh_filename}")

    # load features
    feat = np.load(feature_filename, allow_pickle=True)
    feat = feat.astype(np.float32)

    # load mesh things
    tm =  load_mesh_util(mesh_filename)

    V = np.array(tm.vertices, dtype=np.float32)
    F = np.array(tm.faces)

    if viz_mode ==  "faces":
        pca_colors = np.array(tm.visual.face_colors, dtype=np.float32)
        pca_colors = pca_colors[:,:3] / 255.
        
    else:
        pca_colors = np.array(tm.visual.vertex_colors, dtype=np.float32)
        pca_colors = pca_colors[:,:3] / 255.

    arrgh(V, F, pca_colors, feat)

    print(F)
    print(V[F[1][0]])
    print(V[F[1][1]])
    print(V[F[1][2]])

    return {
        'V' : V, 
        'F' : F, 
        'pca_colors' : pca_colors, 
        'feat_np' : feat,
        'feat_pt' : torch.tensor(feat, device='cuda'),
        'trimesh' : tm,
        'label' : None,
        'num_cluster' : 1,
        'scalar' : None
    }

def prep_feature_mesh(m, name='mesh'):
    ps_mesh = ps.register_surface_mesh(name, m['V'], m['F'])
    ps_mesh.set_selection_mode('faces_only')
    m['ps_mesh'] = ps_mesh

def viz_pca_colors(m):
    m['ps_mesh'].add_color_quantity('pca colors', m['pca_colors'], enabled=True, defined_on=m["viz_mode"])

def viz_feature(m, ind):
    m['ps_mesh'].add_scalar_quantity('pca colors', m['feat_np'][:,ind], cmap='turbo', enabled=True, defined_on=m["viz_mode"])

def feature_distance_np(feats, query_feat):
    # normalize
    feats = feats / np.linalg.norm(feats,axis=1)[:,None]
    query_feat = query_feat / np.linalg.norm(query_feat)
    # cosine distance
    cos_sim = np.dot(feats, query_feat)
    cos_dist = (1 - cos_sim) / 2.
    return cos_dist

def feature_distance_pt(feats, query_feat):
    return (1. - torch.nn.functional.cosine_similarity(feats, query_feat[None,:], dim=-1)) / 2.


# Distinct cluster colors (categorical, not a gradient)
CLUSTER_COLORS = np.array([
    [0.90, 0.20, 0.20],  # red
    [0.20, 0.60, 0.90],  # blue
    [0.20, 0.80, 0.30],  # green
    [0.95, 0.60, 0.10],  # orange
    [0.70, 0.30, 0.85],  # purple
    [0.95, 0.85, 0.10],  # yellow
    [0.10, 0.80, 0.80],  # cyan
    [0.90, 0.40, 0.70],  # pink
    [0.55, 0.35, 0.15],  # brown
    [0.40, 0.75, 0.55],  # teal
    [0.85, 0.50, 0.25],  # burnt orange
    [0.50, 0.50, 0.90],  # periwinkle
    [0.75, 0.85, 0.20],  # lime
    [0.90, 0.30, 0.50],  # rose
    [0.30, 0.50, 0.70],  # steel blue
    [0.80, 0.65, 0.40],  # tan
    [0.55, 0.80, 0.80],  # light teal
    [0.75, 0.40, 0.55],  # mauve
    [0.45, 0.70, 0.30],  # olive green
    [0.85, 0.75, 0.60],  # wheat
    [0.60, 0.40, 0.70],  # medium purple
    [0.30, 0.70, 0.60],  # sea green
    [0.90, 0.55, 0.55],  # salmon
    [0.40, 0.40, 0.60],  # slate
    [0.70, 0.70, 0.20],  # olive
    [0.20, 0.50, 0.50],  # dark teal
    [0.80, 0.30, 0.30],  # dark red
    [0.30, 0.30, 0.80],  # dark blue
    [0.50, 0.80, 0.50],  # light green
    [0.80, 0.50, 0.80],  # light purple
], dtype=np.float32)

DIMMED_COLOR = np.array([0.92, 0.92, 0.92], dtype=np.float32)


def _build_brep_adjacency(vertices, faces, face_map):
    """Build B-Rep face adjacency from shared boundary vertices.

    B-Rep faces are tessellated independently so their triangles don't share
    mesh edges.  Instead, adjacent B-Rep faces have coincident vertices along
    their shared boundary.  We detect these by spatial hashing.
    """
    # For each B-Rep face, collect its vertex positions
    brep_face_verts = defaultdict(set)  # brep_fid -> set of vertex indices
    for tri_idx, brep_fid in enumerate(face_map):
        for vid in faces[tri_idx]:
            brep_face_verts[int(brep_fid)].add(int(vid))

    # Spatial hash: round coords to detect coincident vertices across faces
    # Resolution: 1e-4 (0.1mm for typical CAD tolerances)
    RESOLUTION = 1e4
    pos_to_faces = defaultdict(set)
    for brep_fid, vids in brep_face_verts.items():
        for vid in vids:
            v = vertices[vid]
            key = (round(v[0] * RESOLUTION),
                   round(v[1] * RESOLUTION),
                   round(v[2] * RESOLUTION))
            pos_to_faces[key].add(brep_fid)

    # Two B-Rep faces are adjacent if they share any coincident vertex position
    adjacency = defaultdict(set)
    for key, fids in pos_to_faces.items():
        fids = list(fids)
        for i in range(len(fids)):
            for j in range(i + 1, len(fids)):
                adjacency[fids[i]].add(fids[j])
                adjacency[fids[j]].add(fids[i])

    return adjacency


def _find_brep_connected_components(brep_fids, adjacency):
    """Find connected components among a set of B-Rep face IDs."""
    fid_set = set(brep_fids)
    visited = set()
    components = []

    for start in brep_fids:
        if start in visited:
            continue
        comp = []
        queue = [start]
        while queue:
            node = queue.pop()
            if node in visited:
                continue
            visited.add(node)
            comp.append(node)
            for nb in adjacency.get(node, set()):
                if nb in fid_set and nb not in visited:
                    queue.append(nb)
        components.append(sorted(comp))

    components.sort(key=len, reverse=True)
    return components


def _compute_sub_cohesion(tri_indices, point_feat_normed, face_map):
    """Compute cohesion and B-Rep faces for a set of triangles."""
    feats = point_feat_normed[tri_indices]
    centroid = feats.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm
    cohesion = float((feats @ centroid).mean())

    n_brep = 0
    brep_fids = []
    if face_map is not None:
        brep_fids = np.unique(face_map[tri_indices]).tolist()
        n_brep = len(brep_fids)

    return cohesion, n_brep, brep_fids


def _on_cluster_computed(m, point_feat_normed):
    """Build hierarchical cluster groups: clusters -> sub-features (connected B-Rep regions)."""
    labels = m['cluster_labels']
    m['point_feat_normed'] = point_feat_normed
    face_map = m.get('face_map', None)
    n_clusters = int(labels.max()) + 1

    # Build B-Rep face adjacency (shared boundary vertices)
    brep_adj = None
    if face_map is not None:
        brep_adj = _build_brep_adjacency(m['V'], m['F'], face_map)

    groups = []
    for cid in range(n_clusters):
        tri_indices = np.where(labels == cid)[0]
        if len(tri_indices) == 0:
            continue

        cluster_coh, cluster_n_brep, _ = _compute_sub_cohesion(
            tri_indices, point_feat_normed, face_map)

        subs = []

        if face_map is not None and brep_adj is not None:
            # Split by B-Rep face connectivity
            brep_fids = np.unique(face_map[tri_indices]).tolist()
            components = _find_brep_connected_components(brep_fids, brep_adj)

            for comp_fids in components:
                # Get all triangles belonging to these B-Rep faces
                comp_tris = np.where(np.isin(face_map, comp_fids) & (labels == cid))[0]
                coh, n_brep, bf = _compute_sub_cohesion(
                    comp_tris, point_feat_normed, face_map)
                subs.append({
                    'tri_indices': comp_tris,
                    'n_tris': len(comp_tris),
                    'n_brep': n_brep,
                    'brep_fids': bf,
                    'cohesion': coh,
                    'visible': True,
                })
        else:
            # No face_map: single sub = the whole cluster
            coh, n_brep, bf = _compute_sub_cohesion(
                tri_indices, point_feat_normed, face_map)
            subs.append({
                'tri_indices': tri_indices,
                'n_tris': len(tri_indices),
                'n_brep': n_brep,
                'brep_fids': bf,
                'cohesion': coh,
                'visible': True,
            })

        subs.sort(key=lambda s: s['cohesion'], reverse=True)

        groups.append({
            'cluster_id': cid,
            'n_tris': len(tri_indices),
            'n_brep': cluster_n_brep,
            'cohesion': cluster_coh,
            'subs': subs,
            'visible': True,
        })

    groups.sort(key=lambda g: g['cohesion'], reverse=True)

    m['cluster_groups'] = groups
    n_subs = sum(len(g['subs']) for g in groups)
    print(f"{n_clusters} clusters -> {n_subs} sub-features (B-Rep connectivity)")

    _update_cluster_colors(m)


# Fixed per-type colors so Hole is always red, Shaft always blue, etc.
UNIT_FEATURE_COLORS = {
    "Hole":               np.array([0.90, 0.20, 0.20], dtype=np.float32),
    "Shaft":              np.array([0.20, 0.60, 0.90], dtype=np.float32),
    "Fillet":             np.array([0.20, 0.80, 0.30], dtype=np.float32),
    "Chamfer":            np.array([0.10, 0.80, 0.80], dtype=np.float32),
    "BossExtrudeFeature": np.array([0.95, 0.60, 0.10], dtype=np.float32),
    "CutExtrudeFeature":  np.array([0.70, 0.30, 0.85], dtype=np.float32),
    "PrismaticMilling":   np.array([0.95, 0.85, 0.10], dtype=np.float32),
    "Thread":             np.array([0.55, 0.35, 0.15], dtype=np.float32),
    "Bases":              np.array([0.75, 0.75, 0.20], dtype=np.float32),
}
UNIT_FEATURE_FALLBACK = np.array([
    [0.40, 0.75, 0.55], [0.85, 0.50, 0.25], [0.50, 0.50, 0.90],
    [0.90, 0.40, 0.70], [0.30, 0.50, 0.70], [0.80, 0.65, 0.40],
    [0.55, 0.80, 0.80], [0.75, 0.40, 0.55], [0.45, 0.70, 0.30],
], dtype=np.float32)
UNRECOGNIZED_COLOR = np.array([1.00, 0.05, 0.25], dtype=np.float32)


def _unit_feature_color(name: str) -> np.ndarray:
    if name in UNIT_FEATURE_COLORS:
        return UNIT_FEATURE_COLORS[name]
    idx = abs(hash(name)) % len(UNIT_FEATURE_FALLBACK)
    return UNIT_FEATURE_FALLBACK[idx]


def load_hrep_features(hrep_path: str):
    """Load HRep JSON, return list of {name, face_ids (0-indexed set)}."""
    import json
    with open(hrep_path) as f:
        data = json.load(f)
    features = []
    for feat in data.get("features", []):
        name = feat.get("feature_name")
        if not name:
            continue
        fids = {fid - 1 for fid in feat.get("face_groups", []) if fid}
        if not fids:
            continue
        features.append({"name": name, "face_ids": fids})
    return features


def _break_sub(sub, hrep_features, face_map):
    """Split a sub-cluster's B-Rep faces into unit features from HRep.

    Writes sub['broken'] = True, sub['unit_features'] = [...],
    sub['unrecognized'] = {...} | None. Triangle masks are restricted to
    tris belonging to this sub so other subs are unaffected.
    """
    sub_tris = sub['tri_indices']
    sub_fids = set(sub.get('brep_fids', []))
    if not sub_fids or face_map is None:
        sub['broken'] = True
        sub['unit_features'] = []
        sub['unrecognized'] = None
        return

    sub_tri_face_ids = face_map[sub_tris]

    covered = set()
    unit_features = []
    for feat in hrep_features:
        overlap = feat['face_ids'] & sub_fids
        if not overlap:
            continue
        mask = np.isin(sub_tri_face_ids, list(overlap))
        unit_features.append({
            'name': feat['name'],
            'color': _unit_feature_color(feat['name']),
            'brep_fids': sorted(overlap),
            'tri_indices': sub_tris[mask],
            'visible': True,
        })
        covered.update(overlap)

    # Sort for stable display: by type name, then face count desc
    unit_features.sort(key=lambda u: (u['name'], -len(u['brep_fids'])))

    unrecognized = None
    unrec_fids = sub_fids - covered
    if unrec_fids:
        mask = np.isin(sub_tri_face_ids, list(unrec_fids))
        unrecognized = {
            'brep_fids': sorted(unrec_fids),
            'tri_indices': sub_tris[mask],
            'visible': True,
        }

    sub['broken'] = True
    sub['unit_features'] = unit_features
    sub['unrecognized'] = unrecognized


def _update_cluster_colors(m):
    """Rebuild per-triangle colors based on group/sub visibility. One color per cluster."""
    groups = m['cluster_groups']
    n_tris = len(m['cluster_labels'])

    colors = np.tile(DIMMED_COLOR, (n_tris, 1))

    for gi, g in enumerate(groups):
        if not g['visible']:
            continue
        cluster_color = CLUSTER_COLORS[gi % len(CLUSTER_COLORS)]
        for sub in g['subs']:
            if not sub['visible']:
                continue
            if sub.get('broken'):
                for uf in sub.get('unit_features', []):
                    if uf['visible']:
                        colors[uf['tri_indices']] = uf['color']
                unr = sub.get('unrecognized')
                if unr and unr['visible']:
                    colors[unr['tri_indices']] = UNRECOGNIZED_COLOR
            else:
                colors[sub['tri_indices']] = cluster_color

    m['ps_mesh'].add_color_quantity("cluster_colors", colors,
                                    defined_on=m["viz_mode"], enabled=True)


def ps_callback(opts):
    m = opts.m

    changed, ind = psim.Combo("Mode", modes_list.index(opts.mode), modes_list)
    if changed:
        opts.mode = modes_list[ind]
        m['ps_mesh'].remove_all_quantities()

    if opts.mode == 'pca':
        psim.TextUnformatted("""3-dim PCA embeddeding of features is shown as rgb color""")
        viz_pca_colors(m)

    elif opts.mode == 'feature_viz':
        psim.TextUnformatted("""Use the slider to scrub through all features.\nCtrl-click to type a particular index.""")

        this_changed, opts.i_feature = psim.SliderInt("feature index", opts.i_feature, v_min=0, v_max=(m['feat_np'].shape[-1]-1))
        this_changed = this_changed or changed

        if this_changed:
            viz_feature(m, opts.i_feature)
    
    elif opts.mode == "cluster_agglo":
        psim.TextUnformatted("""Use the slider to toggle the number of desired clusters.""")
        cluster_changed, opts.i_cluster = psim.SliderInt("number of clusters", opts.i_cluster, v_min=1, v_max=30)

        ### To handle different face adjacency options
        mode_changed, ind = psim.Combo("Adj Matrix Def", adj_mode_list.index(opts.adj_mode), adj_mode_list)
        knn_changed, opts.add_knn_edges = psim.Checkbox("Add KNN edges", opts.add_knn_edges)
        
        if mode_changed:
            opts.adj_mode = adj_mode_list[ind]

        if psim.Button("Recompute"):

            ### Run clustering algorithm
            num_clusters = opts.i_cluster

            ### Mesh 1
            point_feat = m['feat_np']
            point_feat = point_feat / np.linalg.norm(point_feat, axis=-1, keepdims=True)
            
            ### Compute adjacency matrix ###
            if opts.adj_mode == "Vanilla":
                adj_matrix = construct_face_adjacency_matrix_naive(opts.m["F"])
            elif opts.adj_mode == "Face_MST":
                adj_matrix = construct_face_adjacency_matrix_facemst(opts.m["F"], opts.m["V"], with_knn=opts.add_knn_edges)
            elif opts.adj_mode == "CC_MST":
                adj_matrix = construct_face_adjacency_matrix_ccmst(opts.m["F"], opts.m["V"], with_knn=opts.add_knn_edges)            
            ################################

            ## Agglomerative clustering
            clustering = AgglomerativeClustering(connectivity= adj_matrix,
                                        n_clusters=num_clusters,
                                        ).fit(point_feat)

            m['cluster_labels'] = clustering.labels_
            m['cluster_method'] = f"agglomerative_n{num_clusters}_{opts.adj_mode}"
            _on_cluster_computed(m, point_feat)
            print("Recomputed.")


    elif opts.mode == "cluster_kmeans":
        psim.TextUnformatted("""Use the slider to toggle the number of desired clusters.""")

        cluster_changed, opts.i_cluster = psim.SliderInt("number of clusters", opts.i_cluster, v_min=1, v_max=30)

        if psim.Button("Recompute"):

            ### Run clustering algorithm
            num_clusters = opts.i_cluster

            ### Mesh 1
            point_feat = m['feat_np']
            point_feat = point_feat / np.linalg.norm(point_feat, axis=-1, keepdims=True)
            clustering = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto").fit(point_feat)

            m['cluster_labels'] = clustering.labels_
            m['cluster_method'] = f"kmeans_n{num_clusters}"
            _on_cluster_computed(m, point_feat)

    # --- Cluster browser & save (available after any clustering) ---
    if 'cluster_groups' in m:
        psim.Separator()

        groups = m['cluster_groups']
        n_groups = len(groups)
        n_subs = sum(len(g['subs']) for g in groups)

        psim.TextUnformatted(f"{n_groups} clusters -> {n_subs} features    {m.get('cluster_method', '?')}")

        # Show All / Show None / Break All / Restore All
        if psim.Button("Show All"):
            for g in groups:
                g['visible'] = True
                for s in g['subs']:
                    s['visible'] = True
            _update_cluster_colors(m)
        psim.SameLine()
        if psim.Button("Show None"):
            for g in groups:
                g['visible'] = False
                for s in g['subs']:
                    s['visible'] = False
            _update_cluster_colors(m)

        hrep_feats = m.get('hrep_features')
        face_map = m.get('face_map')
        if hrep_feats is not None and face_map is not None:
            psim.SameLine()
            if psim.Button("Break All"):
                for g in groups:
                    g['visible'] = True
                    for s in g['subs']:
                        s['visible'] = True
                        if not s.get('broken') and s['n_brep'] > 0:
                            _break_sub(s, hrep_feats, face_map)
                m['expand_all'] = True
                _update_cluster_colors(m)
            psim.SameLine()
            if psim.Button("Restore All"):
                for g in groups:
                    for s in g['subs']:
                        s['broken'] = False
                        s.pop('unit_features', None)
                        s.pop('unrecognized', None)
                _update_cluster_colors(m)

            any_broken = any(s.get('broken') for g in groups for s in g['subs'])
            if any_broken:
                psim.SameLine()
                if psim.Button("Show Unrecognized"):
                    m['expand_all'] = True
                    m['collapse_recognized'] = True
                    for g in groups:
                        has_unrec = False
                        for s in g['subs']:
                            if s.get('broken') and s.get('unrecognized'):
                                has_unrec = True
                                s['visible'] = True
                                for uf in s.get('unit_features', []):
                                    uf['visible'] = False
                                s['unrecognized']['visible'] = True
                            else:
                                s['visible'] = False
                        g['visible'] = has_unrec
                    _update_cluster_colors(m)

        psim.TextUnformatted("(ranked by cohesion, expand to see sub-features)")
        psim.Separator()

        # Hierarchical tree: groups -> sub-features
        for gi, g in enumerate(groups):
            n_sub = len(g['subs'])
            brep_str = f"  {g['n_brep']} faces" if g['n_brep'] > 0 else ""
            parts_str = f"  [{n_sub} parts]" if n_sub > 1 else ""

            # Group-level checkbox + tree node
            toggled, new_val = psim.Checkbox(
                f"##grp_{gi}", g['visible'])
            if toggled:
                g['visible'] = new_val
                for s in g['subs']:
                    s['visible'] = new_val
                _update_cluster_colors(m)

            psim.SameLine()
            node_label = (f"Cluster {gi}  ({g['n_tris']} tri{brep_str})  "
                          f"coh={g['cohesion']:.3f}{parts_str}")
            if m.get('collapse_recognized'):
                has_unrec = any(
                    s.get('broken') and s.get('unrecognized')
                    for s in g['subs'])
                psim.SetNextItemOpen(has_unrec)
            elif m.get('expand_all'):
                psim.SetNextItemOpen(True)
            node_open = psim.TreeNode(node_label)

            # Cohesion bar on same line as tree node header
            # (can't SameLine with TreeNode easily, so put it inside)

            if node_open:
                # Sub-features inside this cluster
                color = CLUSTER_COLORS[gi % len(CLUSTER_COLORS)]
                hrep_feats = m.get('hrep_features')
                face_map = m.get('face_map')
                can_break = hrep_feats is not None and face_map is not None

                for si, sub in enumerate(g['subs']):
                    psim.PushStyleColor(0, (color[0], color[1], color[2], 1.0))

                    sub_brep = f"  {sub['n_brep']} faces" if sub['n_brep'] > 0 else ""
                    sub_label = (f"  Part {si}  ({sub['n_tris']} tri{sub_brep})  "
                                 f"coh={sub['cohesion']:.3f}##sub_{gi}_{si}")

                    sub_toggled, sub_new = psim.Checkbox(sub_label, sub['visible'])
                    psim.PopStyleColor()

                    if sub_toggled:
                        sub['visible'] = sub_new
                        # Update group visibility: on if any sub is on
                        g['visible'] = any(s['visible'] for s in g['subs'])
                        _update_cluster_colors(m)

                    # Hover tooltip on the checkbox row: cluster's face IDs
                    if sub.get('brep_fids') and psim.IsItemHovered():
                        psim.BeginTooltip()
                        fids_str = str(sub['brep_fids'][:20])
                        if len(sub['brep_fids']) > 20:
                            fids_str = fids_str[:-1] + f", ...+{len(sub['brep_fids'])-20}]"
                        psim.TextUnformatted(f"B-Rep face IDs: {fids_str}")
                        psim.EndTooltip()

                    psim.SameLine()
                    psim.ProgressBar(sub['cohesion'], (60, 0))

                    # Break / Restore button (needs HRep JSON + face_map)
                    if can_break and sub['n_brep'] > 0:
                        psim.SameLine()
                        is_broken = sub.get('broken', False)
                        btn_label = (f"Restore##br_{gi}_{si}" if is_broken
                                     else f"Break##br_{gi}_{si}")
                        if psim.SmallButton(btn_label):
                            if is_broken:
                                sub['broken'] = False
                                sub.pop('unit_features', None)
                                sub.pop('unrecognized', None)
                            else:
                                _break_sub(sub, hrep_feats, face_map)
                            _update_cluster_colors(m)

                    # Broken mode: show unit features + unrecognized as children
                    if sub.get('broken'):
                        ufs = sub.get('unit_features', [])
                        for ui, uf in enumerate(ufs):
                            col = uf['color']
                            psim.PushStyleColor(0, (col[0], col[1], col[2], 1.0))
                            n_f = len(uf['brep_fids'])
                            uf_label = (f"      {uf['name']}  ({n_f} faces)"
                                        f"##uf_{gi}_{si}_{ui}")
                            uf_tog, uf_new = psim.Checkbox(uf_label, uf['visible'])
                            psim.PopStyleColor()
                            if uf_tog:
                                uf['visible'] = uf_new
                                _update_cluster_colors(m)
                            if psim.IsItemHovered():
                                psim.BeginTooltip()
                                fids_str = str(uf['brep_fids'][:20])
                                if n_f > 20:
                                    fids_str = fids_str[:-1] + f", ...+{n_f-20}]"
                                psim.TextUnformatted(f"B-Rep face IDs: {fids_str}")
                                psim.EndTooltip()

                        unr = sub.get('unrecognized')
                        if unr:
                            col = UNRECOGNIZED_COLOR
                            psim.PushStyleColor(0, (col[0], col[1], col[2], 1.0))
                            n_u = len(unr['brep_fids'])
                            unr_label = (f"      Unrecognized  ({n_u} faces)"
                                         f"##unr_{gi}_{si}")
                            ut, un = psim.Checkbox(unr_label, unr['visible'])
                            psim.PopStyleColor()
                            if ut:
                                unr['visible'] = un
                                _update_cluster_colors(m)
                            if psim.IsItemHovered():
                                psim.BeginTooltip()
                                fids_str = str(unr['brep_fids'][:20])
                                if n_u > 20:
                                    fids_str = fids_str[:-1] + f", ...+{n_u-20}]"
                                psim.TextUnformatted(f"B-Rep face IDs: {fids_str}")
                                psim.EndTooltip()
                        elif ufs:
                            psim.TextUnformatted("      (all faces recognized)")
                        else:
                            psim.TextUnformatted("      (no HRep matches)")

                psim.TreePop()

        m.pop('expand_all', None)
        m.pop('collapse_recognized', None)

        psim.Separator()
        if psim.Button("Save All Clusters"):
            save_clustering_results(opts, m, selected_only=False)
        psim.SameLine()
        if psim.Button("Save Selected Only"):
            save_clustering_results(opts, m, selected_only=True)


def save_clustering_results(opts, m, selected_only=False):
    """Save clustering as hierarchical JSON. If selected_only, save only visible clusters/subs."""
    import json

    groups = m['cluster_groups']
    method = m.get('cluster_method', 'unknown')

    clusters_out = []
    for gi, g in enumerate(groups):
        if selected_only and not g['visible']:
            continue

        subs_out = []
        for si, sub in enumerate(g['subs']):
            if selected_only and not sub['visible']:
                continue
            subs_out.append({
                "part_idx": si,
                "cohesion": round(sub['cohesion'], 4),
                "n_triangles": sub['n_tris'],
                "face_ids": [fid + 1 for fid in sub.get('brep_fids', [])],
                "n_faces": sub['n_brep'],
            })

        if not subs_out:
            continue

        clusters_out.append({
            "cluster_idx": gi,
            "cohesion": round(g['cohesion'], 4),
            "n_triangles": g['n_tris'],
            "n_faces": g['n_brep'],
            "sub_features": subs_out,
        })

    suffix = "_selected" if selected_only else ""
    result = {
        "indexing": 1,
        "part_id": opts.filename,
        "method": method,
        "selected_only": selected_only,
        "n_clusters": len(clusters_out),
        "n_sub_features": sum(len(c['sub_features']) for c in clusters_out),
        "clusters": clusters_out,
    }

    out_path = os.path.join(opts.output_fol, f"{opts.filename}_partfield_{method}{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    tag = "selected" if selected_only else "all"
    print(f"\nSaved ({tag}) to: {out_path}")
    print(f"  {len(clusters_out)} clusters, {result['n_sub_features']} sub-features")
    for c in clusters_out:
        n_parts = len(c['sub_features'])
        parts_str = f" ({n_parts} parts)" if n_parts > 1 else ""
        print(f"  Cluster {c['cluster_idx']}: coh={c['cohesion']:.3f}  {c['n_faces']} faces{parts_str}")


def main():
    ## Parse args
    # Uses simple_parsing library to automatically construct parser from the dataclass Options
    parser = ArgumentParser()
    parser.add_arguments(Options, dest="options")
    parser.add_argument('--data_root', default="../exp_results/partfield_features/trellis/", help='Path the model features are stored.')
    parser.add_argument('--face_map', default=None, help='Path to face_map.npy for B-Rep face ID mapping.')
    parser.add_argument('--save_dir', default=None, help='Directory to save clustering results (overrides output_fol).')
    parser.add_argument('--hrep_json', default=None, help='Path to <part_id>_hrep.json. Enables per-sub "Break into unit features".')
    args = parser.parse_args()
    opts: Options = args.options

    if args.save_dir:
        opts.output_fol = args.save_dir

    DATA_ROOT = args.data_root

    shape_1 = opts.filename

    if os.path.exists(os.path.join(DATA_ROOT, "part_feat_"+ shape_1 + "_0.npy")):
        feature_fname1 = os.path.join(DATA_ROOT, "part_feat_"+ shape_1 + "_0.npy")
        mesh_fname1 = os.path.join(DATA_ROOT, "feat_pca_"+ shape_1 + "_0.ply")
    else:
        feature_fname1 = os.path.join(DATA_ROOT, "part_feat_"+ shape_1 + "_0_batch.npy")
        mesh_fname1 = os.path.join(DATA_ROOT, "feat_pca_"+ shape_1 + "_0.ply")

    #### To save output ####
    os.makedirs(opts.output_fol, exist_ok=True)
    ########################

    # Load face_map if provided (maps triangle index -> B-Rep face ID)
    face_map = None
    if args.face_map and os.path.exists(args.face_map):
        face_map = np.load(args.face_map)
        print(f"Loaded face_map: {face_map.shape} ({len(np.unique(face_map))} B-Rep faces)")

    # Initialize
    ps.init()

    mesh_dict = load_features(feature_fname1, mesh_fname1, opts.viz_mode)
    prep_feature_mesh(mesh_dict)
    mesh_dict["viz_mode"] = opts.viz_mode
    if face_map is not None:
        mesh_dict["face_map"] = face_map

    # Load HRep features if provided: enables per-sub "Break into unit features"
    if args.hrep_json and os.path.exists(args.hrep_json):
        hrep_features = load_hrep_features(args.hrep_json)
        mesh_dict["hrep_features"] = hrep_features
        print(f"Loaded HRep: {len(hrep_features)} features from {args.hrep_json}")

    opts.m = mesh_dict

    # Start the interactive UI
    ps.set_user_callback(lambda : ps_callback(opts))
    ps.show()


if __name__ == "__main__":
    main()

