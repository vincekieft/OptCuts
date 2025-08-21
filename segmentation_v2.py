import argparse
import sys
import gco
import pyrender
from sklearn.mixture import GaussianMixture; sys.path.append('./sdf120/build')
import sdf120
import trimesh
import numpy as np
import tkinter as tk
from scipy.sparse import coo_matrix
from helpers import (
    calculate_face_directions,
    curvature_direction_lines, 
    inpaint_sdf,
    sdf_to_colors, 
    labels_to_colors,
    smooth_blend_sdf
)
from scipy.sparse.csgraph import connected_components

# Some pyrenderer error about removed infty
if not hasattr(np, "infty"):
    np.infty = np.inf

def log_remap(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    Remaps x to log with a controllable alpha knob.
    Alpha = 0, almost linear
    Alpha >, stronger lift of darks / compression of highs
    Common uses: make thin/thickness signals or images more visible in the low range; 
    dynamic-range compression before visualization or as a feature.
    """
    return np.log(x * alpha + 1.0) / np.log(alpha + 1.0)

def robust_norm01(x: np.ndarray, low_percentile=1.0, high_percentile=99.0) -> np.ndarray:
    """
    Safe normalizes x between zero and one cliping of 1 and 99 percentile highs and lows
    """
    low, high = np.percentile(x, [low_percentile, high_percentile])
    
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros_like(x)
    
    # Shift x so that low and high are clipped to one.
    ZERO_SAFETY = 1e-9 # Prevents dividing by zero
    return np.clip((x - low) / max(high - low, ZERO_SAFETY), 0.0, 1.0)

def calculate_sdf(
    mesh: trimesh.Trimesh,
    *,
    rays: int = 24
) -> np.ndarray:
    """
    Calculates SDF per face of the mesh.
    """
    
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    
    CONE_DEGREES = 120.0
    sdf  = sdf120.compute_sdf(vertices, faces, rays=rays, cone_deg=CONE_DEGREES, postprocess=True)
    sdf = inpaint_sdf(mesh, sdf)
    nsdf = robust_norm01(sdf)
    smooth_sdf = smooth_blend_sdf(mesh, nsdf)
    return smooth_sdf

def gaussian_mixture(nsdf: np.ndarray, k: int):
    length = nsdf.shape[0]
    
    if length <= 1:
        return np.zeros((0, 1))
    
    clamped_components = int(min(k, length)) # Dont ask more gaussians than there are faces
    
    reshaped_nsdf = nsdf.reshape(-1, 1).astype(np.float64)
    
    gm = GaussianMixture(
        n_components=clamped_components,
        max_iter=200,
        n_init=10,
        random_state=0, # make the result deterministc
    ).fit(reshaped_nsdf)
    
    single_gm = GaussianMixture(
        n_components=1,
        max_iter=200,
        n_init=10,
        random_state=0,
    ).fit(reshaped_nsdf)
    
    bicK, bic1 = gm.bic(reshaped_nsdf), single_gm.bic(reshaped_nsdf)
    bic_difference = bic1 - bicK
    
    probabilities = np.clip(gm.predict_proba(reshaped_nsdf), 1e-12, 1.0) # (F, clamped_components)
    
    return probabilities.astype(np.float64), bic_difference

def graph_cut(probabilities: np.ndarray, edges: np.ndarray, edge_weights: np.ndarray) -> np.ndarray:
    unary_cost = (1.0 - probabilities) # Graph cut measures in costs, so low is better.
    
    # Stabalize cost so that the best option always equals zero
    unary_cost -= unary_cost.min(axis=1, keepdims=True)
    
    _, components = probabilities.shape
    # K×K identity (0 on diagonal, 1 elsewhere)
    identity_penalty_matrix = (1.0 - np.eye(components, dtype=np.float64))
    
    labels = gco.cut_general_graph(
        edges, # node adjacency indices
        edge_weights, # larger weight means please dont cut, smaller weight means its oke to cut
        unary_cost * 0.5,  # the cost to assign a certain label to a node
        identity_penalty_matrix, # optional penaltiy to prevent certain faces from connecting (always 1.0 for this implementation)
        algorithm="expansion",
    )
    
    return labels.astype(np.int32)

def face_adjacency(mesh: trimesh.Trimesh) -> np.ndarray:
    return mesh.face_adjacency.astype(np.int32)

def calculate_edge_weights(
    mesh: trimesh.Trimesh,
    face_directions: np.ndarray,
    *,
    dir_power: float = 2.0 # >1 emphasizes strong alignment
) -> np.ndarray:
    vertices = mesh.vertices
    adjacency_edges = mesh.face_adjacency_edges
    face_adjacency = mesh.face_adjacency
    
    # 0 .. pi where 0 means flat and pi means faces are almost flipped relative to each other
    adjacency_angles = mesh.face_adjacency_angles.copy()
    
    # invert/normalize to make flat 1.0 and sharp 0
    edge_weights = 1.0 - (adjacency_angles / np.pi) # invert to
    
    # Edge unit vectors
    evec = vertices[adjacency_edges[:, 1]] - vertices[adjacency_edges[:, 0]]
    elen = np.linalg.norm(evec, axis=1)
    En   = np.zeros_like(evec)
    nz   = elen > 1e-12
    En[nz] = evec[nz] / elen[nz, None]

    # Alignment with both adjacent faces (signless)
    d0 = face_directions[face_adjacency[:, 0]]
    d1 = face_directions[face_adjacency[:, 1]]
    a0 = np.abs(np.einsum('ij,ij->i', En, d0))
    a1 = np.abs(np.einsum('ij,ij->i', En, d1))
    align = 0.5 * (a0 + a1)                      # 0..1
    dir_factor = np.clip(align, 0.0, 1.0) ** float(dir_power)
    
    return (edge_weights * dir_factor).astype(np.float64)

def load_mesh(path: str) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(path, process=False)
    mesh.remove_unreferenced_vertices()
    mesh.remove_infinite_values()
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    return mesh

def split_face_components(mesh: trimesh.Trimesh):
    """
    Returns a list of (submesh, orig_face_idx) where orig_face_idx are indices
    into the *original* mesh.faces.
    """
    face_count = len(mesh.faces)
    if face_count == 0:
        return []

    # build adjacency lists
    adj = [[] for _ in range(face_count)]
    for a, b in mesh.face_adjacency:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))

    visited = np.zeros(face_count, dtype=bool)
    comps = []
    
    # flood fill until all faces are covered
    for i in range(face_count):
        if visited[i]:
            continue
        
        visited[i] = True
        stack = [i]
        comp = []
        
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
        
        comps.append(np.asarray(comp, dtype=np.int64))

    # make submeshes
    out = []
    for comp in comps:
        sub = mesh.submesh([comp], only_watertight=False, append=True, repair=False)
        out.append((sub, comp))
    
    return out

def islands_from_labels(mesh: trimesh.Trimesh, labels: np.ndarray):
    """
    Return (island_id_per_face, islands), where:
      - island_id_per_face: np.ndarray[int] with component index for each face
      - islands: List[np.ndarray[int]] of face indices per connected component
    Faces are connected only if adjacent AND share the same label.
    """
    F = len(labels)
    E = np.asarray(mesh.face_adjacency, dtype=np.int64).reshape(-1, 2)

    # Keep only adjacency edges where labels match
    same_label = labels[E[:, 0]] == labels[E[:, 1]]
    E = E[same_label]
    
    # Undirected graph over faces
    rows = np.concatenate([E[:, 0], E[:, 1]])
    cols = np.concatenate([E[:, 1], E[:, 0]])
    data = np.ones(rows.size, dtype=np.uint8)
    G = coo_matrix((data, (rows, cols)), shape=(F, F))

    n_comp, island_id = connected_components(G, directed=False)
    islands = [np.flatnonzero(island_id == i).astype(np.int64) for i in range(n_comp)]

    return islands

def segment_per_components(
    mesh: trimesh.Trimesh,
    face_directions: np.ndarray
):
    face_count = len(mesh.faces)
    global_nsdf   = np.zeros(face_count, np.float64)

    parts = split_face_components(mesh)
    
    for _, (sub, face_idx) in enumerate(parts):
        if sub.faces.shape[0] <= 1: 
            continue
        
        sdf = calculate_sdf(sub)
        global_nsdf[face_idx] = sdf
    
    global_labels = np.full(face_count, -1, np.int32)
    offset = 0
    for _, (sub, face_idx) in enumerate(parts):
        sub_labels = recursive_labeling(
            mesh,
            sub, 
            global_nsdf,
            face_directions,
            face_idx,
        )
        
        if sub_labels.size and sub_labels.max() >= 0:
            global_labels[face_idx] = sub_labels + offset
            offset += int(sub_labels.max() + 1)
        else:
            global_labels[face_idx] = offset
            offset += 1
    
    _, remap = np.unique(global_labels, return_inverse=True)
    global_labels = remap.astype(np.int32)
    return global_labels, global_nsdf

def recursive_labeling(
    original_mesh: trimesh.Trimesh,
    mesh: trimesh.Trimesh,
    nsdf: np.ndarray,
    face_directions: np.ndarray,
    face_idx: np.ndarray,
    current_depth: int = 0
) -> np.ndarray:
    MIN_FACES = 50
    MAX_DEPTH = 3
    
    if face_idx.size == 0:
        return np.zeros(0, np.int32)
    
    if (current_depth >= MAX_DEPTH or face_idx.size <= MIN_FACES):
        return np.zeros(face_idx.size, np.int32)  # TODO should not be zero, should be merged into surrounding labels

    sub_nsdf = nsdf[face_idx]
    sub_face_directions = face_directions[face_idx]
    edges = face_adjacency(mesh)
    edge_weights = calculate_edge_weights(mesh, sub_face_directions)
    probs, bic_difference = gaussian_mixture(sub_nsdf, 2)
    
    if bic_difference < 500:
        return np.zeros(face_idx.size, np.int32)
    
    labels = graph_cut(probs, edges, edge_weights)
    
    islands = islands_from_labels(mesh, labels)
    
    out = -np.ones(face_idx.size, np.int32)
    next_label = 0
    for island in islands:
        island_face_idx = face_idx[island]
        sub = original_mesh.submesh([island_face_idx], only_watertight=False, append=True, repair=False)
        
        sub_labels = recursive_labeling(
            original_mesh,
            sub, 
            nsdf, 
            face_directions, 
            island_face_idx,
            current_depth+1
        )
        
        if sub_labels.max() <= 0:
            out[island] = next_label
            next_label += 1
        else:
            out[island] = sub_labels + next_label
            next_label += int(sub_labels.max() + 1)
    
    _, remap = np.unique(out, return_inverse=True)
    return remap.astype(np.int32)
        
    
def show_viewer(
    mesh: trimesh.Trimesh,
    sdf: np.ndarray,
    labels: np.ndarray,
    face_directions: np.ndarray,
):
    label_colors = labels_to_colors(labels)
    sdf_colors = sdf_to_colors(sdf)

    def pr_from_colors(colors):
        m = mesh.copy(); 
        m.visual = trimesh.visual.ColorVisuals(mesh=mesh, face_colors=colors)
        return pyrender.Mesh.from_trimesh(m, smooth=False)
    
    # Different visualizations
    pr_sdf = pr_from_colors(sdf_colors)
    pr_labels = pr_from_colors(label_colors)
    
    pr_curvline = curvature_direction_lines(mesh, face_directions, scale=0.6, every=3, color=(1,0,0,1))

    scene = pyrender.Scene(bg_color=(240, 240, 240, 255), ambient_light=np.ones(3))
    viewer = pyrender.Viewer(
        scene=scene,
        viewport_size=(1000, 800),
        render_flags={'flat': True},
        run_in_thread=True,              # << key bit: don’t block, allow UI thread
    )
    
    current = {'node': None, 'mode': 'sdf'}
    
    # view modes can be one or multiple nodes
    modes = {
        'sdf'       : [pr_sdf],
        'labels'    : [pr_labels],
        'curv_lines': [pr_sdf, pr_curvline],   # overlay lines on SDF (pick any base)
    }

    current = {'mode': None, 'nodes': []}

    def switch(mode):
        if current['mode'] == mode: return
        viewer.render_lock.acquire()
        try:
            for nd in current['nodes']:
                scene.remove_node(nd)
            current['nodes'] = [scene.add(m) for m in modes[mode]]
            current['mode']  = mode
        finally:
            viewer.render_lock.release()

    switch('sdf') # default to sdf
    
    root = tk.Tk()
    root.title("Viewer mode")

    tk.Button(root, text="Show SDF", command=lambda: switch('sdf')).pack(fill='x', padx=10, pady=6)
    tk.Button(root, text="Show Labels", command=lambda: switch('labels')).pack(fill='x', padx=10, pady=6)
    tk.Button(root, text="Curv Lines", command=lambda: switch('curv_lines')).pack(fill='x', padx=10, pady=6)
     
    root.mainloop()

def main():
    ap = argparse.ArgumentParser(description="Mesh segmentation")
    ap.add_argument("input", help="path to mesh (.obj/.ply/...)")
    args = ap.parse_args()

    mesh = load_mesh(args.input)
    
    face_directions = calculate_face_directions(mesh)
    labels, nsdf = segment_per_components(mesh, face_directions)
    
    
    #nsdf = calculate_nsdf(mesh)
    #edge_weights = calculate_edge_weights(mesh, face_directions)
    #probs = gaussian_mixture(nsdf, 2)
    #edges = face_adjacency(mesh)
    #labels = graph_cut(probs, edges, edge_weights)
    #islands = islands_from_labels(mesh, labels)
    show_viewer(mesh, nsdf, labels, face_directions)

if __name__ == "__main__":
    main()