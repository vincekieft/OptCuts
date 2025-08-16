import argparse
import sys
import gco
import pyrender
from sklearn.mixture import GaussianMixture; sys.path.append('./sdf120/build')
import sdf120
import trimesh
import numpy as np
import tkinter as tk
from helpers import label_probs_to_colors, sdf_to_colors, labels_to_colors

# Some pyrenderer error about removed infty
if not hasattr(np, "infty"):
    np.infty = np.inf

def log_remap(x: np.ndarray, alpha: float = 3.0) -> np.ndarray:
    """
    Remaps x to log with a controllable alpha knob.
    Alpha = 0, almost linear
    Alpha >, stronger lift of darks / compression of highs
    Common uses: make thin/thickness signals or images more visible in the low range; 
    dynamic-range compression before visualization or as a feature.
    """
    return np.log(x * alpha + 1.0) / np.log(alpha + 1.0)

def robust_norm01(x: np.ndarray, low_percentile=5.0, high_percentile=95.0) -> np.ndarray:
    """
    Safe normalizes x between zero and one cliping of 1 and 99 percentile highs and lows
    """
    low, high = np.percentile(x, [low_percentile, high_percentile])
    
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros_like(x)
    
    # Shift x so that low and high are clipped to one.
    ZERO_SAFETY = 1e-9 # Prevents dividing by zero
    return np.clip((x - low) / max(high - low, ZERO_SAFETY), 0.0, 1.0)

def calculate_nsdf(
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
    return log_remap(robust_norm01(sdf))

def gaussian_mixture(nsdf: np.ndarray, k: int):
    length = nsdf.shape[0]
    
    if length == 0:
        return np.zeros((0, 1))
    
    clamped_components = int(min(k, length)) # Dont ask more gaussians than there are faces
    
    reshaped_nsdf = nsdf.reshape(-1, 1).astype(np.float64)
    
    gm = GaussianMixture(
        n_components=clamped_components,
        max_iter=200,
        n_init=10,
        random_state=0, # make the result deterministc
    ).fit(reshaped_nsdf)
    
    probabilities = np.clip(gm.predict_proba(reshaped_nsdf), 1e-12, 1.0) # (F, clamped_components)
    
    return probabilities.astype(np.float64)

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
        unary_cost,  # the cost to assign a certain label to a node
        identity_penalty_matrix, # optional penaltiy to prevent certain faces from connecting (always 1.0 for this implementation)
        algorithm="expansion",
    )
    
    return labels.astype(np.int32)

def face_adjacency(mesh: trimesh.Trimesh) -> np.ndarray:
    return mesh.face_adjacency.astype(np.int32)

def calculate_edge_weights(mesh: trimesh.Trimesh) -> np.ndarray:
    # 0 .. pi where 0 means flat and pi means faces are almost flipped relative to each other
    adjacency_angles = mesh.face_adjacency_angles.copy()
    
    # invert/normalize to make flat 1.0 and sharp 0
    edge_weights = 1.0 - (adjacency_angles / np.pi) # invert to
    
    return edge_weights.astype(np.float64)

def load_mesh(path: str) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(path, process=False)
    mesh.remove_unreferenced_vertices()
    mesh.remove_infinite_values()
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    return mesh

def show_viewer(
    mesh: trimesh.Trimesh,
    sdf: np.ndarray,
    probabilities: np.ndarray,
    labels: np.ndarray
):
    label_colors = labels_to_colors(labels)
    sdf_colors = sdf_to_colors(sdf)
    prob_colors = label_probs_to_colors(probabilities, gamma_correct=True)
    
    # Different visualizations
    mesh_sdf   = mesh.copy()
    mesh_sdf.visual = trimesh.visual.ColorVisuals(mesh=mesh, face_colors=sdf_colors)
    pr_sdf = pyrender.Mesh.from_trimesh(mesh_sdf, smooth=False)

    mesh_probs   = mesh.copy()
    mesh_probs.visual = trimesh.visual.ColorVisuals(mesh=mesh, face_colors=prob_colors)
    pr_probs = pyrender.Mesh.from_trimesh(mesh_probs, smooth=False)
    
    mesh_labels = mesh.copy()
    mesh_labels.visual = trimesh.visual.ColorVisuals(mesh=mesh, face_colors=label_colors)
    pr_labels = pyrender.Mesh.from_trimesh(mesh_labels, smooth=False)

    scene = pyrender.Scene(bg_color=(240, 240, 240, 255), ambient_light=np.ones(3))
    node = scene.add(pr_sdf) # Start with SDF view
    
    viewer = pyrender.Viewer(
        scene=scene,
        viewport_size=(1000, 800),
        render_flags={'flat': True},
        run_in_thread=True,              # << key bit: don’t block, allow UI thread
    )
    
    current = {'node': node, 'mode': 'sdf'}
    meshes  = {'sdf': pr_sdf, 'probs': pr_probs, 'labels': pr_labels}
    
    def switch(mode: str):
        if current['mode'] == mode:
            return
        # Thread-safe scene mutation while viewer is running
        viewer.render_lock.acquire()
        try:
            scene.remove_node(current['node'])
            current['node'] = scene.add(meshes[mode])
            current['mode'] = mode
        finally:
            viewer.render_lock.release()

    root = tk.Tk()
    root.title("Viewer mode")

    btn1 = tk.Button(root, text="Show SDF",    command=lambda: switch('sdf'))
    btn2 = tk.Button(root, text="Show Label Probs", command=lambda: switch('probs'))
    btn3 = tk.Button(root, text="Show Labels", command=lambda: switch('labels'))
    btn1.pack(fill='x', padx=10, pady=6)
    btn2.pack(fill='x', padx=10, pady=6)
    btn3.pack(fill='x', padx=10, pady=6)

    root.mainloop()

def main():
    ap = argparse.ArgumentParser(description="Mesh segmentation")
    ap.add_argument("input", help="path to mesh (.obj/.ply/...)")
    args = ap.parse_args()

    mesh = load_mesh(args.input)
    
    nsdf = calculate_nsdf(mesh)
    probs = gaussian_mixture(nsdf, 3)
    edges = face_adjacency(mesh)
    edge_weights = calculate_edge_weights(mesh)
    
    labels = graph_cut(probs, edges, edge_weights)
    
    show_viewer(mesh, nsdf, probs, labels)

if __name__ == "__main__":
    main()