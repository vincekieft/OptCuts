#!/usr/bin/env python3
# EXACT Shapira08: SDF(face) -> GMM -> k-way alpha-expansion with dihedral pairwise
# Requires: trimesh, numpy, scikit-learn, gco-wrapper (or pygco), libigl python (Embree)
import argparse
from pathlib import Path
import numpy as np
if not hasattr(np, "infty"):
    np.infty = np.inf

import trimesh
from igl.embree import shape_diameter_function
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
import gco
try:
    import pyrender
except Exception:
    pyrender = None


# ---------------------------
# Helpers
# ---------------------------

def robust_norm01(x: np.ndarray, q_lo=1.0, q_hi=99.0) -> np.ndarray:
    lo, hi = np.percentile(x, [q_lo, q_hi])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)

def log_remap01(x: np.ndarray, alpha: float = 4.0) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.log(x * alpha + 1.0) / np.log(alpha + 1.0)

def random_label_colors(k, alpha=255, seed=42):
    if seed is not None:
        np.random.seed(seed)
    rgb = np.random.randint(0, 256, size=(int(k), 3), dtype=np.uint8)
    return np.hstack([rgb, np.full((int(k), 1), alpha, dtype=np.uint8)])

def shade_colors_by_sdf(face_colors: np.ndarray, nsdf_face: np.ndarray,
                        low: float = 0.60, high: float = 1.40) -> np.ndarray:
    ns = np.clip(nsdf_face.astype(np.float32), 0.0, 1.0)
    scale = (low + ns * (high - low)).astype(np.float32)
    rgb = face_colors[:, :3].astype(np.float32) * scale[:, None]
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return np.concatenate([rgb, face_colors[:, 3:4]], axis=1)

def sdf_to_heatmap(nsdf_face: np.ndarray, alpha: int = 255) -> np.ndarray:
    t = np.clip(nsdf_face.astype(np.float64), 0.0, 1.0)
    pos  = np.array([0.00, 0.25, 0.50, 0.75, 1.00], dtype=np.float64)
    colR = np.array([255, 240, 110,  60,  25], dtype=np.float64)
    colG = np.array([235, 220, 205, 170,  70], dtype=np.float64)
    colB = np.array([ 60, 100, 200, 230, 160], dtype=np.float64)
    r = np.interp(t, pos, colR)
    g = np.interp(t, pos, colG)
    b = np.interp(t, pos, colB)
    rgba = np.stack([r, g, b, np.full_like(r, alpha)], axis=1)
    return np.clip(rgba, 0, 255).astype(np.uint8)


# ---------------------------
# SDF per face
# ---------------------------

def sdf_per_face(mesh: trimesh.Trimesh, rays: int = 128, flip: bool = False) -> np.ndarray:
    V, F = mesh.vertices, mesh.faces
    C = mesh.triangles_center
    Nf = mesh.face_normals
    if flip:
        Nf = -Nf
    sdf = shape_diameter_function(V, F, C, Nf, rays)  # ~[0,1]
    sdf = np.nan_to_num(sdf, nan=np.nanmedian(sdf))
    return sdf.astype(np.float64)

def sdf_per_face_stable(mesh: trimesh.Trimesh, rays: int = 128) -> np.ndarray:
    return sdf_per_face(mesh, rays=rays, flip=False)

# ---------------------------
# Step 1: GMM posteriors (soft labels)
# ---------------------------

def gmm_unary_exact(nsdf_face: np.ndarray, k: int, eps: float = 1e-12):
    X = nsdf_face.reshape(-1, 1)
    gm = GaussianMixture(n_components=k, covariance_type="full",
                         n_init=3, max_iter=200, random_state=0).fit(X)
    P = np.clip(gm.predict_proba(X), eps, 1.0)
    U = -np.log(P)  # (F, K)
    return U

def gmm_unary_auto(nsdf_face: np.ndarray,
                   *,
                   k_min: int = 1,
                   k_max: int = 6,
                   method: str = "bic",            # "bic" or "dp"
                   min_comp_frac: float = 0.03,    # min cluster mass (fraction of faces)
                   eps: float = 1e-12):
    """
    Return (U, K) where U is the unary (F x K) = -log P(label|face).
    K is chosen automatically (<= k_max).
    """
    F = int(nsdf_face.size)
    if F == 0:
        return np.zeros((0, 1)), 1
    X = nsdf_face.reshape(-1, 1)
    k_min = max(1, k_min)
    k_max = max(k_min, min(k_max, F))

    if method == "dp":
        # Bayesian GMM prunes unused components by itself
        bgm = BayesianGaussianMixture(
            n_components=k_max,
            weight_concentration_prior_type="dirichlet_process",
            covariance_type="full",
            max_iter=300, n_init=1, random_state=0
        ).fit(X)
        probs_full = np.clip(bgm.predict_proba(X), eps, 1.0)
        # keep only components with decent mass
        mass = probs_full.mean(axis=0)                      # ~= weights_
        keep = np.where(mass >= max(min_comp_frac, 1.0/F))[0]
        if keep.size == 0:                                  # fallback: biggest only
            keep = np.array([mass.argmax()])
        probs = probs_full[:, keep]
        probs /= probs.sum(axis=1, keepdims=True)
        U = -np.log(probs)
        return U, int(keep.size)

    # default: BIC sweep
    best_gm, best_bic, best_k = None, np.inf, k_min
    for k in range(k_min, k_max + 1):
        gm = GaussianMixture(n_components=k, covariance_type="full",
                             n_init=3, max_iter=200, random_state=0).fit(X)
        bic = gm.bic(X)
        if bic < best_bic:
            best_bic, best_gm, best_k = bic, gm, k
    probs = np.clip(best_gm.predict_proba(X), eps, 1.0)
    U = -np.log(probs)
    return U, int(best_k)

def extract_label_components(mesh: trimesh.Trimesh, labels: np.ndarray, *, min_faces: int = 1):
    """
    Yield (label_id, comp_faces_idx) for each connected component within a label.
    comp_faces_idx are indices w.r.t. the ORIGINAL mesh.faces.
    Not grouped by label; caller can order as desired.
    """
    assert labels.shape[0] == mesh.faces.shape[0], "labels must be per-face"
    F = mesh.faces.shape[0]

    if F == 0:
        return

    E = mesh.face_adjacency.astype(np.int32)

    # No adjacency at all: each face is its own component.
    if E.size == 0:
        if min_faces <= 1:
            for i in range(F):
                yield int(labels[i]), np.array([i], dtype=np.int64)
        return

    # Build adjacency only across same-label neighbors
    same_label_mask = labels[E[:, 0]] == labels[E[:, 1]]
    E_same = E[same_label_mask]

    adj = [[] for _ in range(F)]
    for a, b in E_same:
        a = int(a); b = int(b)
        adj[a].append(b)
        adj[b].append(a)

    visited = np.zeros(F, dtype=bool)

    for f0 in range(F):
        if visited[f0]:
            continue
        lab = int(labels[f0])
        # flood only within this label via same-label adjacency
        stack = [f0]
        comp = []
        visited[f0] = True
        while stack:
            u = stack.pop()
            if labels[u] != lab:
                continue
            comp.append(u)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
        if len(comp) >= min_faces:
            yield lab, np.asarray(comp, dtype=np.int64)

def build_component_mesh(mesh: trimesh.Trimesh, face_idx: np.ndarray) -> trimesh.Trimesh:
    """
    Create a brand-new trimesh for the given faces (no submesh()).
    Remaps vertex indices to a compact [0..n-1] range.
    """
    face_idx = np.asarray(face_idx, dtype=np.int64)
    F_src = mesh.faces[face_idx]                      # (m, 3) original vertex IDs
    unique_verts = np.unique(F_src.reshape(-1))       # used vertex IDs

    # old->new index map
    remap = -np.ones(len(mesh.vertices), dtype=np.int64)
    remap[unique_verts] = np.arange(unique_verts.shape[0], dtype=np.int64)

    V_new = mesh.vertices[unique_verts]
    F_new = remap[F_src]

    comp = trimesh.Trimesh(vertices=V_new, faces=F_new, process=False)

    # Optional: carry over face colors if present (keeps your viz consistent)
    try:
        if hasattr(mesh, "visual") and mesh.visual is not None:
            fc = getattr(mesh.visual, "face_colors", None)
            if fc is not None and len(fc) == len(mesh.faces):
                comp.visual = trimesh.visual.ColorVisuals(mesh=comp, face_colors=fc[face_idx])
    except Exception:
        pass

    return comp

def export_label_components(mesh: trimesh.Trimesh,
                            labels: np.ndarray,
                            out_dir: str | Path,
                            *,
                            stem: str = "mesh",
                            min_faces: int = 1) -> list[Path]:
    """
    Export each unconnected component (within a label) as its own OBJ.
    Builds a new mesh per component (no submesh()).
    Filenames: <stem>_partNNN_labelL.obj
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Gather components, order by smallest face index for stable numbering
    comps = []
    for lab, comp_faces in extract_label_components(mesh, labels, min_faces=min_faces):
        comps.append((lab, comp_faces))
    comps.sort(key=lambda t: int(t[1].min()) if t[1].size else -1)

    written = []
    for part_idx, (lab, comp_faces) in enumerate(comps):
        comp_mesh = build_component_mesh(mesh, comp_faces)
        path = out_dir / f"{stem}_part{part_idx:03d}_label{lab}.obj"
        comp_mesh.export(path, file_type="obj")
        written.append(path)

    return written

# ---------------------------
# Step 2: dihedral pairwise (exact paper formula + optional tweaks)
# ---------------------------

def signed_dihedral_angles(mesh: trimesh.Trimesh) -> np.ndarray:
    E = mesh.face_adjacency
    if E.size == 0:
        return np.empty((0,), np.float64)
    n = mesh.face_normals
    e_idx = mesh.face_adjacency_edges
    v0 = mesh.vertices[e_idx[:, 0]]
    v1 = mesh.vertices[e_idx[:, 1]]
    e_hat = v1 - v0
    e_hat /= np.linalg.norm(e_hat, axis=1, keepdims=True) + 1e-12
    n1 = n[E[:, 0]]
    n2 = n[E[:, 1]]
    cosang = np.clip((n1 * n2).sum(1), -1.0, 1.0)
    ang = np.arccos(cosang)
    s = np.sign((np.cross(n1, n2) * e_hat).sum(1))
    return ang * s

def dihedral_edge_term(mesh: trimesh.Trimesh,
                       *, eps: float = 1e-12,
                       pairwise_floor: float = 0.0,
                       edge_len_power: float = 0.0,
                       concavity_bias: float = 1.0):
    E = mesh.face_adjacency.astype(np.int32)
    if E.size == 0:
        return E, np.empty((0,), np.float64)
    theta = mesh.face_adjacency_angles
    w = -np.log((theta / np.pi) + eps)
    if pairwise_floor > 0.0:
        w = pairwise_floor + w
    if concavity_bias != 1.0:
        theta_s = signed_dihedral_angles(mesh)
        concave = theta_s < 0.0
        w[concave] *= float(concavity_bias)
    if edge_len_power != 0.0:
        se = mesh.face_adjacency_edges
        el = np.linalg.norm(mesh.vertices[se[:, 0]] - mesh.vertices[se[:, 1]], axis=1)
        el = (el / (el.mean() + 1e-12)) ** float(edge_len_power)
        w *= el
    return E, w.astype(np.float64)

# ---------------------------
# α-expansion (float64) — Potts with per-edge weights
# ---------------------------

def shapira08_graph_cut(unary_e1: np.ndarray, edges: np.ndarray, w_edge: np.ndarray,
                        lam: float, add_eps: bool = True) -> np.ndarray:
    N, K = unary_e1.shape
    E = edges.astype(np.int32)
    U = unary_e1.astype(np.float64)
    U -= U.min(axis=1, keepdims=True)
    if add_eps:
        U += (1e-9 * np.arange(K, dtype=np.float64))[None, :]
    W = (lam * w_edge).astype(np.float64)
    V = (1.0 - np.eye(K, dtype=np.float64))
    labels = gco.cut_general_graph(E, W, U, V, algorithm="expansion")
    return labels.astype(np.int32)

# ---------------------------
# Single-mesh pipeline (EXACT Shapira08)
# ---------------------------

def segment_mesh_shapira08_exact(mesh: trimesh.Trimesh,
                                       *,
                                       k_min=1, k_max=4,
                                       rays=128, auto_lambda_ratio=1,
                                       k_method="bic", min_faces_per_cluster=250,
                                       debug=False, lam=None):
    mesh.fix_normals()
    sdf_f = sdf_per_face_stable(mesh, rays=rays)
    nsdf_f = log_remap01(robust_norm01(sdf_f, 1, 99), alpha=4.0)

    # cap K for small parts so each cluster has enough faces
    F = len(mesh.faces)
    k_cap = max(k_min, min(k_max, max(1, F // max(1, min_faces_per_cluster))))

    U, K = gmm_unary_auto(nsdf_f, k_min=k_min, k_max=k_cap,
                          method=k_method,
                          min_comp_frac=max(1.0/F, 1.0/min_faces_per_cluster))
    if debug:
        print(f"[auto-K] F={F} -> K={K} (cap={k_cap})")

    if K <= 1:
        # trivial: everything one label, still return nsdf for viz
        return np.zeros(F, np.int32), nsdf_f, 0.0

    edges, w = dihedral_edge_term(mesh)

    # auto λ if not given
    if lam is None:
        U0 = U - U.min(axis=1, keepdims=True)
        delta = float(np.median(U0.max(axis=1))) if U0.size else 0.0
        deg   = np.bincount(edges.ravel(), minlength=U.shape[0]).mean() if edges.size else 0.0
        wmean = float(w.mean()) if w.size else 1.0
        lam = 0.0 if (delta <= 0 or deg <= 0 or wmean <= 0) else delta / (auto_lambda_ratio * deg * wmean)

    labels = shapira08_graph_cut(U, edges, w, lam, add_eps=True)
    return labels.astype(np.int32), nsdf_f, lam


# ---------------------------
# NEW: split into connected face components and segment each
# ---------------------------

def split_face_components(mesh: trimesh.Trimesh):
    """
    Returns a list of (submesh, orig_face_idx) where orig_face_idx are indices
    into the *original* mesh.faces.
    """
    F = len(mesh.faces)
    if F == 0:
        return []

    # build adjacency lists
    adj = [[] for _ in range(F)]
    for a, b in mesh.face_adjacency:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))

    visited = np.zeros(F, dtype=bool)
    comps = []
    for i in range(F):
        if visited[i]:
            continue
        stack = [i]; visited[i] = True
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
        sub = mesh.submesh([comp], append=True, repair=False)
        out.append((sub, comp))
    return out

def segment_per_components(mesh: trimesh.Trimesh, *, lam: float | None = None,
                           rays: int = 64, auto_lambda_ratio: float = 0.3,
                           debug: bool = False):
    """
    Split the mesh into connected face-components, segment each,
    and stitch back global per-face labels and nsdf.
    """
    F = len(mesh.faces)
    global_labels = np.full(F, -1, np.int32)
    global_nsdf   = np.zeros(F, np.float64)

    label_offset = 0
    parts = split_face_components(mesh)
    if debug:
        sizes = [len(idx) for _, idx in parts]
        print(f"[split] components: {len(parts)} | sizes (faces): min={min(sizes) if sizes else 0}, max={max(sizes) if sizes else 0}")

    for si, (sub, orig_idx) in enumerate(parts):
        if sub.faces.shape[0] <= 1: 
            continue
        
        if debug:
            print(f"[part {si}] faces={len(orig_idx)}")
        labels_sub, nsdf_sub, _ = segment_mesh_shapira08_exact(
            sub, lam=lam, rays=rays, auto_lambda_ratio=auto_lambda_ratio, debug=debug
        )
        # reindex labels to be unique across parts
        if labels_sub.size:
            labels_sub = labels_sub + label_offset
            label_offset += labels_sub.max() + 1 - label_offset
        global_labels[orig_idx] = labels_sub
        global_nsdf[orig_idx] = nsdf_sub

    # compact labels to 0..K-1
    if (global_labels >= 0).any():
        _, remap = np.unique(global_labels, return_inverse=True)
        global_labels = remap.astype(np.int32)

    return global_labels, global_nsdf


# ---------------------------
# CLI / Viewer
# ---------------------------
def main():
    ap = argparse.ArgumentParser(description="EXACT Shapira08 SDF+GMM+k-way graph cut (per connected component)")
    ap.add_argument("input", help="path to mesh (.obj/.ply/...)")
    ap.add_argument("--viz", choices=["labels", "sdf"], default="labels",
                    help="What to visualize: segmentation labels or SDF heatmap.")
    ap.add_argument("--no-split", action="store_true",
                    help="Process the whole mesh as one (default is to split into connected components).")
    ap.add_argument("--no-view", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--output", default="out", help="output directory")
    args = ap.parse_args()

    mesh = trimesh.load_mesh(args.input, process=False)
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    mesh.fix_normals()

    labels, nsdf = segment_per_components(mesh)

    export_label_components(mesh, labels, args.output)

    # --- visualization toggle ---
    if args.viz == "labels":
        k_all = labels.max() + 1 if labels.size else 0
        if k_all:
            base = random_label_colors(k_all, alpha=255, seed=42)[labels]
            shaded = shade_colors_by_sdf(base, nsdf, low=0.60, high=1.40)
            mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, face_colors=shaded)
    else:  # "sdf"
        face_colors = sdf_to_heatmap(nsdf, alpha=255)
        mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, face_colors=face_colors)

    if args.no_view or pyrender is None:
        mode = args.viz
        segs = labels.max()+1 if labels.size else 0
        print(f"Faces: {len(mesh.faces)} | Segments: {segs} | viz={mode} | split={not args.no_split}")
        return

    scene = pyrender.Scene(bg_color=(240, 240, 240, 255), ambient_light=np.ones(3))
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
    pyrender.Viewer(scene=scene, viewport_size=(1000, 800), render_flags={'flat': True})


if __name__ == "__main__":
    main()
