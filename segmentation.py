#!/usr/bin/env python3
# EXACT Shapira08: SDF(face) -> GMM -> k-way alpha-expansion with dihedral pairwise
# Requires: trimesh, numpy, scikit-learn, gco-wrapper (or pygco), libigl python (Embree)
import argparse
from pathlib import Path
import numpy as np
import trimesh
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
import gco

import sys; sys.path.append('./sdf120/build')
import sdf120

if not hasattr(np, "infty"):
    np.infty = np.inf
try:
    import pyrender
except Exception:
    pyrender = None

def _compute_nsdf_once(mesh: trimesh.Trimesh, *, rays: int) -> np.ndarray:
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces,    dtype=np.int32)

    sdf  = sdf120.compute_sdf(V, F, rays=rays, cone_deg=120.0, postprocess=True)
    sdf01 = np.clip(
        (sdf - np.percentile(sdf, 1)) /
        max(np.percentile(sdf, 99) - np.percentile(sdf, 1), 1e-9), 0, 1
    )
    sdf_s = smooth_blend_sdf(
        mesh, sdf01,
        smooth_time=12.0,
        sigma_angle_deg=50.0,
        theta_stop_deg=88.0,
        edge_len_power=1.0,
        inner=None, outer=None,
        gamma=2.5,
        neutral='median'
    )
    nsdf = log_remap(robust_norm01(sdf_s, 1, 99), alpha=4.0)
    return nsdf

# ---- one GC pass on a submesh (EXACTLY your weights & cut) ----
def _gc_split(submesh: trimesh.Trimesh,
              nsdf_sub: np.ndarray,
              *,
              k: int,
              lam: float | None,
              auto_lambda_ratio: float) -> tuple[np.ndarray, float, int, np.ndarray, np.ndarray, np.ndarray]:
    U, k_eff = gmm_unary_fixed(nsdf_sub, k=k)
    if k_eff <= 1:
        return np.zeros(len(submesh.faces), np.int32), 0.0, k_eff, U, np.empty((0,2), np.int32), np.empty((0,), np.float64)

    edges, w = dihedral_edge_term(submesh)

    if lam is None:
        U0 = U - U.min(axis=1, keepdims=True)
        delta = float(np.median(U0.max(axis=1))) if U0.size else 0.0
        deg   = np.bincount(edges.ravel(), minlength=U.shape[0]).mean() if edges.size else 0.0
        wmean = float(w.mean()) if w.size else 1.0
        lam = 0.0 if (delta <= 0 or deg <= 0 or wmean <= 0) else delta / (auto_lambda_ratio * deg * wmean)

    labels = shapira08_graph_cut(U, edges, w, lam, add_eps=True)
    return labels.astype(np.int32), float(lam), k_eff, U, edges.astype(np.int32), w

# ---- recursive splitter that reuses nsdf and your GC each time ----
def _recurse_gc(mesh: trimesh.Trimesh,
                nsdf: np.ndarray,
                face_idx: np.ndarray,
                *,
                depth: int,
                max_depth: int,
                k_this: int,
                k_next: int,
                lam: float | None,
                auto_lambda_ratio: float,
                min_faces: int,
                # acceptance controls:
                accept_rel_gain: float = 0.002,   # require ≥0.2% relative energy gain vs single label
                mu_diff_min: float = 0.03,        # require ≥0.03 gap in nsdf means (thickness difference)
                debug: bool = False) -> np.ndarray:

    if face_idx.size == 0:
        return np.zeros(0, np.int32)

    max_faces = max(min_faces, 3)
    if (depth >= max_depth) or (face_idx.size < max_faces):
        return np.zeros(face_idx.size, np.int32)

    sub = build_component_mesh(mesh, face_idx)
    nsdf_sub = nsdf[face_idx]

    k_use = k_this if depth == 0 else k_next
    labels, lam_used, k_eff, U, edges, w = _gc_split(
        sub, nsdf_sub, k=k_use, lam=lam, auto_lambda_ratio=auto_lambda_ratio
    )

    # if GC produced <2 labels, stop here
    if labels.max() <= 0:
        return np.zeros(face_idx.size, np.int32)

    # ---- acceptance gate: energy improvement AND nsdf thickness separation ----
    E_split  = _potts_energy(U, edges, w, lam_used, labels)
    E_single = _best_single_label_energy(U)
    gain = E_single - E_split                          # positive = split is better
    rel_gain = gain / max(abs(E_single), 1e-9)

    # thickness gap between groups
    uniq = np.unique(labels)
    mu = []
    for lab in uniq:
        vals = nsdf_sub[labels == lab]
        mu.append(float(vals.mean()) if vals.size else 0.0)
    mu_gap = 0.0 if len(mu) < 2 else float(np.max(mu) - np.min(mu))

    if debug:
        print(f"[depth {depth}] faces={face_idx.size} k={k_use} "
              f"E_single={E_single:.4g} E_split={E_split:.4g} rel_gain={rel_gain:.4%} mu_gap={mu_gap:.4f}")

    # reject weak / thickness-similar splits
    if (rel_gain < accept_rel_gain) or (mu_gap < mu_diff_min):
        if debug:
            print(f"[depth {depth}] split REJECTED (rel_gain<{accept_rel_gain} or mu_gap<{mu_diff_min})")
        return np.zeros(face_idx.size, np.int32)

    # ---- otherwise, recurse per island ----
    island_id, islands = _islands_from_labels(sub, labels)
    out = -np.ones(face_idx.size, np.int32)
    next_off = 0
    for isl in islands:
        if isl.size < max_faces or depth == max_depth:
            out[isl] = next_off
            next_off += 1
            continue

        child = _recurse_gc(
            mesh, nsdf, face_idx[isl],
            depth=depth+1, max_depth=max_depth,
            k_this=k_next, k_next=k_next,
            lam=lam, auto_lambda_ratio=auto_lambda_ratio,
            min_faces=min_faces,
            accept_rel_gain=accept_rel_gain,
            mu_diff_min=mu_diff_min,
            debug=debug
        )
        if child.max() <= 0:
            out[isl] = next_off
            next_off += 1
        else:
            out[isl] = child + next_off
            next_off += int(child.max() + 1)

    _, remap = np.unique(out, return_inverse=True)
    return remap.astype(np.int32)

# ----- helpers for energy-based acceptance -----
def _potts_energy(U: np.ndarray, edges: np.ndarray, w: np.ndarray, lam: float, labels: np.ndarray) -> float:
    # unary
    E_u = float(U[np.arange(U.shape[0]), labels].sum())
    # pairwise Potts
    if edges.size:
        cut = labels[edges[:, 0]] != labels[edges[:, 1]]
        E_p = float(lam * w[cut].sum())
    else:
        E_p = 0.0
    return E_u + E_p

def _best_single_label_energy(U: np.ndarray) -> float:
    # no pairwise term if single label everywhere
    return float(U.sum(axis=0).min())

def segment_mesh_iterative_gc(mesh: trimesh.Trimesh,
                              *,
                              rays: int = 30,
                              k0: int = 2,
                              k_next: int = 2,
                              max_depth: int = 3,
                              lam: float | None = None,
                              auto_lambda_ratio: float = 0.3,
                              min_faces: int = 120,
                              split_components_first: bool = True,
                              debug: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Top-down recursion:
      - compute nsdf once
      - initial GC with k0
      - per-island GC again with k_next, repeat up to max_depth
    Uses the same gmm_unary_fixed, dihedral_edge_term, and shapira08_graph_cut.
    """
    F = len(mesh.faces)
    if F == 0:
        return np.zeros(0, np.int32), np.zeros(0, np.float64)

    nsdf = _compute_nsdf_once(mesh, rays=rays)

    parts = [(mesh, np.arange(F, dtype=np.int64))] if not split_components_first else split_face_components(mesh)

    global_labels = np.full(F, -1, np.int32)
    offset = 0
    for si, (sub, orig_idx) in enumerate(parts):
        if debug:
            print(f"[iter-gc] part {si} faces={len(orig_idx)}")
        lab_sub = _recurse_gc(
            mesh, nsdf, orig_idx,
            depth=0, max_depth=max_depth,
            k_this=k0, k_next=k_next,
            lam=lam, auto_lambda_ratio=auto_lambda_ratio,
            min_faces=min_faces,
            debug=debug
        )
        if lab_sub.size and lab_sub.max() >= 0:
            global_labels[orig_idx] = lab_sub + offset
            offset += int(lab_sub.max() + 1)
        else:
            global_labels[orig_idx] = offset
            offset += 1

    # compact global labels
    _, remap = np.unique(global_labels, return_inverse=True)
    global_labels = remap.astype(np.int32)
    return global_labels, nsdf

def _islands_from_labels(mesh: trimesh.Trimesh, labels: np.ndarray):
    """Return island_id per face and a list of arrays (faces in each island)."""
    F = len(labels)
    adj = [[] for _ in range(F)]
    E = np.asarray(mesh.face_adjacency, dtype=np.int64).reshape(-1, 2)
    for a, b in E:
        a = int(a); b = int(b)
        adj[a].append(b); adj[b].append(a)

    island_id = -np.ones(F, dtype=np.int64)
    islands = []
    cur = 0
    for f0 in range(F):
        if island_id[f0] != -1:
            continue
        lab = labels[f0]
        stack = [f0]
        comp = []
        island_id[f0] = cur
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if island_id[v] == -1 and labels[v] == lab:
                    island_id[v] = cur
                    stack.append(v)
        islands.append(np.asarray(comp, dtype=np.int64))
        cur += 1
    return island_id, islands  # len(islands)=R

def _rag_between_islands(mesh: trimesh.Trimesh, island_id: np.ndarray):
    """Build adjacency & boundary stats between islands."""
    E = np.asarray(mesh.face_adjacency, dtype=np.int64).reshape(-1, 2)
    if E.size == 0:
        return (np.zeros((0,2), np.int64),
                np.zeros((0,), np.float64),
                np.zeros((0,), np.float64),
                np.zeros((0,), np.float64))
    I0 = island_id[E[:,0]]; I1 = island_id[E[:,1]]
    mask = I0 != I1
    if not np.any(mask):
        return (np.zeros((0,2), np.int64),
                np.zeros((0,), np.float64),
                np.zeros((0,), np.float64),
                np.zeros((0,), np.float64))
    I0 = I0[mask]; I1 = I1[mask]
    pairs = np.stack([np.minimum(I0, I1), np.maximum(I0, I1)], axis=1)

    # boundary attributes
    theta = np.asarray(mesh.face_adjacency_angles, dtype=np.float64)[mask]       # [0,pi]
    # signed concavity
    eidx  = np.asarray(mesh.face_adjacency_edges, dtype=np.int64)[mask]
    v0, v1 = mesh.vertices[eidx[:,0]], mesh.vertices[eidx[:,1]]
    eh = v1 - v0; eh /= (np.linalg.norm(eh, axis=1, keepdims=True) + 1e-12)
    n  = np.asarray(mesh.face_normals, dtype=np.float64)
    n0, n1 = n[E[mask,0]], n[E[mask,1]]
    sgn = np.sign((np.cross(n0, n1) * eh).sum(1))   # <0 concave

    keys, inv = np.unique(pairs, axis=0, return_inverse=True)
    cnt   = np.bincount(inv).astype(np.float64)
    thavg = np.bincount(inv, weights=theta, minlength=len(keys)) / np.maximum(cnt, 1.0)
    concf = np.bincount(inv, weights=(sgn < 0).astype(np.float64), minlength=len(keys)) / np.maximum(cnt, 1.0)

    return keys.astype(np.int64), thavg, concf, cnt

def _island_stats(nsdf: np.ndarray, islands: list[np.ndarray]):
    R = len(islands)
    size = np.array([len(ix) for ix in islands], dtype=np.int64)
    # mean/std via sums
    mu   = np.zeros(R, dtype=np.float64)
    sig  = np.zeros(R, dtype=np.float64)
    for i, ix in enumerate(islands):
        vals = nsdf[ix]
        mu[i]  = float(np.mean(vals)) if ix.size else 0.0
        sig[i] = float(np.std(vals) + 1e-9)
    return size, mu, sig

def _bhattacharyya_1d(mu1, s1, mu2, s2):
    return 0.25*np.log((s1*s1 + s2*s2)/(2.0*s1*s2)) + ((mu1-mu2)**2)/(4.0*(s1*s1 + s2*s2))

def merge_label_islands(
    submesh: trimesh.Trimesh,
    labels: np.ndarray,   # (F,)
    nsdf:   np.ndarray,   # (F,)
    *,
    min_region_faces: int = 150,     # force-merge islands smaller than this
    theta_keep_deg: float = 65.0,    # avoid merging across stronger creases than this (unless island is tiny)
    concave_veto_frac: float = 0.85, # avoid mostly concave boundaries (unless island is tiny)
    target_islands: int | None = None,  # keep merging similar neighbors until <= target
    mu_tol: float = 0.06,            # SDF-mean similarity threshold
    bhat_tol: float = 0.03,          # OR Bhattacharyya threshold
    max_passes: int = 12,
) -> np.ndarray:
    """Merge works on connected *islands*, not whole labels. Only adjacent islands can merge."""
    labels = labels.astype(np.int32, copy=True)
    F = labels.size
    if F == 0:
        return labels

    # Initial islands
    island_id, islands = _islands_from_labels(submesh, labels)
    θ_keep = np.deg2rad(theta_keep_deg)

    def relabel_from_islands(islands_list):
        out = np.empty(F, dtype=np.int32)
        for nid, ix in enumerate(islands_list):
            out[ix] = nid
        return out

    for _ in range(max_passes):
        R = len(islands)
        if R <= 1:
            break

        keys, thavg, concf, _cnt = _rag_between_islands(submesh, island_id)
        if keys.shape[0] == 0:
            break

        size, mu, sig = _island_stats(nsdf, islands)

        # ---- Pass A: force-merge small islands
        small_ids = [i for i in range(R) if size[i] < min_region_faces]
        did_merge = False
        if small_ids:
            # build neighbor lists
            neigh = {i: [] for i in range(R)}
            for (a, b), th, cf in zip(keys, thavg, concf):
                neigh[a].append((b, th, cf))
                neigh[b].append((a, th, cf))
            for s in small_ids:
                if not neigh.get(s):
                    continue
                best = None; best_score = 1e9
                for nb, th, cf in neigh[s]:
                    # allow crossing stronger creases for tiny shards but penalize it
                    dmu = abs(mu[s] - mu[nb])
                    bh  = _bhattacharyya_1d(mu[s], sig[s], mu[nb], sig[nb])
                    crease_pen = max(0.0, (th - θ_keep)) / θ_keep
                    conc_pen   = max(0.0, cf - concave_veto_frac)
                    # prefer larger neighbor a bit (stability)
                    size_bonus = -0.05 * np.log1p(size[nb])
                    score = 2.0*dmu + 1.0*bh + 0.6*crease_pen + 0.3*conc_pen + size_bonus
                    if score < best_score:
                        best, best_score = nb, score
                if best is not None:
                    # merge island s -> best
                    island_id[islands[s]] = best
                    did_merge = True
            if did_merge:
                # rebuild islands after batch of small merges
                new_ids = island_id.copy()
                # compress ids to 0..R'-1
                uniq, inv = np.unique(new_ids, return_inverse=True)
                island_id = inv.astype(np.int64)
                islands = [np.where(island_id == i)[0] for i in range(len(uniq))]
                continue  # re-run passes with new structure

        # ---- Pass B: similarity-based merges to reach target_islands
        if target_islands is not None and R > max(1, target_islands):
            cand = []
            for (a, b), th, cf in zip(keys, thavg, concf):
                if th > θ_keep and cf > concave_veto_frac:
                    continue
                dmu = abs(mu[a] - mu[b])
                bh  = _bhattacharyya_1d(mu[a], sig[a], mu[b], sig[b])
                if (dmu <= mu_tol) or (bh <= bhat_tol):
                    score = 2.0*dmu + 1.0*bh + 0.25*(th/θ_keep)
                    # bias toward absorbing the smaller into the larger
                    a_big = size[a] >= size[b]
                    cand.append((score, a if a_big else b, b if a_big else a))
            if not cand:
                break
            # merge the single best pair and iterate
            _, keep, drop = min(cand, key=lambda t: t[0])
            island_id[islands[drop]] = keep
            # compress & rebuild
            uniq, inv = np.unique(island_id, return_inverse=True)
            island_id = inv.astype(np.int64)
            islands = [np.where(island_id == i)[0] for i in range(len(uniq))]
            continue

        # nothing to do this round
        break

    # Final: convert island ids back to compact labels
    uniq, remap = np.unique(island_id, return_inverse=True)
    return remap.astype(np.int32)

# ---------------------------
# Helpers
# ---------------------------

def robust_norm01(x: np.ndarray, low_percentile=1.0, high_percentile=99.0) -> np.ndarray:
    low, high = np.percentile(x, [low_percentile, high_percentile])
    
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros_like(x)
    
    # Shift x so that low and high are clipped to one.
    ZERO_SAFETY = 1e-9 # Prevents dividing by zero
    return np.clip((x - low) / max(high - low, ZERO_SAFETY), 0.0, 1.0)

def log_remap(x: np.ndarray, alpha: float = 4.0) -> np.ndarray:
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
                       pairwise_floor: float = 1.0,
                       edge_len_power: float = 1.0,
                       concavity_bias: float = 1.0,
                       crease_power: float = 1.6,
                       theta_clip_deg: float = 2.0):
    """
    Higher crease_power -> much stronger penalty to cut across flat areas,
    much weaker near creases. pairwise_floor keeps a base smoothness everywhere.
    """
    E = mesh.face_adjacency.astype(np.int32)
    if E.size == 0:
        return E, np.empty((0,), np.float64)

    theta = mesh.face_adjacency_angles.copy()
    # clip tiny angles to avoid infinity in -log
    theta = np.clip(theta, np.deg2rad(theta_clip_deg), np.pi)

    # base Shapira: large on flat, small near crease
    w = -np.log((theta / np.pi) + eps)     # [~0, +inf) as theta->pi..0
    w = pairwise_floor + w

    # emphasize angle contrast
    if crease_power != 1.0:
        w = w ** float(crease_power)

    # optional concavity/convexity bias
    if concavity_bias != 1.0:
        theta_s = signed_dihedral_angles(mesh)
        concave = theta_s < 0.0
        w[concave] *= float(concavity_bias)

    # length modulation (discourages tiny “chatter” cuts)
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
                        lam: float, add_eps: bool = True, unary_weight: float = 1.0) -> np.ndarray:
    """
    Energy: sum_f (unary_weight * U_f) + lam * sum_(f,g) w_fg [l_f != l_g]
    Set unary_weight < 1 to rely less on SDF; > 1 to rely more.
    """
    N, K = unary_e1.shape
    E = edges.astype(np.int32)

    U = unary_e1.astype(np.float64)
    U -= U.min(axis=1, keepdims=True)     # stabilize scale
    if add_eps:
        U += (1e-9 * np.arange(K, dtype=np.float64))[None, :]
    U *= float(unary_weight)               # << dial SDF influence

    W = (lam * w_edge).astype(np.float64)
    V = (1.0 - np.eye(K, dtype=np.float64))   # Potts
    labels = gco.cut_general_graph(E, W, U, V, algorithm="expansion")
    return labels.astype(np.int32)
# ---------------------------
# Single-mesh pipeline (EXACT Shapira08)
# ---------------------------

def gmm_unary_fixed(nsdf_face: np.ndarray, k: int, eps: float = 1e-12):
    """
    Fit a 1D GaussianMixture with FIXED k on nsdf_face and return unary = -log P.
    Robust quantile seeding; auto-downgrades if not enough faces or variance ~ 0.
    """
    x = nsdf_face.reshape(-1, 1).astype(np.float64)
    F = x.shape[0]
    if F == 0:
        return np.zeros((0, 1)), 1

    # If the component is too small or nearly constant, just return one label
    if F < 2 or float(np.std(x)) < 1e-8:
        return np.zeros((F, 1), dtype=np.float64), 1

    k_eff = int(max(1, min(k, F)))  # ensure k ≤ faces
    # robust means init at quantiles (5..95%)
    qs = np.linspace(5.0, 95.0, k_eff)
    means_init = np.percentile(x.ravel(), qs).reshape(-1, 1)

    gm = GaussianMixture(
        n_components=k_eff,
        covariance_type="full",
        n_init=1,                 # we seed => no need for multiple inits
        max_iter=200,
        random_state=0,
        means_init=means_init
    ).fit(x)

    P = np.clip(gm.predict_proba(x), eps, 1.0)        # (F, k_eff)
    U = -np.log(P)
    return U, k_eff

def _boundary_vertices_from_faces(F: np.ndarray) -> np.ndarray:
    E = np.vstack([F[:,[0,1]], F[:,[1,2]], F[:,[2,0]]])
    E = np.sort(E, axis=1)
    view = np.ascontiguousarray(E).view([('u', E.dtype), ('v', E.dtype)])
    uniq, cnt = np.unique(view, return_counts=True)
    E1 = uniq.view(E.dtype).reshape(-1,2)[cnt==1]
    return np.unique(E1.ravel())

def _vertex_geodesic_from_sources(V, F, sources):
    ii = np.hstack([F[:,0], F[:,1], F[:,2]])
    jj = np.hstack([F[:,1], F[:,2], F[:,0]])
    ww = np.linalg.norm(V[ii] - V[jj], axis=1)
    G  = sp.coo_matrix((ww, (ii, jj)), shape=(len(V), len(V)))
    G  = (G + G.T).tocsr()
    from scipy.sparse.csgraph import dijkstra
    D = dijkstra(G, directed=False, indices=sources, return_predecessors=False)
    return np.asarray(D).min(axis=0)

def boundary_confidence_faces(mesh: trimesh.Trimesh,
                              inner=None, outer=None, gamma=2.0) -> np.ndarray:
    """Per-face confidence c in [0,1]: 0 near open boundary, 1 inside."""
    V = np.asarray(mesh.vertices); F = np.asarray(mesh.faces, np.int64)
    # auto distances from median edge length if not given
    eidx = np.vstack([F[:,[0,1]], F[:,[1,2]], F[:,[2,0]]])
    med_e = float(np.median(np.linalg.norm(V[eidx[:,0]]-V[eidx[:,1]], axis=1)))
    if inner is None: inner = 2.0 * med_e
    if outer is None: outer = 10.0 * med_e

    bverts = _boundary_vertices_from_faces(F)
    if bverts.size == 0:
        return np.ones(len(F), dtype=np.float64)

    dV = _vertex_geodesic_from_sources(V, F, bverts)
    dF = dV[F].mean(axis=1)

    t = np.clip((dF - inner) / max(outer - inner, 1e-9), 0.0, 1.0)
    c = t*t*(3-2*t)            # smoothstep
    return np.power(c, float(gamma))  # sharpen confidence falloff

# ---------- angle-aware Laplacian on faces ----------
def face_laplacian(mesh: trimesh.Trimesh,
                   sigma_angle_deg=40.0,
                   theta_stop_deg=85.0,
                   edge_len_power=0.5) -> sp.csr_matrix:
    Fcount = len(mesh.faces)
    if Fcount == 0:
        return sp.csr_matrix((0,0))
    E = np.asarray(mesh.face_adjacency, np.int64).reshape(-1,2)
    theta = np.asarray(mesh.face_adjacency_angles, np.float64)
    # weights: high on flat, near zero across strong creases
    sig = np.deg2rad(max(1e-3, sigma_angle_deg))
    w = np.exp(-(theta/sig)**2)
    w[theta > np.deg2rad(theta_stop_deg)] = 0.0
    if edge_len_power != 0.0:
        eidx = np.asarray(mesh.face_adjacency_edges, np.int64)
        el = np.linalg.norm(mesh.vertices[eidx[:,0]]-mesh.vertices[eidx[:,1]], axis=1)
        el = (el/ (el.mean()+1e-12)) ** float(edge_len_power)
        w *= el
    i, j = E[:,0], E[:,1]
    W = sp.coo_matrix((w, (i, j)), shape=(Fcount, Fcount)).tocsr()
    W = W + W.T
    d = np.array(W.sum(axis=1)).ravel()
    L = sp.diags(d) - W
    return L

# ---------- main smoother/blender ----------
def smooth_blend_sdf(mesh: trimesh.Trimesh, sdf: np.ndarray,
                     *,
                     smooth_time=8.0,                 # ↑ stronger smoothing
                     sigma_angle_deg=40.0,
                     theta_stop_deg=85.0,
                     edge_len_power=0.5,
                     inner=None, outer=None, gamma=2.0,  # boundary confidence
                     neutral='median') -> np.ndarray:
    """Solve (C + t L) x = C s  ; C = diag(confidence) from boundary distance."""
    s = np.asarray(sdf, np.float64)
    F = len(mesh.faces)
    if F == 0: return s.copy()

    # 1) confidence (low near open boundaries)
    conf = boundary_confidence_faces(mesh, inner=inner, outer=outer, gamma=gamma)

    # 2) optionally blend input toward neutral near boundary before smoothing
    if neutral == 'median':
        base = float(np.nanmedian(s))
    elif neutral == 'min':
        base = float(np.nanmin(s))
    elif neutral == 'max':
        base = float(np.nanmax(s))
    else:
        base = float(neutral)
    s0 = conf * s + (1.0 - conf) * base

    # 3) Laplacian
    L = face_laplacian(mesh, sigma_angle_deg, theta_stop_deg, edge_len_power)

    # 4) Solve (C + tL) x = C*s0
    C = sp.diags(conf)
    A = C + smooth_time * L
    b = C @ s0
    x = spla.spsolve(A.tocsr(), b)
    return np.asarray(x)

def segment_mesh_shapira08_fixed_k(mesh: trimesh.Trimesh,
                                   *,
                                   k: int = 3,
                                   rays: int = 24,
                                   lam: float | None = None,
                                   auto_lambda_ratio: float = 0.3,
                                   debug: bool = False):
    """
    Shapira08: SDF (cone=120°) -> log-remapped nsdf -> GMM(K fixed) -> α-expansion (dihedral Potts)
    """
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces,    dtype=np.int32)

    # SDF with the CGAL-backed module you built (preserves face order)
    sdf  = sdf120.compute_sdf(V, F, rays=rays, cone_deg=120.0, postprocess=True)
    # After computing SDF and mapping to [0,1]
    sdf01 = np.clip((sdf - np.percentile(sdf,1)) /
                    max(np.percentile(sdf,99)-np.percentile(sdf,1),1e-9), 0, 1)

    # Stronger smoothing & stronger boundary blend:
    sdf_s = smooth_blend_sdf(
        mesh, sdf01,
        smooth_time=12.0,        # try 8 → 12 → 20 for more smoothing
        sigma_angle_deg=50.0,    # wider angular kernel (smoother)
        theta_stop_deg=88.0,     # allow smoothing across mild creases
        edge_len_power=1.0,
        inner=None, outer=None,  # auto from median edge length
        gamma=2.5,               # more aggressive “confidence” drop near boundary
        neutral='median'
    )


    nsdf_f = log_remap(robust_norm01(sdf_s, 1, 99), alpha=4.0)

    # Fixed-K unary
    U, k_eff = gmm_unary_fixed(nsdf_f, k=k)
    if debug:
        print(f"[fixed-K] requested K={k}, used K={k_eff}, faces={len(F)}")

    if k_eff <= 1:
        return np.zeros(len(mesh.faces), np.int32), nsdf_f, 0.0

    # Dihedral Potts weights (penalize label changes on flat areas, allow on creases)
    edges, w = dihedral_edge_term(mesh)

    # λ: same simple rule as before if not provided
    if lam is None:
        U0 = U - U.min(axis=1, keepdims=True)
        delta = float(np.median(U0.max(axis=1))) if U0.size else 0.0
        deg   = np.bincount(edges.ravel(), minlength=U.shape[0]).mean() if edges.size else 0.0
        wmean = float(w.mean()) if w.size else 1.0
        lam = 0.0 if (delta <= 0 or deg <= 0 or wmean <= 0) else delta / (auto_lambda_ratio * deg * wmean)

    labels = shapira08_graph_cut(U, edges, w, lam, add_eps=True)
    # labels = merge_label_islands(
    #     mesh, 
    #     labels, 
    #     nsdf_f,
    #     min_region_faces=150,       # force-merge shards
    #     theta_keep_deg=65.0,        # don’t cross very strong creases
    #     mu_tol=None, bhat_tol=None  # auto thresholds from data
    # )
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

def segment_per_components(mesh: trimesh.Trimesh, *, lam: float | None,
                           rays: int = 24, auto_lambda_ratio: float = 0.3,
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
        labels_sub, nsdf_sub = segment_mesh_iterative_gc(
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
    ap.add_argument("--lambda_ratio", type=float, default=0.3,
                    help="λ for the pairwise term. If omitted, λ is auto-chosen.")
    ap.add_argument("--lam", type=float, default=None,
                    help="λ for the pairwise term. If omitted, λ is auto-chosen.")
    ap.add_argument("--rays", type=int, default=30)
    ap.add_argument("--viz", choices=["labels", "sdf"], default="labels",
                    help="What to visualize: segmentation labels or SDF heatmap.")
    ap.add_argument("--no-split", action="store_true",
                    help="Process the whole mesh as one (default is to split into connected components).")
    ap.add_argument("--no-view", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--output", default="out", help="output directory")
    args = ap.parse_args()

    mesh = trimesh.load_mesh(args.input, process=False)
    mesh.remove_unreferenced_vertices()
    mesh.remove_infinite_values()
    mesh.merge_vertices(merge_tex=True, merge_norm=True)

    if args.no_split:
        labels, nsdf = segment_mesh_iterative_gc(
            mesh, lam=args.lam, rays=args.rays, debug=args.debug
        )
    else:
        labels, nsdf = segment_per_components(
            mesh, 
            lam=args.lam, 
            rays=args.rays, 
            auto_lambda_ratio=args.lambda_ratio, 
            debug=args.debug
        )

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
