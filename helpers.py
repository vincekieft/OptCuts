import igl
import numpy as np
import pyrender
import scipy.sparse as sp
import trimesh
import scipy.sparse.linalg as spla
from matplotlib import cm

def _hsv_to_rgb_vec(h, s, v):
    """Vectorized HSV→RGB for arrays in [0,1]."""
    h = np.mod(h, 1.0)
    i = np.floor(h * 6).astype(int)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))
    i = i % 6
    r = np.where(i==0, v, np.where(i==1, q, np.where(i==2, p, np.where(i==3, p, np.where(i==4, t, v)))))
    g = np.where(i==0, t, np.where(i==1, v, np.where(i==2, v, np.where(i==3, q, np.where(i==4, p, p)))))
    b = np.where(i==0, p, np.where(i==1, p, np.where(i==2, t, np.where(i==3, v, np.where(i==4, v, q)))))
    return np.stack([r, g, b], axis=1)

def _default_palette(k, s=0.85, v=0.95, hue_offset=0.12):
    """K distinct bright colors in RGB [0,1]."""
    hues = (np.linspace(0, 1, k, endpoint=False) + hue_offset) % 1.0
    return _hsv_to_rgb_vec(hues, np.full(k, s), np.full(k, v))

def _srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

def _linear_to_srgb(c):
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * (c ** (1/2.4)) - 0.055)

def label_probs_to_colors(P, palette=None, gamma_correct=False,
                       alpha_from_conf=False, return_uint8=True):
    """
    P: (F,K) probabilities per face per label.
    palette: (K,3) or (K,4) in [0,1]. If None, a distinct RGB palette is generated.
    gamma_correct: blend in linear light (recommended=True for nicer mixes).
    alpha_from_conf: set alpha from max probability per face (confidence).
    return_uint8: return RGBA uint8 (0..255) for trimesh.
    """
    P = np.asarray(P, dtype=np.float64)
    assert P.ndim == 2, "P must be (F,K)"
    F, K = P.shape

    # normalize rows (robust if they don't sum to 1)
    row_sum = P.sum(axis=1, keepdims=True)
    P = P / np.clip(row_sum, 1e-12, None)

    if palette is None:
        palette = _default_palette(K)  # (K,3)
    pal = np.asarray(palette, dtype=np.float64)
    if pal.shape[0] != K:
        raise ValueError(f"palette has {pal.shape[0]} colors, but P has K={K}")

    # split RGB / optional alpha
    if pal.shape[1] == 3:
        pal_rgb, pal_a = pal, None
    elif pal.shape[1] == 4:
        pal_rgb, pal_a = pal[:, :3], pal[:, 3:4]
    else:
        raise ValueError("palette must be (K,3) or (K,4)")

    # optional gamma-correct mixing
    rgb_basis = _srgb_to_linear(pal_rgb) if gamma_correct else pal_rgb
    blended_rgb = P @ rgb_basis
    blended_rgb = _linear_to_srgb(blended_rgb) if gamma_correct else blended_rgb
    blended_rgb = np.clip(blended_rgb, 0, 1)

    # alpha: palette-weighted or confidence-based or 1
    if alpha_from_conf:
        A = 0.3 + 0.7 * P.max(axis=1, keepdims=True)   # 0.3..1.0
    elif pal_a is not None:
        A = np.clip(P @ pal_a, 0, 1)
    else:
        A = np.ones((F, 1), dtype=np.float64)

    rgba = np.concatenate([blended_rgb, A], axis=1)
    if return_uint8:
        return (rgba * 255).round().astype(np.uint8)
    return rgba

def sdf_to_colors(nsdf_face: np.ndarray, alpha: int = 255) -> np.ndarray:
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

def random_label_colors(k, alpha=255, seed=42):
    if seed is not None:
        np.random.seed(seed)
    rgb = np.random.randint(0, 256, size=(int(k), 3), dtype=np.uint8)
    return np.hstack([rgb, np.full((int(k), 1), alpha, dtype=np.uint8)])

def labels_to_colors(labels: np.ndarray) -> np.ndarray:
    label_count = labels.max() + 1
    
    return random_label_colors(label_count, alpha=255, seed=42)[labels]

def curvature_to_face_colors(curv: np.ndarray,
                             cmap: str = "viridis",
                             percentiles=(1.0, 99.0),
                             return_uint8: bool = True) -> np.ndarray:
    """
    Map a per-face scalar (curv) to RGBA colors.
    Uses robust [p1, p99] normalization to avoid outliers dominating.
    """
    x = np.asarray(curv, np.float64).reshape(-1)
    lo, hi = np.nanpercentile(x, percentiles)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        y = np.zeros_like(x)
    else:
        y = np.clip((x - lo) / (hi - lo + 1e-12), 0.0, 1.0)

    rgba = cm.get_cmap(cmap)(y)  # float RGBA in [0,1]
    return (rgba * 255).round().astype(np.uint8) if return_uint8 else rgba

def curvature_direction_lines(mesh: trimesh.Trimesh,
                              dir_f: np.ndarray,
                              *,
                              scale=0.6,            # multiplies the per-face size
                              every=3,
                              lift=0.003,          # fraction of bbox diagonal
                              size_by="edge_rms",  # 'edge_rms' | 'edge_mean' | 'sqrt_area'
                              color=(1,0,0,1)) -> pyrender.Mesh:
    """
    Draw small line segments centered on faces, oriented by dir_f (F,3).
    Segment length scales per face using `size_by`.
    """
    import numpy as np
    import pyrender

    Fcnt = len(mesh.faces)
    idx = np.arange(Fcnt, dtype=np.int32)[::every]
    if idx.size == 0:
        raise ValueError("'every' too large for number of faces")

    # Centers, normals, directions for selected faces
    C = mesh.triangles_center[idx].astype(np.float32)   # (m,3)
    N = mesh.face_normals[idx].astype(np.float32)       # (m,3)
    D = np.asarray(dir_f, dtype=np.float32)[idx]        # (m,3)

    # Ensure tangent & unit
    D = D - (D * N).sum(1, keepdims=True) * N
    D /= (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12)

    # --- Per-face size ---
    # Use triangle coordinates to get edge lengths
    tri = mesh.triangles[idx].astype(np.float32)        # (m,3,3)
    e0 = tri[:,1] - tri[:,0]
    e1 = tri[:,2] - tri[:,1]
    e2 = tri[:,0] - tri[:,2]
    L0 = np.linalg.norm(e0, axis=1)
    L1 = np.linalg.norm(e1, axis=1)
    L2 = np.linalg.norm(e2, axis=1)

    if size_by == "edge_rms":
        face_size = np.sqrt((L0**2 + L1**2 + L2**2) / 3.0)   # robust for skinny tris
    elif size_by == "edge_mean":
        face_size = (L0 + L1 + L2) / 3.0
    elif size_by == "sqrt_area":
        face_size = np.sqrt(mesh.area_faces[idx])
    else:
        raise ValueError("size_by must be 'edge_rms', 'edge_mean', or 'sqrt_area'")

    # Half-length per face (full length = scale * face_size)
    half = (0.5 * scale * face_size)[:, None].astype(np.float32)

    # Lift lines off the surface along the face normal
    bbox_diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    lift_abs  = (lift * bbox_diag) if bbox_diag > 0 else float(lift)
    offset    = (lift_abs * N).astype(np.float32)

    P0 = C - half * D + offset
    P1 = C + half * D + offset

    # Interleave endpoints so (2i, 2i+1) is one GL_LINES segment
    m = idx.size
    positions = np.empty((2*m, 3), dtype=np.float32)
    positions[0::2] = P0
    positions[1::2] = P1
    indices = np.arange(2*m, dtype=np.int32).reshape(m, 2)

    prim = pyrender.Primitive(
        positions=positions,
        indices=indices,
        material=pyrender.MetallicRoughnessMaterial(baseColorFactor=color),
        mode=pyrender.constants.GLTF.LINES,
    )
    return pyrender.Mesh([prim])

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
                     smooth_time=40.0,
                     sigma_angle_deg=40.0,
                     theta_stop_deg=85.0,
                     edge_len_power=2,
                     inner=None, outer=None, gamma=1.0) -> np.ndarray:
    """Solve (C + t L) x = C s ; C = diag(confidence). Robust to flat/zero SDF."""
    s = np.asarray(sdf, np.float64)
    F = int(len(mesh.faces))
    if F == 0:
        return s.copy()

    # sanitize input
    s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)

    # if SDF is (near) constant, nothing to smooth
    if np.ptp(s) < 1e-12:  # max - min
        return s.copy()

    # 1) confidence (low near open boundaries)
    conf = boundary_confidence_faces(mesh, inner=inner, outer=outer, gamma=gamma).astype(np.float64)
    conf = np.nan_to_num(conf, nan=0.0, posinf=1.0, neginf=0.0)
    conf = np.clip(conf, 0.0, 1.0)

    # if everything is zero confidence, we cannot anchor the solution — return constant
    if conf.max() <= 0.0:
        base = float(np.nanmedian(s))
        return np.full_like(s, base)

    # small floor so (C + tL) is strictly PD
    conf = np.maximum(conf, 1e-8)

    # 2) blend toward neutral near boundary
    base = float(np.nanmedian(s))
    s0 = conf * s + (1.0 - conf) * base

    # 3) Laplacian
    L = face_laplacian(mesh, sigma_angle_deg, theta_stop_deg, edge_len_power).tocsr()
    if np.isnan(L.data).any():
        L.data[np.isnan(L.data)] = 0.0

    # 4) Solve (C + tL + εI) x = C s0
    C = sp.diags(conf)
    ridge = 1e-8
    A = (C + smooth_time * L + ridge * sp.eye(F, format='csr'))
    b = C @ s0

    try:
        x = spla.spsolve(A, b)
    except Exception:
        # fallback to CG if direct solve complains
        x, _ = spla.cg(A, b, maxiter=2000, tol=1e-10)
    return np.asarray(x)

def calculate_face_directions(mesh, mode="min", radius=10) -> np.ndarray:
    """
    Per-face principal direction, globally sign-coherent, tangent, unit.
    which: 'min' ≈ along limb; 'max' ≈ around limb.
    radius: larger => smoother/more stable directions.
    """
    V = np.asarray(mesh.vertices, float)
    F = np.asarray(mesh.faces,   np.int64)

    # directions first, values second (libigl convention)
    pd1, pd2, _, _, _ = igl.principal_curvature(V, F, radius=radius, useKring=True)
    dir_v = pd2 if mode == "min" else pd1

    # --- face average with per-face sign fix (avoid cancellation inside a face)
    a, b, c = F[:, 0], F[:, 1], F[:, 2]
    v0, v1, v2 = dir_v[a], dir_v[b], dir_v[c]
    v1 = np.where(((v1*v0).sum(1, keepdims=True) < 0), -v1, v1)
    v2 = np.where(((v2*v0).sum(1, keepdims=True) < 0), -v2, v2)
    d = (v0 + v1 + v2) / 3.0                               # (F,3)

    # project to tangent & normalize
    n = mesh.face_normals
    d -= (d*n).sum(1, keepdims=True) * n
    d /= (np.linalg.norm(d, axis=1, keepdims=True) + 1e-12)

    # --- global sign propagation over the face graph (make neighbors agree)
    E = mesh.face_adjacency.astype(np.int32)
    Fcnt = len(F)
    nbrs = [[] for _ in range(Fcnt)]
    for u, v in E:
        nbrs[u].append(v); nbrs[v].append(u)

    seen = np.zeros(Fcnt, bool)
    for seed in range(Fcnt):
        if seen[seed]: continue
        stack = [seed]; seen[seed] = True
        while stack:
            i = stack.pop()
            for j in nbrs[i]:
                if not seen[j]:
                    if np.dot(d[i], d[j]) < 0: d[j] = -d[j]
                    seen[j] = True
                    stack.append(j)

    # optional tiny neighbor smoothing (helps flats)
    for _ in range(3):
        acc = d.copy()
        for i in range(Fcnt):
            for j in nbrs[i]:
                v = d[j]
                if np.dot(v, d[i]) < 0: v = -v
                acc[i] += v
        acc -= (acc*n).sum(1, keepdims=True) * n
        d = acc / (np.linalg.norm(acc, axis=1, keepdims=True) + 1e-12)

    return d

def inpaint_sdf_nans(mesh, sdf, *, smooth_time=40.0,
                     sigma_angle_deg=40.0, theta_stop_deg=85.0, edge_len_power=2):
    """
    Fill NaNs in 'sdf' by solving (C + t L) x = C s0 over faces.
    C=1 for valid faces, 0 for NaN faces. Tiny ridge added for SPD.
    """
    s = np.asarray(sdf, np.float64)
    F = int(len(mesh.faces))
    if F == 0 or not np.isnan(s).any():
        return np.nan_to_num(s, nan=0.0)

    # confidence: valid faces anchor the solution
    valid = np.isfinite(s)
    conf  = valid.astype(np.float64)
    conf = np.maximum(conf, 1e-8)         # floor to keep matrix invertible

    # neutral value for unknowns = median of knowns
    base = float(np.nanmedian(s)) if np.isfinite(np.nanmedian(s)) else 0.0
    s0   = np.where(valid, s, base)

    # face Laplacian you already have
    L = face_laplacian(mesh, sigma_angle_deg, theta_stop_deg, edge_len_power).tocsr()

    C = sp.diags(conf)
    A = C + smooth_time * L + 1e-8 * sp.eye(F, format='csr')   # ridge for SPD
    b = C @ s0

    try:
        x = spla.spsolve(A, b)
    except Exception:
        x, _ = spla.cg(A, b, tol=1e-10, maxiter=2000)
    return np.asarray(x, dtype=np.float64)