import numpy as np
import xatlas
import time
import trimesh
from pathlib import Path
from trimesh.proximity import ProximityQuery
from trimesh.triangles import points_to_barycentric

def unwrap_with_xatlas(comp_mesh: trimesh.Trimesh) -> trimesh.Trimesh | None:
    """
    Unwrap a component mesh with xatlas and return a trimesh carrying per-vertex UVs.
    Geometry is the original 3D surface (duplicated per UV split).
    """
    if xatlas is None:
        print("[xatlas] not available; cannot fallback")
        return None

    V = comp_mesh.vertices.astype(np.float32)
    F = comp_mesh.faces.astype(np.uint32)

    atlas = xatlas.Atlas()
    atlas.add_mesh(V, F)
    atlas.generate()  # use xatlas defaults; tweak options if you want

    vmapping, indices, uvs = atlas[0]  # first (and only) mesh
    vmapping = np.asarray(vmapping, dtype=np.int64)
    indices  = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
    uvs      = np.asarray(uvs, dtype=np.float64)

    V_out3d = comp_mesh.vertices[vmapping]
    pmesh = trimesh.Trimesh(vertices=V_out3d.astype(np.float64), faces=indices, process=False)
    pmesh.visual = trimesh.visual.TextureVisuals(uv=uvs)
    return pmesh

def build_component_mesh(mesh: trimesh.Trimesh, face_idx: np.ndarray) -> trimesh.Trimesh:
    """
    Create a brand-new trimesh for the given faces. Remaps vertex indices.
    """
    face_idx = np.asarray(face_idx, dtype=np.int64)
    F_src = mesh.faces[face_idx]              # (m, 3) original vertex IDs
    used_verts = np.unique(F_src.reshape(-1)) # sorted

    remap = -np.ones(len(mesh.vertices), dtype=np.int64)
    remap[used_verts] = np.arange(used_verts.shape[0], dtype=np.int64)

    V_new = mesh.vertices[used_verts]
    F_new = remap[F_src]
    comp = trimesh.Trimesh(vertices=V_new, faces=F_new, process=False)
    return comp

def resolve_optcuts_output(input_obj: Path) -> Path | None:
    """
    Try common output patterns. Adjust if your OptCuts build writes a specific filename.
    Order: overwrite (same path), *_out.obj, *_uv.obj, *_result.obj
    """
    # If OptCuts overwrites input, that's already the path to read.
    if input_obj.exists():
        return input_obj
    stem = input_obj.with_suffix("")
    candidates = [
        stem.with_name(stem.name + "_out").with_suffix(".obj"),
        stem.with_name(stem.name + "_uv").with_suffix(".obj"),
        stem.with_name(stem.name + "_result").with_suffix(".obj"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None

def _get_mesh_uv_array(m: trimesh.Trimesh) -> np.ndarray | None:
    try:
        uv = getattr(m.visual, "uv", None)
        if uv is None:
            return None
        uv = np.asanyarray(uv, dtype=np.float64)
        return uv if uv.ndim == 2 and uv.shape[1] == 2 else None
    except Exception:
        return None

def reproject_part_uvs_to_original(original_mesh: trimesh.Trimesh,
                                   processed_mesh: trimesh.Trimesh,
                                   part_face_ids: np.ndarray,
                                   uv_face_corner_accum: np.ndarray):
    """
    Sample UVs from processed_mesh to the original_mesh corners for faces in part_face_ids.
    Writes into uv_face_corner_accum (F,3,2) in-place.
    """
    uv = _get_mesh_uv_array(processed_mesh)
    if uv is None:
        return False

    prox = ProximityQuery(processed_mesh)

    ofaces = np.asarray(part_face_ids, dtype=np.int64)
    P = original_mesh.vertices[original_mesh.faces[ofaces]]   # (m,3,3)
    P_flat = P.reshape(-1, 3)

    closest_pts, _, tri_idx = prox.on_surface(P_flat)
    tri_idx = tri_idx.astype(np.int64)

    tris_xyz = processed_mesh.triangles[tri_idx]              # (N,3,3)
    tris_vid = processed_mesh.faces[tri_idx].astype(np.int64) # (N,3)
    tris_uv  = uv[tris_vid]                                   # (N,3,2)

    bary = points_to_barycentric(tris_xyz, closest_pts)       # (N,3)
    uvs_interp = (bary[..., None] * tris_uv).sum(axis=1)      # (N,2)

    m = ofaces.shape[0]
    face_ids_rep = np.repeat(ofaces, 3)
    corner_ids   = np.tile(np.arange(3, dtype=np.int64), m)
    uv_face_corner_accum[face_ids_rep, corner_ids] = uvs_interp
    return True

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

def run_optcuts_and_project(mesh: trimesh.Trimesh,
                            labels: np.ndarray,
                            *,
                            exe: str | Path,
                            jobs: int = 4,
                            min_faces: int = 1,
                            keep_temp: bool = False,
                            verbose: bool = True,
                            timeout_sec: float = 30.0) -> np.ndarray:
    """
    Split -> run OptCuts per part (with timeout) -> project UVs back.
    If OptCuts fails or times out, fall back to xatlas for that part.
    """
    exe = Path(exe)
    if not exe.exists():
        raise FileNotFoundError(f"OptCuts binary not found: {exe}")

    F_total = mesh.faces.shape[0]
    uv_face_corner = np.full((F_total, 3, 2), np.nan, dtype=np.float64)

    # Build work list
    parts: list[tuple[int, np.ndarray]] = [
        (int(lab), np.asarray(comp_faces, dtype=np.int64))
        for lab, comp_faces in extract_label_components(mesh, labels, min_faces=min_faces)
    ]
    if verbose:
        print(f"[optcuts] queued parts: {len(parts)} (jobs={jobs}, timeout={timeout_sec:.1f}s)")

    import tempfile, subprocess
    tmp_ctx = tempfile.TemporaryDirectory(prefix="optcuts_parts_")
    tmpdir = Path(tmp_ctx.name)

    def start_one(pi: int, lab: int, face_ids: np.ndarray) -> tuple[subprocess.Popen, dict]:
        comp_mesh = build_component_mesh(mesh, face_ids)
        obj_path = tmpdir / f"part{pi:03d}_label{lab}.obj"
        comp_mesh.export(obj_path, file_type="obj")
        cmd = [str(exe), "100", str(obj_path), "0.999", "1", "0", "4.1", "1", "0"]
        pop = subprocess.Popen(cmd, start_new_session=True)
        meta = {
            "pi": pi, "lab": lab, "face_ids": face_ids,
            "obj_path": obj_path, "start": time.time()
        }
        return pop, meta

    def kill_proc(p: subprocess.Popen):
        try:
            p.terminate()
            p.wait(timeout=1.5)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass

    def fallback_with_xatlas(face_ids: np.ndarray):
        comp_mesh = build_component_mesh(mesh, face_ids)
        pmesh_fb = unwrap_with_xatlas(comp_mesh)
        if pmesh_fb is not None:
            reproject_part_uvs_to_original(mesh, pmesh_fb, face_ids, uv_face_corner)
            return True
        return False

    running: list[tuple[subprocess.Popen, dict]] = []
    idx = 0

    # Prime
    while idx < len(parts) and len(running) < jobs:
        lab, faces = parts[idx]
        running.append(start_one(idx, lab, faces))
        idx += 1

    # Loop
    while running:
        still: list[tuple[subprocess.Popen, dict]] = []
        for p, meta in running:
            pi, lab, face_ids, obj_path, t0 = meta["pi"], meta["lab"], meta["face_ids"], meta["obj_path"], meta["start"]

            # Timeout?
            if (time.time() - t0) > timeout_sec:
                if verbose:
                    print(f"[optcuts] ⏱ timeout part {pi} (label {lab}) after {timeout_sec:.1f}s -> killing & xatlas fallback")
                kill_proc(p)
                # Fallback
                ok = fallback_with_xatlas(face_ids)
                if verbose:
                    print(f"[optcuts]    fallback {'OK' if ok else 'FAILED (no xatlas?)'} for part {pi}")
                # start next if available
                if idx < len(parts):
                    lab2, faces2 = parts[idx]; idx += 1
                    still.append(start_one(idx-1, lab2, faces2))
                continue

            rc = p.poll()
            if rc is None:
                still.append((p, meta))
                continue

            # Finished
            if rc != 0:
                if verbose:
                    print(f"[optcuts] ✖ part {pi} (label {lab}) exit={rc} -> xatlas fallback")
                ok = fallback_with_xatlas(face_ids)
                if verbose:
                    print(f"[optcuts]    fallback {'OK' if ok else 'FAILED (no xatlas?)'} for part {pi}")
            else:
                out_path = resolve_optcuts_output(obj_path)
                if out_path is None:
                    if verbose:
                        print(f"[optcuts] ? part {pi} output not found -> xatlas fallback")
                    ok = fallback_with_xatlas(face_ids)
                    if verbose:
                        print(f"[optcuts]    fallback {'OK' if ok else 'FAILED (no xatlas?)'} for part {pi}")
                else:
                    try:
                        pmesh = trimesh.load_mesh(out_path, process=False)
                        uv_ok = reproject_part_uvs_to_original(mesh, pmesh, face_ids, uv_face_corner)
                        if not uv_ok:
                            if verbose:
                                print(f"[optcuts] ! part {pi} had no UVs -> xatlas fallback")
                            ok = fallback_with_xatlas(face_ids)
                            if verbose:
                                print(f"[optcuts]    fallback {'OK' if ok else 'FAILED (no xatlas?)'} for part {pi}")
                        elif verbose:
                            print(f"[optcuts] ✔ part {pi} (label {lab}) UVs OK")
                    except Exception as e:
                        if verbose:
                            print(f"[optcuts] ! load/project failed for part {pi}: {e} -> xatlas fallback")
                        ok = fallback_with_xatlas(face_ids)
                        if verbose:
                            print(f"[optcuts]    fallback {'OK' if ok else 'FAILED (no xatlas?)'} for part {pi}")

            # Start next if available
            if idx < len(parts):
                lab2, faces2 = parts[idx]; idx += 1
                still.append(start_one(idx-1, lab2, faces2))

        running = still
        if running:
            time.sleep(0.05)

    if keep_temp:
        print(f"[optcuts] kept temp: {tmpdir}")
        tmp_ctx.cleanup = lambda: None
    else:
        tmp_ctx.cleanup()

    return uv_face_corner
