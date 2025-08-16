import argparse
import os
import shutil
import signal
import subprocess
import tempfile
import numpy as np
import xatlas
import time
import trimesh
from pathlib import Path
import igl

from segmentation import export_label_components, segment_per_components

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

def find_optcuts_result(workdir: Path) -> Path | None:
    """
    Search the per-job working directory for OptCuts outputs.
    Prefers finalResult_mesh_normalizedUV.obj, then finalResult_mesh.obj,
    else the first .obj found under output/.
    """
    outdir = workdir / "output"
    if not outdir.exists():
        return None

    # Prefer normalized UVs
    hits = list(outdir.rglob("finalResult_mesh_normalizedUV.obj"))
    if hits:
        return hits[0]

    hits = list(outdir.rglob("finalResult_mesh.obj"))
    if hits:
        return hits[0]

    hits = list(outdir.rglob("*.obj"))
    return hits[0] if hits else None

def _get_mesh_uv_array(m: trimesh.Trimesh) -> np.ndarray | None:
    try:
        uv = getattr(m.visual, "uv", None)
        if uv is None:
            return None
        uv = np.asanyarray(uv, dtype=np.float64)
        return uv if uv.ndim == 2 and uv.shape[1] == 2 else None
    except Exception:
        return None

def repack_with_xatlas(merged: trimesh.Trimesh,
                       resolution: int = 2048,
                       padding_px: int = 4,
                       texels_per_unit: float | None = None,
                       rotate: bool = True) -> trimesh.Trimesh:
    V = merged.vertices.astype(np.float32)
    F = merged.faces.astype(np.uint32)
    UV = np.asarray(merged.visual.uv, dtype=np.float32)

    atlas = xatlas.Atlas()

    # Many builds of the python wrapper accept existing UVs to preserve charts.
    # If your wrapper doesn't support the kwarg, see Option B below.
    atlas.add_mesh(V, F, uvs=UV)

    # Options (attribute names differ slightly across builds; guard with hasattr)
    chart = xatlas.ChartOptions()
    para  = xatlas.ParameterizeOptions()
    pack  = xatlas.PackOptions()

    if hasattr(pack, "resolution"):       pack.resolution = int(resolution)
    if hasattr(pack, "padding"):          pack.padding = int(padding_px)
    if hasattr(pack, "rotateCharts"):     pack.rotateCharts = bool(rotate)
    if hasattr(pack, "blockAlign"):       pack.blockAlign = True
    if hasattr(pack, "bruteForce"):       pack.bruteForce = True
    if hasattr(pack, "texelsPerUnit"):    pack.texelsPerUnit = float(texels_per_unit or 0.0)
    # When texelsPerUnit==0, xatlas auto-scales to fill the texture;
    # set a value (e.g. 256) to enforce a specific density.

    atlas.generate(chart, para, pack)

    vmapping, indices, new_uv = atlas[0]
    vmapping = np.asarray(vmapping, dtype=np.int64)
    indices  = np.asarray(indices,  dtype=np.int64).reshape(-1, 3)
    new_uv   = np.asarray(new_uv,   dtype=np.float64)

    # Build a packed mesh (same geometry; vertices may be duplicated along seams)
    V_out = merged.vertices[vmapping]
    packed = trimesh.Trimesh(vertices=V_out, faces=indices, process=False)
    packed.visual = trimesh.visual.TextureVisuals(uv=new_uv)
    return packed

def reproject_part_uvs_to_original(original_mesh: trimesh.Trimesh,
                                   processed_mesh: trimesh.Trimesh,
                                   part_face_ids: np.ndarray,
                                   uv_face_corner_accum: np.ndarray) -> bool:
    """
    Seam-stable UV transfer:
      - Build the same local component we fed to OptCuts
      - Match each local face to the nearest processed face via centroids (igl)
      - For each corner, copy UV from the nearest vertex of that matched processed face
    """
    uv = _get_mesh_uv_array(processed_mesh)
    if uv is None:
        return False

    ofaces = np.asarray(part_face_ids, dtype=np.int64)

    # Local component (same as what we exported)
    comp = build_component_mesh(original_mesh, ofaces)

    # --- Make all arrays plain, C-contiguous, correct dtypes ---
    C_local = np.ascontiguousarray(np.asarray(comp.triangles_center, dtype=np.float64))  # (m,3)
    Vp      = np.ascontiguousarray(np.asarray(processed_mesh.vertices, dtype=np.float64)) # (Nv,3)
    Fp      = np.ascontiguousarray(np.asarray(processed_mesh.faces, dtype=np.int32))      # (Mf,3)
    uv_proc = np.ascontiguousarray(np.asarray(uv, dtype=np.float64))                      # (Nv,2)

    # Closest processed triangle for each local face centroid
    _, tri_idx, _ = igl.point_mesh_squared_distance(C_local, Vp, Fp)
    tri_idx = np.asarray(tri_idx, dtype=np.int64).ravel()                                  # (m,)

    # Gather processed triangle data
    proc_face_vid = Fp[tri_idx].astype(np.int64)      # (m,3)
    proc_pos      = Vp[proc_face_vid]                 # (m,3,3)
    proc_uv       = uv_proc[proc_face_vid]            # (m,3,2)

    comp_faces = np.ascontiguousarray(np.asarray(comp.faces, dtype=np.int64))             # (m,3)
    comp_verts = np.ascontiguousarray(np.asarray(comp.vertices, dtype=np.float64))        # (Nc,3)

    # For each local face, map its 3 corners
    m = ofaces.shape[0]
    for li in range(m):
        orig_f  = int(ofaces[li])
        comp_vid = comp_faces[li]                     # (3,)
        comp_pos = comp_verts[comp_vid]               # (3,3)

        # distances from each local corner to the 3 processed verts (3x3)
        d2 = ((proc_pos[li][None, :, :] - comp_pos[:, None, :]) ** 2).sum(axis=2)
        nearest_idx = d2.argmin(axis=1)               # (3,) choose which processed vertex
        uv_face_corner_accum[orig_f] = proc_uv[li][nearest_idx]

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

def resolve_absolute_path(path: str) -> str:
    script_dir = Path(__file__).resolve().parent
    exe_path = Path(path).expanduser()
    if not exe_path.is_absolute():
        exe_path = (script_dir / exe_path).resolve()
        
    if not exe_path.exists():
        raise FileNotFoundError(
            f"OptCuts binary not found. Passed '{path}', resolved to '{exe_path}'")
        
    return exe_path

def run_optcuts_and_project(mesh: trimesh.Trimesh,
                            labels: np.ndarray,
                            *,
                            exe: str | Path,
                            jobs: int = 4,
                            min_faces: int = 1,
                            keep_temp: bool = False,
                            verbose: bool = True,
                            timeout_sec: float = 60.0) -> np.ndarray:
    """
    Split -> run OptCuts per part (with timeout) -> project UVs back.
    If OptCuts fails or times out, fall back to xatlas for that part.
    """
    exe_path = resolve_absolute_path(exe)
        
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
        # Make a dedicated workdir for this job
        workdir = (tmpdir / f"job_{pi:03d}")
        (workdir / "input").mkdir(parents=True, exist_ok=True)

        comp_mesh = build_component_mesh(mesh, face_ids)
        obj_path = workdir / "input" / f"part{pi:03d}_label{lab}.obj"
        comp_mesh.export(obj_path, file_type="obj")

        # Run OptCuts with cwd=workdir so its "output/..." lands inside this job folder
        cmd = [str(exe_path), "100", str(obj_path), "0.999", "1", "0", "4.1", "1", "0"]
        pop = subprocess.Popen(
            cmd, 
            start_new_session=True, 
            cwd=workdir,
            stdout=subprocess.DEVNULL,          # <-- swallow stdout
            stderr=subprocess.DEVNULL           # <-- swallow stderr
        )
        meta = {
            "pi": pi, "lab": lab, "face_ids": face_ids,
            "workdir": workdir, "obj_path": obj_path, "start": time.time()
        }
        return pop, meta

    def kill_proc(p: subprocess.Popen):
        try:
            # kill the whole process group if possible
            if hasattr(os, "killpg"):
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                p.wait(timeout=1.0)
            else:
                p.terminate()
                p.wait(timeout=1.5)
        except Exception:
            try:
                if hasattr(os, "killpg"):
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                else:
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
            pi, lab, face_ids, workdir, t0 = meta["pi"], meta["lab"], meta["face_ids"], meta["workdir"], meta["start"]

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
                out_path = find_optcuts_result(workdir)
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

def _assert_has_uv(mesh: trimesh.Trimesh):
    uv = getattr(mesh.visual, "uv", None)
    if uv is None:
        raise ValueError("mesh has no per-vertex UVs; run bake_uvs_with_vertex_splits first")
    return uv

def pack_with_xapack_cli(
    merged_mesh: trimesh.Trimesh,
    *,
    exe: str | Path,              # path to your compiled xapack binary
    resolution: int = 2048,
    padding_px: int = 8,
    rotate: bool = True,
    brute: bool = False,          # xapack optional last arg (if you enabled it)
    timeout_sec: float = 120.0,
    keep_artifacts: bool = False,
) -> trimesh.Trimesh:
    """
    Export `merged_mesh` (already has final islands/UVs), call xapack (pack-only),
    re-import the packed OBJ, and return it as a new trimesh. Raises on failure.
    """
    _assert_has_uv(merged_mesh)

    exe_path = Path(exe).expanduser().resolve()
    if not exe_path.exists():
        raise FileNotFoundError(f"xapack binary not found: {exe_path}")

    tmp = tempfile.TemporaryDirectory(prefix="xapack_")
    tdir = Path(tmp.name)
    in_obj  = tdir / "in_with_uv.obj"
    out_obj = tdir / "out_packed.obj"

    # Write the mesh as OBJ (triangles, per-vertex UVs)
    merged_mesh.export(in_obj, file_type="obj")

    # xapack usage: xapack in.obj out.obj <resolution> <padding_px> <rotate0|1> [texelsPerUnit] [brute0|1]
    # We pass tpu=0 to keep our existing density (you’re already doing uniform scaling upstream).
    cmd = [
        str(exe_path),
        str(in_obj),
        str(out_obj),
        str(int(resolution)),
        str(int(padding_px)),
        "1" if rotate else "0",
        "0",                                 # texelsPerUnit = 0 (preserve scale)
    ]
    if brute:
        cmd.append("1")

    try:
        proc = subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired:
        tmp.cleanup()
        raise TimeoutError(f"xapack timed out after {timeout_sec:.1f}s")

    if proc.returncode != 0 or not out_obj.exists():
        err = proc.stderr.decode(errors="ignore")
        tmp.cleanup()
        raise RuntimeError(f"xapack failed (exit {proc.returncode}). Stderr:\n{err}")

    packed = trimesh.load_mesh(out_obj, process=False)

    if keep_artifacts:
        # keep the OBJ beside your script for inspection
        saved = Path.cwd() / out_obj.name
        shutil.copy2(out_obj, saved)
        print(f"[xapack] saved packed OBJ -> {saved}")

    tmp.cleanup()
    return packed

def bake_uvs_with_vertex_splits(original_mesh: trimesh.Trimesh,
                                uv_face_corner: np.ndarray,
                                *,
                                round_uv: int = 7) -> trimesh.Trimesh:
    F = original_mesh.faces.astype(np.int64)
    V = original_mesh.vertices
    F_count = F.shape[0]

    index_map = {}  # (orig_vid, ru, rv) -> new_vid
    verts_out = []
    uvs_out = []
    faces_out = np.empty_like(F)

    for i in range(F_count):
        for j in range(3):
            vid = int(F[i, j])
            uv = uv_face_corner[i, j]
            if not np.isfinite(uv).all():
                uv = np.array([0.0, 0.0], dtype=np.float64)  # fallback if anything missed
            key = (vid, round(float(uv[0]), round_uv), round(float(uv[1]), round_uv))
            if key not in index_map:
                index_map[key] = len(verts_out)
                verts_out.append(V[vid].tolist())
                uvs_out.append([float(uv[0]), float(uv[1])])
            faces_out[i, j] = index_map[key]

    mesh_out = trimesh.Trimesh(vertices=np.asarray(verts_out, dtype=np.float64),
                               faces=faces_out.astype(np.int64),
                               process=False)
    mesh_out.visual = trimesh.visual.TextureVisuals(uv=np.asarray(uvs_out, dtype=np.float64))
    return mesh_out

def main():
    ap = argparse.ArgumentParser(description="EXACT Shapira08 SDF+GMM+k-way graph cut (per connected component)")
    ap.add_argument("input", help="path to mesh (.obj/.ply/...)")
    ap.add_argument("--output", default="out", help="output directory")
    ap.add_argument("--exe", "-e", default="build/OptCuts_bin", help="Path to OptCuts binary")
    ap.add_argument("--packer_exe", "-pe", default="ext/xatlas/build/xapack", help="Path to packer")
    args = ap.parse_args()
    
    mesh = trimesh.load_mesh(args.input, process=False)
    mesh.merge_vertices(merge_tex=True, merge_norm=True)

    labels0, prob = segment_per_components(mesh)
    
    # Make unaries (optional but best):
    if prob is not None:
        unary = -np.log(np.clip(prob, 1e-9, 1.0))
    else:
        unary = None

    labels = refine_labels_length_potts(
        mesh, labels0,
        unary_neglog=unary, lam=12.0, beta_sharp=1.5, stick=2.0
    )
    
    export_label_components(mesh, labels, args.output)
    
    uv_face_corner = run_optcuts_and_project(mesh, labels, exe=args.exe, timeout_sec=60.0)
    merged = bake_uvs_with_vertex_splits(mesh, uv_face_corner)
    
    packed = pack_with_xapack_cli(merged, exe=resolve_absolute_path(args.packer_exe))
    
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (Path(args.input).stem + "_with_uv.obj")
    packed.export(out_path, file_type="obj")
    print(f"[merge] Wrote packed OBJ with UVs -> {out_path}")


if __name__ == "__main__":
    main()