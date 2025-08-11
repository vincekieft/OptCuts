#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path
from time import sleep

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", default="input", help="Folder with meshes")
    parser.add_argument("--exe", "-e", default="build/OptCuts_bin", help="Path to OptCuts binary")
    parser.add_argument("--jobs", "-j", type=int,
                        default=8,
                        help="Max concurrent OptCuts processes")
    args = parser.parse_args()

    mesh_dir = Path(args.input)
    exe = Path(args.exe)

    files = [p for p in mesh_dir.iterdir() if p.is_file()]
    if not files:
        print(f"No files found in {mesh_dir}")
        return

    print(f"Running {len(files)} jobs with up to {args.jobs} concurrent processes…")

    procs = []   # list[(Popen, Path)]
    idx = 0

    def start_one(path: Path):
        cmd = [str(exe), "100", str(path), "0.999", "1", "0", "4.1", "1", "0"]
        return subprocess.Popen(cmd)

    # Prime the pool
    while idx < len(files) and len(procs) < args.jobs:
        f = files[idx]
        procs.append((start_one(f), f))
        idx += 1

    # Refill as processes finish
    exits_ok = 0
    exits_bad = 0
    while procs:
        still_running = []
        for p, f in procs:
            rc = p.poll()
            if rc is None:
                still_running.append((p, f))
                continue
            if rc == 0:
                exits_ok += 1
                print(f"✔ {f.name}")
            else:
                exits_bad += 1
                print(f"✖ {f.name} (exit {rc})")
            # start next if available
            if idx < len(files):
                nf = files[idx]
                still_running.append((start_one(nf), nf))
                idx += 1
        procs = still_running
        if procs:
            sleep(0.05)

    print(f"Done. OK: {exits_ok}, Failed: {exits_bad}")

if __name__ == "__main__":
    main()
