#!/usr/bin/env python3
"""
make_storm_worklist.py -- Build the StORM input table for the HD 35929 grid.

Unlike the GYRE stage, StORM needs a per-profile rotation rate, so this writes
one row per PROFILE (not per model):

    model, profile, v_rot_kms, omega_div_omega_crit, model_number

Only MAIN-SEQUENCE profiles are kept: model_number >= ZAMS, where ZAMS is
located physically via center_h1 depletion (same criterion as the GYRE stage).

It also writes a worker-ng worklist with one row per MODEL, since a work item
is still "one model" -- the runner looks up its own profiles in this table.

Usage:
    ./make_storm_worklist.py --profile-root /path/to/extracted/batch2_... \\
                             --output storm_profiles_batch2.csv \\
                             --worklist worklist_storm_batch2.csv
"""

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


class ModelError(Exception):
    pass


# ---------------------------------------------------------------------
# PARSING
# ---------------------------------------------------------------------
def parse_profiles_index(path):
    """Return {profile_number: model_number}."""
    mapping = {}
    with open(path) as fh:
        lines = fh.readlines()
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 3 or line.strip().startswith("#"):
            continue
        try:
            mapping[int(parts[2])] = int(parts[0])
        except ValueError:
            continue
    if not mapping:
        raise ModelError(f"no usable rows in {path}")
    return mapping


def parse_history(path):
    """
    Return (zams_model, {model_number: (v_rot_kms, omega_div_omega_crit)}).

    ZAMS is the first model where center_h1 has fallen 0.002 below its
    initial value. Unlike the GYRE stage we cannot stop there, because the
    rotation rate is needed for every main-sequence model afterwards.
    """
    with open(path) as fh:
        for _ in range(5):
            if not fh.readline():
                raise ModelError(f"{path} truncated in header")
        name_line = fh.readline()
        if not name_line:
            raise ModelError(f"{path} has no column-name line")
        cols = name_line.split()

        def idx(name):
            return cols.index(name) if name in cols else None

        i_model = idx("model_number")
        i_xc = idx("center_h1")
        i_vrot = idx("surf_avg_v_rot")
        i_ocrit = idx("surf_avg_omega_div_omega_crit")

        missing = [n for n, i in (("model_number", i_model),
                                  ("center_h1", i_xc),
                                  ("surf_avg_v_rot", i_vrot)) if i is None]
        if missing:
            raise ModelError(f"{path} missing columns: {', '.join(missing)}")

        zams = None
        xc_initial = None
        rot = {}

        for line in fh:
            vals = line.split()
            if len(vals) <= i_model:
                continue
            try:
                mnum = int(float(vals[i_model]))
                xc = float(vals[i_xc])
                vrot = float(vals[i_vrot])
                ocrit = float(vals[i_ocrit]) if i_ocrit is not None else float("nan")
            except (ValueError, IndexError):
                continue

            if xc_initial is None:
                xc_initial = xc
            if zams is None and xc <= xc_initial - 0.002:
                zams = mnum
            rot[mnum] = (vrot, ocrit)

    if zams is None:
        raise ModelError(f"{path}: ZAMS criterion never satisfied")
    return zams, rot


# ---------------------------------------------------------------------
# PER-MODEL
# ---------------------------------------------------------------------
def collect_model(model_dir):
    model_dir = Path(model_dir)
    name = model_dir.name
    logs = model_dir / "LOGS"

    for f in ("profiles.index", "history.data"):
        if not (logs / f).is_file():
            raise ModelError(f"{name}: missing {f}")

    prof_to_model = parse_profiles_index(logs / "profiles.index")
    zams, rot = parse_history(logs / "history.data")

    on_disk = {}
    for gf in logs.glob("profile*.data.GYRE"):
        stem = gf.name.replace("profile", "").replace(".data.GYRE", "")
        try:
            on_disk[int(stem)] = gf.name
        except ValueError:
            continue
    if not on_disk:
        raise ModelError(f"{name}: no profile*.data.GYRE files")

    rows = []
    n_pms = 0
    missing_rot = 0
    for pnum in sorted(on_disk):
        mnum = prof_to_model.get(pnum)
        if mnum is None:
            continue
        if mnum < zams:
            n_pms += 1
            continue                       # pre-main-sequence: skip
        if mnum not in rot:
            missing_rot += 1
            continue
        vrot, ocrit = rot[mnum]
        rows.append({
            "model": name,
            "profile": pnum,
            "model_number": mnum,
            "v_rot_kms": f"{vrot:.6g}",
            "omega_div_omega_crit": f"{ocrit:.6g}",
        })

    if not rows:
        raise ModelError(f"{name}: no main-sequence profiles found "
                         f"(zams={zams}, {len(on_disk)} profiles on disk)")

    stats = {"model": name, "zams": zams, "n_ms": len(rows),
             "n_pms_skipped": n_pms, "n_missing_rot": missing_rot,
             "v_rot_min": min(float(r["v_rot_kms"]) for r in rows),
             "v_rot_max": max(float(r["v_rot_kms"]) for r in rows)}
    return rows, stats


def _worker(model_dir):
    try:
        rows, stats = collect_model(model_dir)
        return ("ok", model_dir, rows, stats)
    except ModelError as exc:
        return ("error", model_dir, str(exc), None)
    except Exception as exc:
        return ("error", model_dir, f"unexpected: {exc!r}", None)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile-root", required=True,
                    help="directory holding the extracted M* model directories")
    ap.add_argument("--output", required=True,
                    help="CSV of MS profiles with rotation rates")
    ap.add_argument("--worklist", required=True,
                    help="worker-ng CSV, one row per model")
    ap.add_argument("--nproc", type=int, default=4)
    args = ap.parse_args()

    root = Path(args.profile_root)
    if not root.is_dir():
        sys.exit(f"ERROR: profile root not found: {root}")

    for target in (args.output, args.worklist):
        Path(target).parent.mkdir(parents=True, exist_ok=True)

    model_dirs = sorted(d for d in root.glob("M*") if d.is_dir())
    if not model_dirs:
        sys.exit(f"ERROR: no M* directories in {root}")

    print(f"Scanning {len(model_dirs)} models with {args.nproc} workers...")

    all_rows, all_stats, errors = [], [], []
    with ProcessPoolExecutor(max_workers=args.nproc) as ex:
        futures = [ex.submit(_worker, d) for d in model_dirs]
        for i, fut in enumerate(as_completed(futures), 1):
            status, d, payload, stats = fut.result()
            if status == "ok":
                all_rows.extend(payload)
                all_stats.append(stats)
            else:
                errors.append((Path(d).name, payload))
            if i % 250 == 0:
                print(f"  {i}/{len(model_dirs)} ({len(errors)} errors)")

    all_rows.sort(key=lambda r: (r["model"], r["profile"]))

    with open(args.output, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "profile", "model_number",
                                           "v_rot_kms", "omega_div_omega_crit"])
        w.writeheader()
        w.writerows(all_rows)

    # Worker-ng worklist: one work item per model, longest first.
    all_stats.sort(key=lambda s: (-s["n_ms"], s["model"]))
    with open(args.worklist, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "n_profiles"])
        for s in all_stats:
            w.writerow([s["model"], s["n_ms"]])

    n_ms = sum(s["n_ms"] for s in all_stats)
    n_pms = sum(s["n_pms_skipped"] for s in all_stats)
    vmins = [s["v_rot_min"] for s in all_stats]
    vmaxs = [s["v_rot_max"] for s in all_stats]

    print()
    print("-" * 58)
    print(f"Models ok          : {len(all_stats)}")
    print(f"Models failed      : {len(errors)}")
    print(f"MS profiles kept   : {n_ms}")
    print(f"PMS profiles skipped: {n_pms}")
    print(f"MS per model       : min {min(s['n_ms'] for s in all_stats)}, "
          f"max {max(s['n_ms'] for s in all_stats)}")
    print(f"v_rot range (km/s) : {min(vmins):.2f} -> {max(vmaxs):.2f}")
    print(f"  -> {args.output}")
    print(f"  -> {args.worklist}")

    if errors:
        print()
        print(f"ERRORS ({len(errors)}):")
        for name, msg in errors[:20]:
            print(f"  {name}: {msg}")
        if len(errors) > 20:
            print(f"  ... and {len(errors)-20} more")
        sys.exit(1)


if __name__ == "__main__":
    main()
