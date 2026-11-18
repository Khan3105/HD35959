#!/usr/bin/env python3
"""
run_storm_pms_model.py -- Run StORM on every PMS profile of a SINGLE
non-rotating MESA model, once per imposed constant rotation rate.

MESA injects rotation at the ZAMS, so PMS models carry Omega = 0. Here the
rotation is imposed in StORM with `set-rotation-constant`, and the P2
deformation is computed for that same rate (`deform --surface-rotation`
reads it back from the model).

Rotation labels follow the MS grid (MESA omega = Omega/Omega_crit at ZAMS).
MESA measures Omega_crit against the rotating star's equatorial radius, so
the label is NOT Omega in units of sqrt(GM/R^3). LABEL_TO_DYNAMICAL holds
the measured conversion: the mean Omega/sqrt(GM/R^3) over every batch3 MS
history row (M2.4-2.6, all Z and f_ov) where surf_avg_omega_div_omega_crit
equals the label to within 0.002. Spread within each label is <= 2%.

Scans, families and output properties are the same as run_storm_model.py.

Configuration comes from the environment (set by submit_STORM_pms.slurm):
    STORM_BIN        path to the storm binary
    PROFILE_ROOT     extracted PMS grid (contains <model>/LOGS/...)
    OUTPUT_ROOT      where results are written (creates <model>/...)
    STORM_PMS_LABELS rotation labels to run, comma separated
                     (default: all keys of LABEL_TO_DYNAMICAL)
    STORM_FREQ_MIN / STORM_FREQ_MAX / STORM_N_SCAN   scan window (4 / 38 / 850)
    STORM_FAMILIES   parity families (default even)
    STORM_PROFILES_OUTPUT  mode profiles to write (default as run_storm_model.py;
                     set to an empty string to write none)
    STORM_TIMEOUT    seconds per StORM run (default 900)

Output per model:
    <OUTPUT_ROOT>/<model>/profile<N>_rot<label>_<family>.hdf5
    <OUTPUT_ROOT>/<model>/run_log.csv

Usage:
    ./run_storm_pms_model.py --model M3.5_Z0.014_omega0.00_fov0.017
"""

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

from run_storm_model import (DEFAULT_PROFILES, DEFAULT_PROPERTIES, FAMILIES,
                             M_VALUES, run_one)

# MS-grid label (MESA Omega/Omega_crit) -> Omega / sqrt(GM/R^3).
LABEL_TO_DYNAMICAL = {
    "0.00": 0.0,
    "0.05": 0.0495,
    "0.15": 0.1479,
    "0.25": 0.2443,
    "0.35": 0.3374,
    "0.45": 0.4257,
    "0.55": 0.5081,
    "0.65": 0.5829,
    "0.75": 0.6482,
}

G_CGS = 6.67430e-8
SECONDS_PER_DAY = 86400.0
MODEL_PROPERTIES = "dynamical-frequency,deformation-rotation-frequency"


def env(name, default=None, required=False):
    val = os.environ.get(name, default)
    if required and val is None:
        sys.exit(f"ERROR: required environment variable {name} is not set")
    return val


def dynamical_frequency_cd(gyre_file):
    """sqrt(GM/R^3) in cycles/day, from the .data.GYRE header (n M R L ver)."""
    with open(gyre_file) as fh:
        parts = fh.readline().replace("D", "E").split()
    M, R = float(parts[1]), float(parts[2])
    return math.sqrt(G_CGS * M / R ** 3) * SECONDS_PER_DAY / (2.0 * math.pi)


def build_commands(gyre_file, omega_dyn, out_files, cfg):
    """StORM script: impose constant rotation, deform, scan, perturb, output."""
    lines = [f"input --format gyre {gyre_file}",
             f"set-rotation-constant --frequency-units=dynamical {omega_dyn}",
             "deform --surface-rotation",
             ""]
    for family in cfg["families"]:
        degrees = FAMILIES[family]
        for l in degrees:
            for m in M_VALUES:
                if abs(m) <= l:
                    lines.append(f"scan --frequency-units=cycles-per-day "
                                 f"{l} {m} {cfg['fmin']} {cfg['fmax']} "
                                 f"{cfg['nscan']}")
            lines.append("")
        lines.append("post-process")
        lines.append("")
        for m in M_VALUES:
            if any(abs(m) <= l for l in degrees):
                lines.append(f"perturb-deformed {m}")
        lines.append("")
        output = (f"output {out_files[family]} "
                  f"--frequency-units cycles-per-day "
                  f"--properties {cfg['properties']} ")
        # Mode profiles (xi_r, xi_h, ...) are ~90% of each file; an empty
        # STORM_PROFILES_OUTPUT leaves them out.
        if cfg["profiles"]:
            output += f"--profiles {cfg['profiles']} "
        lines.append(output + f"--model-properties {MODEL_PROPERTIES}")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=None,
                    help="only run the first N profiles (for testing)")
    ap.add_argument("--profile", type=int, default=None,
                    help="only run this profile number")
    args = ap.parse_args()

    storm_bin = Path(env("STORM_BIN", required=True))
    profile_root = Path(env("PROFILE_ROOT", required=True))
    output_root = Path(env("OUTPUT_ROOT", required=True))
    timeout = float(env("STORM_TIMEOUT", "900"))

    labels = [s.strip() for s in
              env("STORM_PMS_LABELS", ",".join(LABEL_TO_DYNAMICAL)).split(",")
              if s.strip()]
    for lab in labels:
        if lab not in LABEL_TO_DYNAMICAL:
            sys.exit(f"ERROR: unknown rotation label '{lab}'; choose from "
                     f"{', '.join(LABEL_TO_DYNAMICAL)}")

    cfg = {
        "fmin": env("STORM_FREQ_MIN", "4"),
        "fmax": env("STORM_FREQ_MAX", "38"),
        "nscan": env("STORM_N_SCAN", "850"),
        "properties": env("STORM_PROPERTIES", DEFAULT_PROPERTIES),
        "profiles": env("STORM_PROFILES_OUTPUT", DEFAULT_PROFILES),
        "families": [f.strip() for f in env("STORM_FAMILIES", "even").split(",")
                     if f.strip()],
    }
    for f in cfg["families"]:
        if f not in FAMILIES:
            sys.exit(f"ERROR: unknown family '{f}' in STORM_FAMILIES; "
                     f"choose from {', '.join(FAMILIES)}")

    if not storm_bin.exists():
        sys.exit(f"ERROR: storm binary not found at {storm_bin}")
    logs = profile_root / args.model / "LOGS"
    if not logs.is_dir():
        sys.exit(f"ERROR: no LOGS directory for {args.model} under {profile_root}")

    def profile_number(p):
        return int(p.name.replace("profile", "").replace(".data.GYRE", ""))

    # The extracted PMS grid holds only the in-box profiles, so run them all.
    gyre_files = sorted(logs.glob("profile*.data.GYRE"), key=profile_number)
    if not gyre_files:
        sys.exit(f"ERROR: no profile*.data.GYRE files in {logs}")
    if args.profile is not None:
        gyre_files = [g for g in gyre_files if profile_number(g) == args.profile]
        if not gyre_files:
            sys.exit(f"ERROR: no profile{args.profile}.data.GYRE in {logs}")
    if args.limit:
        gyre_files = gyre_files[:args.limit]

    model_out = output_root / args.model
    model_out.mkdir(parents=True, exist_ok=True)

    storm_env = os.environ.copy()
    storm_env.setdefault("RAYON_NUM_THREADS",
                         os.environ.get("SLURM_CPUS_PER_TASK", "1"))

    tally = {}
    durations = []
    t_start = time.time()

    with open(model_out / "run_log.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["profile", "rot_label", "omega_dynamical",
                         "omega_cd", "status", "seconds", "message"])

        for gf in gyre_files:
            pnum = profile_number(gf)
            dyn_cd = dynamical_frequency_cd(gf)

            for lab in labels:
                om = LABEL_TO_DYNAMICAL[lab]
                out_files = {fam: model_out / f"profile{pnum}_rot{lab}_{fam}.hdf5"
                             for fam in cfg["families"]}
                script = build_commands(gf, om, out_files, cfg)
                status, dt, msg = run_one(storm_bin, script, out_files,
                                          timeout, storm_env)
                tally[status] = tally.get(status, 0) + 1
                if status == "completed":
                    durations.append(dt)
                writer.writerow([pnum, lab, f"{om:.4f}", f"{om * dyn_cd:.6f}",
                                 status, f"{dt:.2f}", msg])

                if status in ("failed", "timeout", "error", "no_output"):
                    print(f"  {status.upper()}: profile{pnum} rot{lab}: "
                          f"{msg[:150]}", file=sys.stderr)
                    (model_out / f"profile{pnum}_rot{lab}.storm.failed"
                     ).write_text(script)

    elapsed = time.time() - t_start
    n_runs = len(gyre_files) * len(labels)
    print(f"[{args.model}] {len(gyre_files)} profiles x {len(labels)} "
          f"rotations = {n_runs} runs in {elapsed/60:.1f} min")
    print("  " + " ".join(f"{k}={v}" for k, v in sorted(tally.items())))
    if durations:
        print(f"  mean {sum(durations)/len(durations):.2f} s/run, "
              f"max {max(durations):.2f} s")

    if any(k in tally for k in ("failed", "timeout", "error", "no_output")):
        sys.exit(1)


if __name__ == "__main__":
    main()
