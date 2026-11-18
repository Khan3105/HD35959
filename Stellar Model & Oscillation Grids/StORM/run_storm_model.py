#!/usr/bin/env python3
"""
run_storm_model.py -- Run StORM on every main-sequence profile of a SINGLE
MESA model. This is the body of one worker-ng work item.

For each profile it generates a StORM command script and pipes it to the
binary. The even-parity family (l = 0, 2, 4) is scanned so that the m = -2..2
quintuplet can be extracted from the deformed, perturbed modes.

Which (l, m) pairs are scanned follows from the physics: rotation does not
couple different azimuthal orders, so for each target m only the degrees in
the parity family with l >= |m| are needed.

Configuration comes from the environment (set by submit_STORM.slurm):
    STORM_BIN        path to the storm binary
    PROFILE_ROOT     where the MESA batch was extracted (<model>/LOGS/...)
    OUTPUT_ROOT      where results are written    (creates <model>/...)
    STORM_PROFILES   CSV from make_storm_worklist.py (model, profile, v_rot_kms)
    STORM_FREQ_MIN   scan lower bound, cycles/day        (default 7)
    STORM_FREQ_MAX   scan upper bound, cycles/day        (default 28)
    STORM_N_SCAN     scan steps                          (default 500)
    STORM_TIMEOUT    seconds per profile                 (default 900)
    STORM_DISABLE_SYMMETRIC  '1' to pass --disable-symmetric to deform
    STORM_SET_ROTATION       '1' to emit set-rotation-constant (default 1)
    STORM_PROPERTIES         output --properties list
    STORM_PROFILES_OUTPUT    output --profiles list

Usage:
    ./run_storm_model.py --model M2.3_Z0.006_omega0.05_fov0.005
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

# The P2 deformation couples l with l+-2, so even and odd degrees form two
# separate, non-communicating families. They must be scanned, post-processed
# and perturbed independently: putting all degrees into one post-process
# would diagonalise a matrix mixing parities, which is not physical.
FAMILIES = {
    "even": (0, 2, 4),
    "odd": (1, 3),
}
# Azimuthal orders wanted in the output (the quintuplet). Within a family
# only degrees with l >= |m| contribute at a given m.
M_VALUES = (0, 1, 2, -1, -2)

DEFAULT_PROPERTIES = ("frequency,radial-order,degree,azimuthal-order,"
                      "deformed-frequency,deformed-eigenvector")

DEFAULT_PROFILES = ("radial-coordinate,xi_r,xi_h")


def env(name, default=None, required=False):
    val = os.environ.get(name, default)
    if required and val is None:
        sys.exit(f"ERROR: required environment variable {name} is not set")
    return val


def build_commands(gyre_file, v_rot, out_files, cfg):
    """
    Return the StORM command script for one profile.

    out_files maps family name -> output path. The model is loaded and
    deformed once; each family is then scanned, post-processed, perturbed
    and written. `output` clears the mode computations but leaves the
    deformation in place, so the second family reuses it.
    """
    lines = [f"input --format gyre {gyre_file}"]

    # NOTE: set-rotation-constant is fatal on a model that already carries
    # rotation ("would result in an inconsistent model"), which is every
    # model in this grid. It is off by default and kept only as an escape
    # hatch for non-rotating models.
    if cfg["set_rotation"]:
        lines.append("set-rotation-constant "
                     f"--frequency-units=equatorial-velocity {v_rot}")
    lines.append("")

    # Newer StORM can read the surface rotation rate straight from the model,
    # so no value need be passed. The spherically symmetric component of the
    # deformation is disabled automatically when the model is rotating.
    if cfg["surface_rotation"]:
        deform = "deform --surface-rotation"
    else:
        deform = ("deform --frequency-units=equatorial-velocity "
                  f"--rotation={v_rot}")
    if cfg["disable_symmetric"]:
        deform += " --disable-symmetric"
    lines.append(deform)
    lines.append("")

    for family in cfg["families"]:
        degrees = FAMILIES[family]
        ms_in_family = [m for m in M_VALUES
                        if any(abs(m) <= l for l in degrees)]

        for l in degrees:
            for m in M_VALUES:
                if abs(m) > l:
                    continue
                lines.append(f"scan --frequency-units=cycles-per-day "
                             f"{l} {m} {cfg['fmin']} {cfg['fmax']} "
                             f"{cfg['nscan']}")
            lines.append("")

        lines.append("post-process")
        lines.append("")
        for m in ms_in_family:
            lines.append(f"perturb-deformed {m}")
        lines.append("")
        lines.append(f"output {out_files[family]} "
                     f"--frequency-units cycles-per-day "
                     f"--properties {cfg['properties']} "
                     f"--profiles {cfg['profiles']}")
        lines.append("")

    return "\n".join(lines)


def run_one(storm_bin, script, out_files, timeout, storm_env):
    """Run StORM once. Returns (status, seconds, message)."""
    if all(p.exists() and p.stat().st_size > 0 for p in out_files.values()):
        return ("already_done", 0.0, "")

    t0 = time.time()
    try:
        res = subprocess.run(
            [str(storm_bin)], input=script, capture_output=True, text=True,
            timeout=timeout, env=storm_env,
            cwd=str(next(iter(out_files.values())).parent),
        )
    except subprocess.TimeoutExpired:
        return ("timeout", time.time() - t0, f"exceeded {timeout}s")
    except Exception as exc:
        return ("error", time.time() - t0, repr(exc))

    dt = time.time() - t0

    if res.returncode != 0:
        msg = (res.stderr or res.stdout or "no output")[-300:].replace("\n", " ")
        # rc=101 is a Rust panic. Near critical rotation the P2 deformation
        # drives the discriminant non-finite and StORM aborts
        # ("assertion failed: next.f.is_finite()"). That is a domain limit of
        # the perturbative treatment, not a bug in the setup, so it is
        # recorded separately and does not fail the work item.
        if res.returncode == 101 and "is_finite" in msg:
            return ("diverged", dt, "deformation diverged (near-critical rotation)")
        return ("failed", dt, f"rc={res.returncode} {msg}")

    missing = [k for k, p in out_files.items()
               if not (p.exists() and p.stat().st_size > 0)]
    if missing:
        # Clean exit but nothing written: usually every scan found 0 modes.
        found = [ln for ln in (res.stdout + res.stderr).splitlines()
                 if "Found" in ln]
        return ("no_output", dt,
                f"clean exit, no {'/'.join(missing)} file; "
                f"{len(found)} scan lines")

    # Count modes for the log -- cheap and very useful for spotting a scan
    # window that is too narrow. StORM prints "Found N modes" on stderr.
    n_modes = sum(int(ln.split()[-2]) for ln in
                  (res.stdout + res.stderr).splitlines()
                  if ln.strip().startswith("Found"))
    return ("completed", dt, f"{n_modes} modes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=None,
                    help="only run the first N profiles (for testing)")
    args = ap.parse_args()

    storm_bin = Path(env("STORM_BIN", required=True))
    profile_root = Path(env("PROFILE_ROOT", required=True))
    output_root = Path(env("OUTPUT_ROOT", required=True))
    profiles_csv = Path(env("STORM_PROFILES", required=True))

    cfg = {
        "fmin": env("STORM_FREQ_MIN", "4"),
        "fmax": env("STORM_FREQ_MAX", "38"),
        "nscan": env("STORM_N_SCAN", "850"),
        "properties": env("STORM_PROPERTIES", DEFAULT_PROPERTIES),
        "profiles": env("STORM_PROFILES_OUTPUT", DEFAULT_PROFILES),
        "disable_symmetric": env("STORM_DISABLE_SYMMETRIC", "0") == "1",
        # Default OFF: fatal on models that already carry rotation.
        "set_rotation": env("STORM_SET_ROTATION", "0") == "1",
        # Default ON: let StORM read the surface rate from the model itself.
        "surface_rotation": env("STORM_SURFACE_ROTATION", "1") == "1",
        # Which parity families to compute, comma separated.
        "families": [f.strip() for f in
                     env("STORM_FAMILIES", "even,odd").split(",")
                     if f.strip()],
    }
    for f in cfg["families"]:
        if f not in FAMILIES:
            sys.exit(f"ERROR: unknown family '{f}' in STORM_FAMILIES; "
                     f"choose from {', '.join(FAMILIES)}")
    timeout = float(env("STORM_TIMEOUT", "900"))

    # StORM's P2 deformation is perturbative in the rotation rate and starts
    # to diverge as omega/omega_crit approaches 1. Profiles above this ratio
    # are skipped rather than run, so the log distinguishes "outside the
    # method's domain" from "genuinely failed". Set to 0 to disable.
    max_ocrit = float(env("STORM_MAX_OMEGA_CRIT", "0"))

    if not storm_bin.exists():
        sys.exit(f"ERROR: storm binary not found at {storm_bin}")
    if not profiles_csv.exists():
        sys.exit(f"ERROR: profile table not found at {profiles_csv}")

    # Rows for this model only.
    rows = []
    with open(profiles_csv) as fh:
        for r in csv.DictReader(fh):
            if r["model"] == args.model:
                rows.append(r)
    if not rows:
        sys.exit(f"ERROR: no rows for {args.model} in {profiles_csv}")
    rows.sort(key=lambda r: int(r["profile"]))
    if args.limit:
        rows = rows[:args.limit]

    logs = profile_root / args.model / "LOGS"
    if not logs.is_dir():
        sys.exit(f"ERROR: no LOGS directory for {args.model} under {profile_root}")

    model_out = output_root / args.model
    model_out.mkdir(parents=True, exist_ok=True)

    # StORM is multithreaded via rayon; pin it so one work item uses one core
    # unless the job explicitly asks for more.
    storm_env = os.environ.copy()
    storm_env.setdefault("RAYON_NUM_THREADS",
                         os.environ.get("SLURM_CPUS_PER_TASK", "1"))

    tally = {"completed": 0, "already_done": 0, "no_output": 0,
             "too_fast": 0, "diverged": 0,
             "failed": 0, "timeout": 0, "error": 0}
    durations = []
    t_start = time.time()

    with open(model_out / "run_log.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["profile", "v_rot_kms", "omega_div_omega_crit",
                         "status", "seconds", "message"])

        for r in rows:
            pnum = int(r["profile"])
            v_rot = r["v_rot_kms"]
            ocrit = r.get("omega_div_omega_crit", "")

            # Above the ceiling the perturbative deformation is unreliable,
            # so skip rather than run and record why.
            if max_ocrit > 0 and ocrit not in ("", None):
                try:
                    if float(ocrit) > max_ocrit:
                        tally["too_fast"] += 1
                        writer.writerow([pnum, v_rot, ocrit, "too_fast", "0.00",
                                         f"omega/omega_crit above {max_ocrit}"])
                        continue
                except ValueError:
                    pass

            gyre_file = logs / f"profile{pnum}.data.GYRE"
            out_files = {fam: model_out / f"profile{pnum}_{fam}.hdf5"
                         for fam in cfg["families"]}

            if not gyre_file.exists():
                writer.writerow([pnum, v_rot, ocrit, "missing_input", "0",
                                 str(gyre_file)])
                tally["error"] += 1
                print(f"  MISSING: {gyre_file}", file=sys.stderr)
                continue

            script = build_commands(gyre_file, v_rot, out_files, cfg)
            status, dt, msg = run_one(storm_bin, script, out_files,
                                      timeout, storm_env)
            tally[status] = tally.get(status, 0) + 1
            if status == "completed":
                durations.append(dt)
            writer.writerow([pnum, v_rot, ocrit, status, f"{dt:.2f}", msg])

            if status in ("failed", "timeout", "error", "no_output"):
                print(f"  {status.upper()}: profile{pnum} "
                      f"(v_rot={v_rot}): {msg[:150]}", file=sys.stderr)
                # Keep the failing script for debugging.
                (model_out / f"profile{pnum}.storm.failed").write_text(script)

    elapsed = time.time() - t_start
    mean = sum(durations) / len(durations) if durations else 0.0

    print(f"[{args.model}] {len(rows)} profiles in {elapsed/60:.1f} min")
    print(f"  completed={tally['completed']} skipped={tally['already_done']} "
          f"too_fast={tally['too_fast']} diverged={tally['diverged']} "
          f"no_output={tally['no_output']} failed={tally['failed']} "
          f"timeout={tally['timeout']} error={tally['error']}")
    if durations:
        print(f"  mean {mean:.2f} s/profile, max {max(durations):.2f} s")

    if tally["failed"] or tally["timeout"] or tally["error"]:
        sys.exit(1)


if __name__ == "__main__":
    main()