#!/usr/bin/env python3
"""
HD 35929 — GYRE Grid Runner
Runs GYRE (g-modes + p-modes) on all MESA GYRE profile files.
Mirrors the MESA grid directory structure.
"""

import os, sys, subprocess, argparse, csv
from pathlib import Path
from glob import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

# =====================================================================
# CONFIGURATION
# =====================================================================
GYRE_DIR       = Path(os.path.expanduser("~/mesa-24.08.1/gyre-8.1"))
GYRE_BIN       = GYRE_DIR / "bin" / "gyre"
MESA_GRID_DIR  = Path("/STER/kianTedeskino/HD35929/mesa_grid/grid_runs")
GYRE_GRID_DIR  = Path("/STER/kianTedeskino/HD35929/gyre_grid")
TEMPLATE_DIR   = Path(os.path.expanduser("~/ConnyProj/MESA_grid/templates"))
CSV_FILE       = GYRE_GRID_DIR / "gyre_progress.csv"

# GYRE needs these environment variables
GYRE_ENV = os.environ.copy()
GYRE_ENV["GYRE_DIR"] = str(GYRE_DIR)

# =====================================================================
# CORE FUNCTIONS
# =====================================================================
def find_all_gyre_files(pms_every_n=10):
    """Find GYRE files, keeping every Nth PMS profile but all MS profiles."""
    all_files = []
    
    for model_dir in sorted(MESA_GRID_DIR.glob("M*")):
        gyre_files = sorted(model_dir.glob("LOGS/profile*.data.GYRE"),
                           key=lambda f: int(f.name.replace("profile","").replace(".data.GYRE","")))
        
        # Read history to find ZAMS model number
        history_file = model_dir / "LOGS" / "history.data"
        zams_profile = 0
        if history_file.exists():
            try:
                with open(history_file) as hf:
                    lines = hf.readlines()
                cols = lines[5].split()
                lnuc_idx = cols.index('log_Lnuc')
                logl_idx = cols.index('log_L')
                model_idx = cols.index('model_number')
                for line in lines[6:]:
                    vals = line.split()
                    if float(vals[lnuc_idx]) - float(vals[logl_idx]) > -0.004:  # Lnuc/L > 0.99
                        zams_profile = len(all_files)  # approximate
                        break
            except Exception:
                pass
        
        # Keep every Nth PMS profile, all MS profiles
        for i, gf in enumerate(gyre_files):
            if i < len(gyre_files) * 0.7:  # rough PMS cutoff (~70% of profiles)
                if i % pms_every_n == 0:
                    all_files.append(gf)
            else:
                all_files.append(gf)
    
    print(f"Selected {len(all_files)} GYRE profiles (PMS every {pms_every_n}th, all MS)")
    return all_files


def get_model_name(gyre_file):
    """Extract model directory name from GYRE file path."""
    # e.g. .../grid_runs/M2.0_Z0.014_omega0.25_fov0.017/LOGS/profile5.data.GYRE
    return gyre_file.parent.parent.name


def get_profile_id(gyre_file):
    """Extract profile number from GYRE file name."""
    # e.g. profile5.data.GYRE → 5
    fname = gyre_file.name  # profile5.data.GYRE
    num = fname.replace("profile", "").replace(".data.GYRE", "")
    return num


def run_gyre_on_profile(gyre_file, mode_type="g"):
    """Run GYRE on a single profile. Returns (model, profile, mode, status, msg)."""
    model_name = get_model_name(gyre_file)
    profile_id = get_profile_id(gyre_file)
    
    # Create output directory mirroring MESA structure
    output_dir = GYRE_GRID_DIR / model_name / f"profile{profile_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Choose template
    if mode_type == "g":
        template_file = TEMPLATE_DIR / "gyre_ad_g-modes.in"
        summary_file = output_dir / "summary_g.txt"
    else:
        template_file = TEMPLATE_DIR / "gyre_ad_p-modes.in"
        summary_file = output_dir / "summary_p.txt"

    # Skip if already done
    if summary_file.exists() and summary_file.stat().st_size > 0:
        return (model_name, profile_id, mode_type, "already_done", "")

    # Read template and substitute
    template = template_file.read_text()
    inlist = template.replace("__GYRE_FILE__", str(gyre_file))
    inlist = inlist.replace("__OUTPUT_DIR__", str(output_dir))

    # Write temporary inlist in output dir
    inlist_path = output_dir / f"gyre_{mode_type}.in"
    inlist_path.write_text(inlist)

    # Run GYRE
    try:
        result = subprocess.run(
            [str(GYRE_BIN), str(inlist_path)],
            cwd=str(output_dir),
            capture_output=True,
            text=True,
            timeout=60,  # 1h timeout per run
            env=GYRE_ENV
        )
        if result.returncode == 0 and summary_file.exists():
            return (model_name, profile_id, mode_type, "completed", "")
        else:
            err_msg = result.stderr[-500:] if result.stderr else "no stderr"
            return (model_name, profile_id, mode_type, "failed", err_msg)
    except subprocess.TimeoutExpired:
        return (model_name, profile_id, mode_type, "timeout", "1h timeout")
    except Exception as e:
        return (model_name, profile_id, mode_type, "error", str(e))


def run_both_modes(gyre_file):
    """Run both g-mode and p-mode GYRE on a single profile."""
    results = []
    for mode in ["g", "p"]:
        results.append(run_gyre_on_profile(gyre_file, mode_type=mode))
    return results


# =====================================================================
# GRID EXECUTION
# =====================================================================
def run_grid(n_parallel=1):
    """Run GYRE on all profiles."""
    gyre_files = find_all_gyre_files()
    
    if not gyre_files:
        print("No GYRE files found. Check MESA_GRID_DIR path.")
        sys.exit(1)

    completed = 0
    failed = 0
    skipped = 0
    total = len(gyre_files) * 2  # g + p modes

    print(f"Running GYRE on {len(gyre_files)} profiles (g + p = {total} runs)")
    print(f"Parallel workers: {n_parallel}")
    print(f"Output: {GYRE_GRID_DIR}")
    print()

    if n_parallel == 1:
        for i, gf in enumerate(gyre_files):
            results = run_both_modes(gf)
            for model, prof, mode, status, msg in results:
                if status == "completed":
                    completed += 1
                elif status == "already_done":
                    skipped += 1
                else:
                    failed += 1
                    print(f"  FAIL: {model}/profile{prof} {mode}-modes: {msg[:80]}")
            if (i + 1) % 100 == 0:
                print(f"  Progress: {i+1}/{len(gyre_files)} profiles "
                      f"({completed} done, {skipped} skipped, {failed} failed)")
    else:
        with ProcessPoolExecutor(max_workers=n_parallel) as executor:
            futures = {executor.submit(run_both_modes, gf): gf for gf in gyre_files}
            for i, future in enumerate(as_completed(futures)):
                results = future.result()
                for model, prof, mode, status, msg in results:
                    if status == "completed":
                        completed += 1
                    elif status == "already_done":
                        skipped += 1
                    else:
                        failed += 1
                        print(f"  FAIL: {model}/profile{prof} {mode}: {msg[:80]}")
                if (i + 1) % 500 == 0:
                    print(f"  Progress: {i+1}/{len(gyre_files)} profiles "
                          f"({completed} done, {skipped} skipped, {failed} failed)")

    print(f"\nDone: {completed} completed, {skipped} skipped, {failed} failed (of {total})")


def count_grid():
    """Count the grid size."""
    gyre_files = find_all_gyre_files()
    models = set(get_model_name(gf) for gf in gyre_files)
    print(f"MESA models with GYRE files: {len(models)}")
    print(f"Total GYRE profiles: {len(gyre_files)}")
    print(f"Total GYRE runs (g + p): {len(gyre_files) * 2}")
    
    # Estimate runtime
    est_seconds = len(gyre_files) * 2 * 30  # ~30s per run
    print(f"Estimated runtime (serial): {est_seconds/3600:.0f} hours")
    print(f"Estimated runtime (4 parallel): {est_seconds/3600/4:.0f} hours")


def check_progress():
    """Check how many GYRE runs are complete."""
    g_done = len(list(GYRE_GRID_DIR.glob("M*/profile*/summary_g.txt")))
    p_done = len(list(GYRE_GRID_DIR.glob("M*/profile*/summary_p.txt")))
    total_profiles = len(find_all_gyre_files())
    print(f"g-mode summaries: {g_done}/{total_profiles}")
    print(f"p-mode summaries: {p_done}/{total_profiles}")
    print(f"Total: {g_done + p_done}/{total_profiles * 2}")


# =====================================================================
# MAIN
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HD 35929 GYRE Grid")
    parser.add_argument("--run", action="store_true", help="Run GYRE grid")
    parser.add_argument("--parallel", type=int, default=1, help="Parallel workers")
    parser.add_argument("--count", action="store_true", help="Count grid size")
    parser.add_argument("--progress", action="store_true", help="Check progress")
    args = parser.parse_args()

    # Verify GYRE binary
    if not GYRE_BIN.exists():
        print(f"ERROR: GYRE binary not found at {GYRE_BIN}")
        print("Check GYRE_DIR path and try: ls $GYRE_DIR/bin/")
        sys.exit(1)

    if args.count:
        count_grid()
    elif args.progress:
        check_progress()
    elif args.run:
        run_grid(args.parallel)
    else:
        parser.print_help()
