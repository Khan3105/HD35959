#!/usr/bin/env python3
"""
HD 35929 — MESA Grid Runner (v2)
Single-stage: non-rotating PMS → auto-inject rotation at ZAMS → evolve MS to TAMS.
Uses near_zams_relax_omega for rotation injection (no two-stage pipeline needed).
Grid parameter: Omega/Omega_crit at ZAMS (Mombarg & Mathis 2026).
"""

import os, sys, csv, shutil, subprocess, itertools, argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# =====================================================================
# CONFIGURATION
# =====================================================================
MESA_DIR       = os.path.expandvars("$MESA_DIR")
MESA_WORK_DIR  = Path(os.path.expanduser("~/ConnyProj/MESA_grid/work_template/work"))
TEMPLATE_DIR   = Path(os.path.expanduser("~/ConnyProj/MESA_grid/templates"))
OUTPUT_BASE    = Path("/STER/kianTedeskino/HD35929/mesa_grid/grid_runs_1")
CSV_FILE       = Path("/STER/kianTedeskino/HD35929/mesa_grid/parameter_grid.csv")

# =====================================================================
# GRID PARAMETER SPACE
# =====================================================================
# Mass: 1.8 to 3.2 Msun in 0.1 steps (15 values)
M_INI_LIST = [round(1.8 + 0.1*i, 1) for i in range(15)]

# Metallicity: 0.006 to 0.022 in 0.004 steps (5 values)
Z_INI_LIST = [round(0.006 + 0.004*i, 3) for i in range(5)]

# Omega/Omega_crit at ZAMS (Mombarg & Mathis 2026 parameterisation)
OMEGA_FRAC_LIST = [0.00, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75]

# Exponential overshooting f_ov (5 values)
F_OV_LIST = [0.005, 0.010, 0.017, 0.025, 0.035]

# --- TEST SUBSET---
# M_INI_LIST = [2.0, 2.5, 3.0]
# Z_INI_LIST = [0.010, 0.014, 0.018]
# OMEGA_FRAC_LIST = [0.10, 0.30, 0.50]
# F_OV_LIST = [0.010, 0.017, 0.025]

# Helium enrichment law (Mombarg et al. 2024b, Brogaard+2012, Verma+2019)
def compute_composition(Z):
    Y = 0.28 + (Z - 0.0142) * 1.4
    X = 1.0 - Y - Z
    return round(X, 6), round(Y, 6)


# =====================================================================
# GRID SETUP
# =====================================================================
def make_run_name(M, Z, omega, fov):
    return f"M{M:.1f}_Z{Z:.3f}_omega{omega:.2f}_fov{fov:.3f}"


def setup_grid():
    """Create all run directories and the parameter CSV."""
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    # Read templates
    template_inlist_project = (TEMPLATE_DIR / 'inlist_project').read_text()
    template_inlist = (TEMPLATE_DIR / 'inlist').read_text()

    # Verify MESA work dir
    if not MESA_WORK_DIR.is_dir():
        print(f"ERROR: {MESA_WORK_DIR} not found")
        print(f"Create: cp -r {MESA_DIR}/star/work {MESA_WORK_DIR}")
        sys.exit(1)

    star_binary = MESA_WORK_DIR / 'star'
    if not star_binary.is_file():
        print(f"ERROR: MESA not compiled in {MESA_WORK_DIR}")
        print(f"Run: cd {MESA_WORK_DIR} && ./mk")
        sys.exit(1)

    combos = list(itertools.product(M_INI_LIST, Z_INI_LIST, OMEGA_FRAC_LIST, F_OV_LIST))
    print(f"Setting up {len(combos)} models...")

    with open(CSV_FILE, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['run_name', 'Mini', 'Zini', 'Xini', 'Yini',
                        'omega_frac', 'fov', 'run_dir', 'status'])

        for M, Z, omega, fov in combos:
            X, Y = compute_composition(Z)
            name = make_run_name(M, Z, omega, fov)
            run_dir = OUTPUT_BASE / name

            # ALWAYS write CSV row
            writer.writerow([name, M, Z, X, Y, omega, fov, str(run_dir), 'pending'])

            # Skip file creation if already exists
            if run_dir.exists():
                continue

            run_dir.mkdir(parents=True, exist_ok=True)

            # Symlink the star binary
            src = MESA_WORK_DIR / 'star'
            dst = run_dir / 'star'
            if not dst.exists():
                os.symlink(src, dst)

            # Copy scripts and column lists
            for fname in ['mk', 'rn', 're', 'clean',
                         'history_columns.list', 'profile_columns.list']:
                src = MESA_WORK_DIR / fname
                dst = run_dir / fname
                if src.is_file() and not dst.exists():
                    shutil.copy2(src, dst)

            # Create LOGS directory
            (run_dir / 'LOGS').mkdir(exist_ok=True)

            # Write the REDIRECTOR inlist (critical fix!)
            (run_dir / 'inlist').write_text(template_inlist)

            # Write parameterised inlist_project
            inlist = template_inlist_project
            inlist = inlist.replace('__MINI__', f'{M:.2f}')
            inlist = inlist.replace('__ZINI__', f'{Z:.4f}')
            inlist = inlist.replace('__YINI__', f'{Y:.6f}')
            inlist = inlist.replace('__FOV__', f'{fov:.4f}')
            inlist = inlist.replace('__OMEGA_FRAC__', f'{omega:.2f}d0')
            (run_dir / 'inlist_project').write_text(inlist)

    print(f"Grid setup complete. {len(combos)} models in {OUTPUT_BASE}")
    print(f"Parameter file: {CSV_FILE}")


# =====================================================================
# GRID EXECUTION
# =====================================================================
def run_single_model(row):
    """Run a single MESA model. Returns (run_name, status, message)."""
    name = row['run_name']
    run_dir = row['run_dir']

    # Check if already completed
    final_mod = os.path.join(run_dir, 'final.mod')
    if os.path.isfile(final_mod):
        return (name, 'already_done', 'final.mod exists')

    print(f"  Starting {name}...")
    try:
        result = subprocess.run(
            ['./rn'],
            cwd=run_dir,
            capture_output=True,
            text=True,
            timeout=172800  # 48 hour timeout
        )
        # in run_single_model, right after subprocess.run
        with open(os.path.join(run_dir, 'mesa.log'), 'w') as f:
            f.write(f"MESA_DIR={os.environ.get('MESA_DIR')}\n")
            f.write(f"MESASDK_ROOT={os.environ.get('MESASDK_ROOT')}\n")
            f.write(f"PATH={os.environ.get('PATH')}\n\n")
            f.write(result.stdout or '')
            f.write('\n--- STDERR ---\n')
            f.write(result.stderr or '')

        if not os.path.isfile(os.path.join(run_dir, 'LOGS', 'history.data')):
            return (name, 'no_output', f'exited {result.returncode}, no history.data')
        
        if result.returncode == 0:
            return (name, 'completed', '')
        else:
            err_file = os.path.join(run_dir, 'error.log')
            with open(err_file, 'w') as f:
                f.write(result.stdout[-5000:] if result.stdout else '')
                f.write('\n--- STDERR ---\n')
                f.write(result.stderr[-5000:] if result.stderr else '')
            return (name, 'failed', f'returncode={result.returncode}')
    except subprocess.TimeoutExpired:
        return (name, 'timeout', '48h exceeded')
    except Exception as e:
        return (name, 'error', str(e))


def run_grid(n_parallel=1):
    """Run the grid, optionally in parallel."""
    if not CSV_FILE.is_file():
        print(f"ERROR: {CSV_FILE} not found. Run --setup first.")
        sys.exit(1)

    with open(CSV_FILE, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    pending = [r for r in rows if r['status'] == 'pending']
    print(f"Found {len(pending)} pending models (of {len(rows)} total)")

    if not pending:
        print("Nothing to run.")
        return

    if n_parallel == 1:
        for row in pending:
            name, status, msg = run_single_model(row)
            row['status'] = status
            print(f"  {name}: {status} {msg}")
            _update_csv(rows)
    else:
        with ProcessPoolExecutor(max_workers=n_parallel) as executor:
            futures = {executor.submit(run_single_model, r): r for r in pending}
            for future in as_completed(futures):
                row = futures[future]
                name, status, msg = future.result()
                row['status'] = status
                print(f"  {name}: {status} {msg}")
        _update_csv(rows)

    statuses = {}
    for r in rows:
        s = r['status']
        statuses[s] = statuses.get(s, 0) + 1
    print(f"\nGrid summary: {statuses}")


def _update_csv(rows):
    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


# =====================================================================
# VERIFICATION
# =====================================================================
def verify_setup():
    """Check a random run directory to make sure substitution worked."""
    run_dirs = sorted(OUTPUT_BASE.glob('M*'))
    if not run_dirs:
        print("No run directories found. Run --setup first.")
        return

    d = run_dirs[0]
    print(f"Checking {d.name}...")

    # Check redirector
    inlist = (d / 'inlist').read_text()
    if 'extra_star_job_inlist_name' not in inlist:
        print("  ERROR: inlist is NOT a redirector!")
        return
    print("  OK: inlist is a proper redirector")

    # Check inlist_project has substituted values
    ip = (d / 'inlist_project').read_text()
    if '__MINI__' in ip or '__ZINI__' in ip:
        print("  ERROR: template variables not substituted!")
        return
    print("  OK: template variables substituted")

    # Check column files exist
    for f in ['history_columns.list', 'profile_columns.list']:
        if (d / f).exists():
            print(f"  OK: {f} present")
        else:
            print(f"  MISSING: {f}")

    # Print key parameters
    for line in ip.split('\n'):
        for kw in ['initial_mass', 'initial_z', 'initial_y',
                   'new_near_zams_omega_div_omega_crit', 'overshoot_f(1)']:
            if kw in line and '!' not in line.split(kw)[0]:
                print(f"  {line.strip()}")


# =====================================================================
# MAIN
# =====================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HD 35929 MESA Grid v2')
    parser.add_argument('--setup', action='store_true', help='Create run directories')
    parser.add_argument('--run', action='store_true', help='Execute pending models')
    parser.add_argument('--parallel', type=int, default=1, help='Parallel models')
    parser.add_argument('--count', action='store_true', help='Count grid size')
    parser.add_argument('--verify', action='store_true', help='Check setup is correct')
    args = parser.parse_args()

    if args.count:
        n = len(M_INI_LIST) * len(Z_INI_LIST) * len(OMEGA_FRAC_LIST) * len(F_OV_LIST)
        print(f"Grid: {len(M_INI_LIST)} M × {len(Z_INI_LIST)} Z × "
              f"{len(OMEGA_FRAC_LIST)} ω × {len(F_OV_LIST)} fov = {n} models")
        sys.exit(0)

    if args.setup:
        setup_grid()

    if args.verify:
        verify_setup()

    if args.run:
        run_grid(args.parallel)

    if not any([args.setup, args.run, args.count, args.verify]):
        parser.print_help()
