#!/usr/bin/env python3
"""
HD 35929 — MESA Grid Runner, PMS-only extension to 4.0 Msun
Non-rotating PMS from the pre-MS model, stopped at ZAMS (Lnuc/L >= 0.99).
No rotation injection, no MS. Same physics as run_grid_new.py otherwise
(templates/inlist_project_pms differs from inlist_project only in that).

Runs are named like the main grid's non-rotating tracks
(M3.3_Z0.014_omega0.00_fov0.017), so index_mesa.py and the GYRE scripts
pick them up unchanged, but live in their own folder on STER with their
own parameter CSV -- the main grid's parameter_grid.csv is not touched.

Usage (on a compute node, inside screen/tmux):
    python3 run_grid_pms_ext.py --count
    python3 run_grid_pms_ext.py --setup --verify
    python3 run_grid_pms_ext.py --run --parallel 8
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
TEMPLATE_FILE  = "inlist_project_pms"
OUTPUT_BASE    = Path("/STER/kianTedeskino/HD35929/mesa_grid/grid_runs_pms_ext2")
CSV_FILE       = OUTPUT_BASE / "parameter_grid_pms_ext2.csv"

# =====================================================================
# GRID PARAMETER SPACE
# =====================================================================
# # Mass: 3.3 to 4.2 Msun in 0.1 steps (10 values) -- the main grid stops at 3.2
M_INI_LIST = [round(3.3 + 0.1*i, 1) for i in range(10)]

# Metallicity: 0.006 to 0.022 in 0.004 steps (5 values), as in the main grid
Z_INI_LIST = [round(0.006 + 0.004*i, 3) for i in range(5)]

# Exponential overshooting f_ov (5 values), as in the main grid. The PMS
# tracks are NOT identical across f_ov (envelope / transient-core overshoot):
# the main grid's M3.2 Z0.014 tracks at f_ov 0.005 and 0.035 differ by up to
# 0.04 dex in log L before ZAMS.
F_OV_LIST = [0.005, 0.010, 0.017, 0.025, 0.035]

OMEGA_FRAC = 0.00   # naming only: no rotation is injected

# --- TEST SUBSET ---
# M_INI_LIST = [4.0]
# Z_INI_LIST = [0.014]
# F_OV_LIST = [0.017]

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
    template_inlist_project = (TEMPLATE_DIR / TEMPLATE_FILE).read_text()
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

    combos = list(itertools.product(M_INI_LIST, Z_INI_LIST, F_OV_LIST))
    print(f"Setting up {len(combos)} models...")

    with open(CSV_FILE, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['run_name', 'Mini', 'Zini', 'Xini', 'Yini',
                        'omega_frac', 'fov', 'run_dir', 'status'])

        for M, Z, fov in combos:
            X, Y = compute_composition(Z)
            name = make_run_name(M, Z, OMEGA_FRAC, fov)
            run_dir = OUTPUT_BASE / name

            # ALWAYS write CSV row
            writer.writerow([name, M, Z, X, Y, OMEGA_FRAC, fov, str(run_dir), 'pending'])

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

            # Write the REDIRECTOR inlist
            (run_dir / 'inlist').write_text(template_inlist)

            # Write parameterised inlist_project
            inlist = template_inlist_project
            inlist = inlist.replace('__MINI__', f'{M:.2f}')
            inlist = inlist.replace('__ZINI__', f'{Z:.4f}')
            inlist = inlist.replace('__YINI__', f'{Y:.6f}')
            inlist = inlist.replace('__FOV__', f'{fov:.4f}')
            inlist = inlist.replace('__XC_STOP__', f'{X - 0.01:.6f}d0')
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

    print(f"  Starting {name}...", flush=True)
    try:
        result = subprocess.run(
            ['./rn'],
            cwd=run_dir,
            capture_output=True,
            text=True,
            timeout=172800  # 48 hour timeout
        )
        with open(os.path.join(run_dir, 'mesa.log'), 'w') as f:
            f.write(f"MESA_DIR={os.environ.get('MESA_DIR')}\n")
            f.write(f"MESASDK_ROOT={os.environ.get('MESASDK_ROOT')}\n")
            f.write(f"PATH={os.environ.get('PATH')}\n\n")
            f.write(result.stdout or '')
            f.write('\n--- STDERR ---\n')
            f.write(result.stderr or '')

        if not os.path.isfile(os.path.join(run_dir, 'LOGS', 'history.data')):
            return (name, 'no_output', f'exited {result.returncode}, no history.data')

        if result.returncode != 0:
            err_file = os.path.join(run_dir, 'error.log')
            with open(err_file, 'w') as f:
                f.write(result.stdout[-5000:] if result.stdout else '')
                f.write('\n--- STDERR ---\n')
                f.write(result.stderr[-5000:] if result.stderr else '')
            return (name, 'failed', f'returncode={result.returncode}')

        # Did it stop at ZAMS, or on the X_c safety net / something else?
        out = result.stdout or ''
        if 'Lnuc_div_L >= Lnuc_div_L_zams_limit' in out:
            return (name, 'completed', 'stopped at ZAMS')
        if 'central lower limit for h1' in out:
            return (name, 'completed_xc_stop', 'X_c safety net fired, not the ZAMS stop')
        return (name, 'completed_other', 'normal exit, stop reason not recognised (see mesa.log)')
    except subprocess.TimeoutExpired:
        return (name, 'timeout', '48h exceeded')
    except Exception as e:
        return (name, 'error', str(e))


def check_mesa_env():
    """Fail fast if ./star cannot load its shared libraries -- otherwise every
    model 'finishes' in a second with no output."""
    r = subprocess.run(['ldd', str(MESA_WORK_DIR / 'star')],
                       capture_output=True, text=True)
    missing = [l.split()[0] for l in r.stdout.splitlines() if 'not found' in l]
    if missing:
        print(f"ERROR: star cannot load {missing}. Initialise the MESA SDK first:")
        print("  export MESASDK_ROOT=~/Codes/mesasdk   # wherever the SDK lives")
        print("  source $MESASDK_ROOT/bin/mesasdk_init.sh")
        sys.exit(1)


def run_grid(n_parallel=1):
    """Run the grid, optionally in parallel. The CSV is rewritten after every
    finished model, so progress can be followed with grep and an interrupted
    run resumes where it stopped (models with final.mod are skipped)."""
    if not CSV_FILE.is_file():
        print(f"ERROR: {CSV_FILE} not found. Run --setup first.")
        sys.exit(1)
    check_mesa_env()

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
            print(f"  {name}: {status} {msg}", flush=True)
            _update_csv(rows)
    else:
        with ProcessPoolExecutor(max_workers=n_parallel) as executor:
            futures = {executor.submit(run_single_model, r): r for r in pending}
            for future in as_completed(futures):
                row = futures[future]
                name, status, msg = future.result()
                row['status'] = status
                print(f"  {name}: {status} {msg}", flush=True)
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
    """Check a run directory to make sure substitution worked."""
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
    left = [v for v in ('__MINI__', '__ZINI__', '__YINI__', '__FOV__', '__XC_STOP__')
            if v in ip]
    if left:
        print(f"  ERROR: template variables not substituted: {left}")
        return
    print("  OK: template variables substituted")

    # No rotation anywhere, and the ZAMS stop is on
    active = [l.split('!')[0].strip() for l in ip.split('\n')]
    active = [l for l in active if l]
    if any('omega' in l.lower() and 'relax' in l.lower() for l in active):
        print("  ERROR: a rotation relax setting is active!")
        return
    print("  OK: no rotation injection")
    if 'stop_near_zams = .true.' not in active:
        print("  ERROR: stop_near_zams is not set!")
        return
    print("  OK: stops at ZAMS")

    # Check column files exist
    for f in ['history_columns.list', 'profile_columns.list']:
        if (d / f).exists():
            print(f"  OK: {f} present")
        else:
            print(f"  MISSING: {f}")

    # Print key parameters
    for line in active:
        for kw in ['initial_mass', 'initial_z', 'initial_y', 'overshoot_f(1)',
                   'stop_near_zams', 'Lnuc_div_L_zams_limit',
                   'xa_central_lower_limit(1)']:
            if line.startswith(kw):
                print(f"  {line}")


# =====================================================================
# MAIN
# =====================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HD 35929 MESA grid, PMS-only extension')
    parser.add_argument('--setup', action='store_true', help='Create run directories')
    parser.add_argument('--run', action='store_true', help='Execute pending models')
    parser.add_argument('--parallel', type=int, default=1, help='Parallel models')
    parser.add_argument('--count', action='store_true', help='Count grid size')
    parser.add_argument('--verify', action='store_true', help='Check setup is correct')
    args = parser.parse_args()

    if args.count:
        n = len(M_INI_LIST) * len(Z_INI_LIST) * len(F_OV_LIST)
        print(f"Grid: {len(M_INI_LIST)} M × {len(Z_INI_LIST)} Z × "
              f"{len(F_OV_LIST)} fov = {n} models (PMS only, no rotation)")
        print(f"  M = {M_INI_LIST}")
        print(f"  output: {OUTPUT_BASE}")
        sys.exit(0)

    if args.setup:
        setup_grid()

    if args.verify:
        verify_setup()

    if args.run:
        run_grid(args.parallel)

    if not any([args.setup, args.run, args.count, args.verify]):
        parser.print_help()
