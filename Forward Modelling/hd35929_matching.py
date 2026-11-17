"""
hd35929_matching.py — dataframe plumbing for the HD 35929 mode matching.

No physics or statistics decisions live in here. Everything tunable is a
function argument, set from the notebook. This file is responsible only for
getting the numbers into the right shape and doing the arithmetic fast.

The one structural choice worth knowing about: matching is vectorised over
the whole grid rather than looped over models. A row-wise loop over
~4e7 (model, order, m-assignment) trials is hours in Python; pivoting the
multiplets to wide form and evaluating all models at once for each of the
10 m-assignments is 10 vectorised operations and takes seconds.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

M_VALUES = (-2, -1, 0, 1, 2)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load_modes(path, mode_type, l, m=None, freq_lo=None, freq_hi=None,
               columns=None) -> pd.DataFrame:
    """Read modes.parquet with predicate pushdown.

    The frequency window is applied inside pyarrow, so a 35M-row table never
    materialises in pandas. Widen it generously: when multiplets are
    synthesised from m=0 the stored frequency can sit up to 2*beta*Omega
    (~1.3 d^-1 here) away from the observed member it must reproduce.
    """
    filt = (pc.field("mode_type") == mode_type) & (pc.field("l") == l)
    if m is not None:
        filt = filt & (pc.field("m") == m)
    if freq_lo is not None:
        filt = filt & (pc.field("freq_d") >= freq_lo)
    if freq_hi is not None:
        filt = filt & (pc.field("freq_d") <= freq_hi)

    cols = columns or ["track_id", "M", "Z", "omega", "fov", "profile",
                       "l", "m", "n_p", "n_g", "n_pg", "freq_d", "beta"]
    return pq.read_table(path, columns=cols, filters=filt).to_pandas()


def synthesise_multiplet(df, omega_col="f_rot_model_cpd", beta_col="beta",
                         freq_col="freq_d", m_values=M_VALUES,
                         beta_override=None, carry=()) -> pd.DataFrame:
    """Build a multiplet from m=0 frequencies by first-order Ledoux splitting.

        nu_m = nu_0 + m * beta * Omega,      beta = 1 - C_nl

    GYRE treats p-mode rotation as first-order Ledoux by construction (the
    TAR applies only to gravito-inertial modes), so this reproduces exactly
    what m-resolved GYRE p-mode runs would have written. Running m=0 only
    costs nothing. Produces the same wide layout as multiplet_table, so
    match_multiplet is unchanged downstream.

    Omega is per-model (from MESA), not a single observed value, so the
    grid's rotation axis genuinely constrains the fit: a model reproduces
    the observed splitting only if its own beta * Omega matches it.

    The sign convention for m does not affect chi2 or the ranking, because
    match_multiplet pairs observed to predicted in ascending frequency
    order. It affects only the m labels reported in m_assign.

    The second-order term (m^2 * D * Omega^2 / nu_0) is absent because GYRE
    does not model it for p-modes, not because it is dropped here. Predicted
    splittings are therefore exactly linear in m, and no GYRE model can
    reproduce a multiplet asymmetry. For HD 35929 the observed departure
    from equal spacing is ~1.6e-3 d^-1 at m = -2; since it is unfittable by
    every model in the grid it shifts chi2 near-uniformly and leaves the
    ranking intact. Reproducing it is a job for STORM.
    """
    out = df[["track_id", "profile", "n_pg"]].copy()
    f0 = df[freq_col].to_numpy(dtype=float)
    om = df[omega_col].to_numpy(dtype=float)
    beta = (np.full(len(df), float(beta_override)) if beta_override is not None
            else df[beta_col].to_numpy(dtype=float))

    for m in m_values:
        out[f"m{m:+d}"] = f0 + m * beta * om

    out["freq_m0"] = f0
    out["beta_model"] = beta
    out["C_model"] = 1.0 - beta
    out["omega_used"] = om
    out["split_model"] = beta * om
    for c in carry:                     # e.g. omega_source, rotating_structure
        if c in df.columns:
            out[c] = df[c].to_numpy()
    return out


def solve_multiplet_asymmetry(freqs, m_assign=(-2, 0, 1)):
    """Recover nu_0, a = beta*Omega and the second-order coefficient X from
    three observed multiplet members, with no model input.

    CAVEAT: with three members and three unknowns this is an exact solve,
    not a fit. There are no residuals and no goodness-of-fit -- any three
    frequencies can be written this way. It says only "if the assignment is
    m_assign, the departure from equal spacing is X". A fourth observed
    member would over-determine the system and make X testable.

    X is not something GYRE can fit: its p-mode splitting is exactly linear
    in m. Treat X as a prediction target for a code that models second-order
    rotation (STORM), and as a rough scale for the irreducible residual that
    every GYRE model in the grid carries.
    """
    f = np.sort(np.asarray(freqs, dtype=float))
    m = np.asarray(m_assign, dtype=float)
    if len(f) != 3:
        raise ValueError("need exactly three multiplet members")
    A = np.column_stack([np.ones(3), m, m ** 2])
    nu0, a, X = np.linalg.solve(A, f)
    return a, X


# --------------------------------------------------------------------------
# multiplet matching
# --------------------------------------------------------------------------


def attach_models(df, models_path, how="inner"):
    """Join the per-(source, track_id, profile) MESA quantities onto modes."""
    models = pd.read_parquet(models_path)
    keys = ["track_id", "profile"]
    if "source" in models.columns and "source" in df.columns:
        keys = ["source"] + keys
    else:
        print("WARNING: merging without `source` - unsafe once two grids "
              "share track_id/profile (GYRE and StORM will)")
 
    df = df.copy()
    for k in keys:
        if k in ("source", "track_id"):
            models[k] = models[k].astype(str)
            df[k] = df[k].astype(str)
 
    dup = [c for c in models.columns if c in df.columns and c not in keys]
    return df.merge(models.drop(columns=dup), on=keys, how=how,
                    validate="many_to_one")
 


# --------------------------------------------------------------------------
# observed candidates
# --------------------------------------------------------------------------
_FREQ_COLS = ["freq", "Freq_d", "freq_d", "frequency"]
_ERR_COLS = ["freq_err_mc", "freq_err", "Freq_err_d", "freq_err_d"]
_ID_COLS = ["ID", "id", "Name", "name"]


def _pick(df, options, label):
    for c in options:
        if c in df.columns:
            return c
    raise KeyError(f"no {label} column found; tried {options}, "
                   f"have {list(df.columns)}")


def load_candidates(path, ids=None, freq_col=None, err_col=None,
                    id_col=None) -> pd.DataFrame:
    """Read an observed-frequency list into a tidy ascending frame.

    Returns columns [ID, freq, err], sorted ascending in frequency. Matching
    pairs observed to predicted by frequency order, so the sort here is what
    makes the pairing well defined regardless of how the IDs were numbered.

    path=None returns a well-formed empty frame, which is what lets the
    g-mode branch be skipped without a special case at the call site.

    Delimiter and column names are sniffed, so both the comma-separated
    prewhitening output and the older whitespace candidate files load
    without configuration.
    """
    cols = ["ID", "freq", "err"]
    if path is None:
        return pd.DataFrame(columns=cols)

    df = pd.read_csv(path)
    if df.shape[1] == 1:                       # not comma-separated
        df = pd.read_csv(path, sep=r"\s+")
    if df.empty:
        return pd.DataFrame(columns=cols)

    id_col = id_col or _pick(df, _ID_COLS, "ID")
    freq_col = freq_col or _pick(df, _FREQ_COLS, "frequency")
    err_col = err_col or _pick(df, _ERR_COLS, "frequency error")

    if ids is not None:
        missing = set(ids) - set(df[id_col])
        if missing:
            raise KeyError(f"requested IDs not in {path}: {sorted(missing)}")
        df = df[df[id_col].isin(ids)]

    out = (df[[id_col, freq_col, err_col]]
           .rename(columns={id_col: "ID", freq_col: "freq", err_col: "err"})
           .sort_values("freq")
           .reset_index(drop=True))
    return out


# --------------------------------------------------------------------------
# multiplet matching
# --------------------------------------------------------------------------
def multiplet_table(modes) -> pd.DataFrame:
    """Pivot to one row per (track_id, profile, n_pg), one column per m.

    validate="many_to_one" upstream and the duplicate check here are the two
    places a doubled ingest would surface. If this raises, modes.parquet has
    repeated rows and the ingest needs rerunning.
    """
    key = ["track_id", "profile", "n_pg"]
    if modes.duplicated(subset=key + ["m"]).any():
        raise ValueError(
            "duplicate (track_id, profile, n_pg, m) rows — modes.parquet "
            "contains a doubled tree; rerun ingest_gyre.py"
        )
    wide = modes.pivot_table(index=key, columns="m", values="freq_d",
                             aggfunc="first")
    wide = wide.reindex(columns=list(M_VALUES))
    wide.columns = [f"m{m:+d}" for m in wide.columns]
    return wide.reset_index()


def match_multiplet(wide, obs_freqs, obs_errs, floor=0.0,
                    combos=None, keep="best") -> pd.DataFrame:
    """chi2 for every m-assignment, vectorised over all models at once.

    obs_freqs must be ascending. Predicted frequencies are sorted ascending
    too, so the pairing is by frequency order rather than by m label — the m
    of an observed peak is not known a priori. The m ordering that produced
    the match is recorded in `m_assign`.

    floor is added in quadrature to obs_errs. With raw TESS errors of ~1e-5
    d^-1 the chi2 is dominated by whichever frequency happens to sit closest
    and the other two barely contribute; the floor is what makes all three
    carry comparable weight.

    keep="best" returns one row per (track, profile, n_pg) — the assignment
    with lowest chi2. keep="all" returns all assignments, which you want if
    you care about how degenerate the assignment is.
    """
    obs_freqs = np.asarray(obs_freqs, dtype=float)
    obs_errs = np.asarray(obs_errs, dtype=float)
    if not np.all(np.diff(obs_freqs) > 0):
        raise ValueError("obs_freqs must be strictly ascending")

    sig = np.sqrt(obs_errs ** 2 + floor ** 2)
    n_obs = len(obs_freqs)
    combos = combos or list(combinations(M_VALUES, n_obs))

    cols = [f"m{m:+d}" for m in M_VALUES]
    out = []
    for combo in combos:
        take = [f"m{m:+d}" for m in combo]
        pred = wide[take].to_numpy(dtype=float)          # (N, n_obs)

        order = np.argsort(pred, axis=1)
        pred_sorted = np.take_along_axis(pred, order, axis=1)
        m_sorted = np.asarray(combo)[order]              # m in freq order

        chi2 = (((pred_sorted - obs_freqs) / sig) ** 2).sum(axis=1)

        block = wide[["track_id", "profile", "n_pg"]].copy()
        block["chi2_seis"] = chi2
        block["m_assign"] = [",".join(map(str, r)) for r in m_sorted]
        for j in range(n_obs):
            block[f"pred_{j}"] = pred_sorted[:, j]
            block[f"resid_{j}"] = pred_sorted[:, j] - obs_freqs[j]
        block["split_pred"] = pred_sorted[:, -1] - pred_sorted[:, 0]
        out.append(block)

    res = pd.concat(out, ignore_index=True)
    res = res[np.isfinite(res["chi2_seis"])]             # incomplete multiplets
    if keep == "best":
        res = (res.sort_values("chi2_seis")
                  .drop_duplicates(subset=["track_id", "profile", "n_pg"]))
    return res.reset_index(drop=True)


def match_singlets(modes, obs_freqs, obs_errs, floor=0.0,
                   group=("track_id", "profile", "l", "m")) -> pd.DataFrame:
    """chi2 for unsplit frequencies (the g-mode path), one-to-one.

    Each observed frequency takes the nearest model mode within its group,
    with the constraint that two observed frequencies cannot claim the same
    model mode — the old notebook allowed that, and in a dense g-mode
    spectrum it happens. Enforced greedily in ascending-residual order.

    Returns an empty frame with the right columns when obs_freqs is empty,
    so the caller needs no special case.
    """
    cols = ["chi2_seis", "n_matched"] + list(group)
    if len(obs_freqs) == 0:
        return pd.DataFrame(columns=cols)

    obs_freqs = np.asarray(obs_freqs, dtype=float)
    sig = np.sqrt(np.asarray(obs_errs, dtype=float) ** 2 + floor ** 2)

    rows = []
    for gkey, g in modes.groupby(list(group), sort=False):
        f = g["freq_d"].to_numpy()
        npg = g["n_pg"].to_numpy()
        if len(f) < len(obs_freqs):
            continue
        d = np.abs(f[None, :] - obs_freqs[:, None]) / sig[:, None]

        used, chi2, assigned = set(), 0.0, {}
        for i in np.argsort(d.min(axis=1)):              # easiest first
            order = np.argsort(d[i])
            pick = next((j for j in order if j not in used), None)
            if pick is None:
                break
            used.add(pick)
            chi2 += d[i, pick] ** 2
            assigned[i] = npg[pick]
        if len(assigned) < len(obs_freqs):
            continue

        row = dict(zip(group, gkey if isinstance(gkey, tuple) else (gkey,)))
        row["chi2_seis"] = chi2
        row["n_matched"] = len(assigned)
        for i, npg_i in assigned.items():
            row[f"g{i}_n_pg"] = npg_i
        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=cols)


# --------------------------------------------------------------------------
# filtering
# --------------------------------------------------------------------------
def apply_filters(df, cuts) -> pd.DataFrame:
    """Add a boolean pass_<name> column per cut, plus pass_all.

    cuts is a dict name -> callable(df) -> boolean Series. Nothing is
    dropped: every model keeps a record of which cuts it failed, which is
    what makes the N-1 cutflow below possible.
    """
    df = df.copy()
    for name, fn in cuts.items():
        df[f"pass_{name}"] = fn(df).fillna(False).astype(bool)
    pass_cols = [f"pass_{n}" for n in cuts]
    df["pass_all"] = df[pass_cols].all(axis=1)
    df["n_failed"] = (~df[pass_cols]).sum(axis=1)
    return df


def cutflow(df, cuts) -> pd.DataFrame:
    """Sequential and N-1 survival per cut.

    The N-1 column is the informative one: it is how many models would
    survive if that single cut were removed, so it isolates which constraint
    is actually doing the work rather than which happens to be applied first.
    """
    names = list(cuts)
    rows, mask = [], pd.Series(True, index=df.index)
    for n in names:
        mask &= df[f"pass_{n}"]
        others = [f"pass_{o}" for o in names if o != n]
        n_minus_1 = df[others].all(axis=1).sum() if others else len(df)
        rows.append({
            "cut": n,
            "passes_alone": int(df[f"pass_{n}"].sum()),
            "cumulative": int(mask.sum()),
            "n_minus_1": int(n_minus_1),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# derived geometry
# --------------------------------------------------------------------------
R_SUN = 6.957e8
G_SI = 6.674e-11
M_SUN = 1.989e30


def add_geometry(df, vsini_kms, f_rot_cpd) -> pd.DataFrame:
    """sin i, inclination, v_eq and v/v_crit from the model radius.

    v_eq is computed from the model radius and the OBSERVED rotation
    frequency, not from the model's own surf_avg_v_rot, so that sin_i is a
    statement about the observation. The model's own value is kept alongside
    as v_eq_model for comparison; a large discrepancy means the model's
    rotation is inconsistent with F1 regardless of how well it fits.
    """
    df = df.copy()
    R_m = df["R"].to_numpy() * R_SUN
    M_kg = df["star_mass"].to_numpy() * M_SUN
    f_hz = f_rot_cpd / 86400.0

    v_eq = 2 * np.pi * R_m * f_hz                        # m/s
    df["v_eq_kms"] = v_eq / 1e3
    df["sin_i"] = (vsini_kms * 1e3) / v_eq
    with np.errstate(invalid="ignore"):
        df["incl_deg"] = np.degrees(np.arcsin(np.clip(df["sin_i"], -1, 1)))
    df.loc[df["sin_i"] > 1, "incl_deg"] = np.nan         # geometrically excluded

    v_crit = np.sqrt(G_SI * M_kg / R_m)
    df["v_crit_kms"] = v_crit / 1e3
    df["v_over_vcrit"] = v_eq / v_crit
    return df


def add_residual_decomposition(df, n_obs=3):
    """Split multiplet residuals into centroid and splitting components.

    Operates on the output of match_multiplet, which carries resid_0..resid_j
    and the m ordering in m_assign. Exact least-squares in closed form -- the
    model is linear in m, so no iteration is needed.
    """
    m = np.array([[int(x) for x in s.split(",")] for s in df["m_assign"]],
                    dtype=float)
    r = df[[f"resid_{j}" for j in range(n_obs)]].to_numpy(dtype=float)

    mbar = m.mean(axis=1, keepdims=True)
    rbar = r.mean(axis=1, keepdims=True)
    denom = ((m - mbar) ** 2).sum(axis=1)

    dsplit = ((m - mbar) * (r - rbar)).sum(axis=1) / denom
    dnu0 = rbar[:, 0] - dsplit * mbar[:, 0]

    out = df.copy()
    out["dnu0"] = dnu0
    out["dsplit"] = dsplit
    # residual left after removing both -- the part no linear-in-m model can
    # absorb, i.e. the multiplet asymmetry.
    out["resid_asym"] = np.sqrt(
        ((r - (dnu0[:, None] + m * dsplit[:, None])) ** 2).mean(axis=1))
    return out
