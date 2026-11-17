"""
prewhiten.py -- Iterative pre-whitening and significance assessment for
unevenly sampled photometric time series.

Written for the HD 35929 TESS analysis. Reproduces the Period04 procedure
(Lenz & Breger 2005) in a scripted, reproducible form, and adds the
diagnostics needed to defend the frequency list in a referee report:

  * classical Deeming (1975) amplitude spectrum, Period04 normalisation
  * iterative pre-whitening with a SIMULTANEOUS non-linear refit of all
    frequencies at every step (variable-projection / VARPRO formulation)
  * local noise from the FINAL residual spectrum in a box of fixed width
    in d^-1 (not in array elements)
  * Monte Carlo uncertainties (Period04-style)
  * analytical uncertainties, Montgomery & O'Donoghue (1999), using the
    TIME-DOMAIN residual rms -- with the Schwarzenberg-Czerny (1991)
    correlation factor D applied
  * spectral-window overlays to test whether a candidate is an alias
  * injection-recovery to calibrate the detection threshold empirically
  * cross-dataset frequency matching at the Rayleigh limit of the SHORTER
    dataset, not at the formal error

Conventions
-----------
Model:      y(t) = Z + sum_i A_i * sin( 2*pi*( f_i*(t-t0) + phi_i ) )
            with phi_i the FRACTIONAL phase in [0, 1), as in Period04.
Amplitude:  same units as the input y. Feed magnitudes (or relative flux)
            and you get amplitudes in mag (or relative flux).
Frequency:  cycles per unit time. Feed time in days -> frequencies in d^-1.

References
----------
Breger et al. 1993, A&A 271, 482            -- S/N > 4 criterion
Baran & Koen 2021, AcA 71, 113              -- S/N ~ 5 for space photometry
Deeming 1975, Ap&SS 36, 137                 -- classical DFT
Lenz & Breger 2005, CoAst 146, 53           -- Period04
Loumos & Deeming 1978, Ap&SS 56, 285        -- 1.5/T resolution criterion
Montgomery & O'Donoghue 1999, DSSN 13, 28   -- analytical uncertainties
Schwarzenberg-Czerny 1991, MNRAS 253, 198   -- correlation factor D
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.ndimage import uniform_filter1d


__all__ = [
    "amplitude_spectrum",
    "spectral_window",
    "fit_sinusoids",
    "local_noise",
    "correlation_factor",
    "prewhiten",
    "monte_carlo_errors",
    "analytic_errors",
    "injection_recovery",
    "match_frequencies",
    "rayleigh",
    "equal_run_weights",
    "coherence_ratios",
    "window_match",
]


# --------------------------------------------------------------------------
# Amplitude spectrum
# --------------------------------------------------------------------------

def rayleigh(t):
    """Rayleigh frequency resolution 1/T. Loumos & Deeming use 1.5/T."""
    return 1.0 / (np.max(t) - np.min(t))


def amplitude_spectrum(t, y, fmin=None, fmax=None, oversample=20,
                       freq=None, method="dft", chunk=2000, subtract_mean=True,
                       weights=None):
    """
    Amplitude spectrum in Period04 normalisation: a pure sinusoid of
    amplitude A produces a peak of height A.

    method : {"dft", "ls"}
        "dft" -- exact classical Deeming DFT. O(N_f * N). Use this for the
                 spectra you publish.
        "ls"  -- astropy Lomb-Scargle with fit_mean=False, center_data=True
                 (i.e. the Scargle 1982 periodogram) converted to amplitude.
                 O(N log N) with method="fast". Use for very large grids
                 during the search; the difference from the DFT is far below
                 the peak-refinement step that follows.
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)

    # weights : per-point weights, normalised internally so that a pure
    # sinusoid of amplitude A still returns a peak of height A.
    #
    # The natural (unweighted) merge is dominated by whichever run has the
    # most points -- for HD 35929 that is Run 3 with 82% -- so the merged
    # window is essentially Run 3's window and the merge barely tests
    # coherence. Weighting the runs EQUALLY restores the three-cluster fringe
    # structure and sharpens the coherent/incoherent contrast from
    # 1.000 vs 0.834 to 1.000 vs 0.577. Use it as a DIAGNOSTIC only: it
    # upweights the noisiest data, so detection should still use the natural
    # weighting.
    if weights is not None:
        weights = np.asarray(weights, float)
        if weights.shape != t.shape:
            raise ValueError("weights must have the same shape as t")
        if np.any(weights < 0):
            raise ValueError("weights must be non-negative")
        weights = weights / weights.sum()
        if method == "ls":
            raise ValueError(
                "weights are only implemented for method='dft'. The "
                "Lomb-Scargle fast path applies its own internal weighting "
                "and would silently give a different normalisation.")
    if subtract_mean:
        y = y - (np.average(y, weights=weights) if weights is not None
                 else np.mean(y))

    if freq is None:
        if fmin is None:
            fmin = 0.0
        if fmax is None:
            fmax = 0.5 / np.median(np.diff(np.sort(t)))
        step = rayleigh(t) / float(oversample)
        freq = np.arange(max(fmin, step), fmax + step, step)
    freq = np.asarray(freq, float)

    if method == "ls":
        from astropy.timeseries import LombScargle
        power = LombScargle(t, y, fit_mean=False,
                            center_data=True).power(freq, normalization="psd")
        return freq, np.sqrt(np.abs(4.0 * power / len(t)))

    t0 = t.mean()
    tt = t - t0
    amp = np.empty(freq.size)
    if weights is None:
        wy = y
        norm = 2.0 / len(t)
    else:
        wy = weights * y          # weights already sum to 1
        norm = 2.0
    for i in range(0, freq.size, chunk):
        f = freq[i:i + chunk]
        ph = 2.0 * np.pi * np.outer(f, tt)
        re = np.cos(ph) @ wy
        im = np.sin(ph) @ wy
        amp[i:i + chunk] = np.hypot(re, im) * norm
    return freq, amp


def equal_run_weights(t, boundaries):
    """
    Per-point weights giving each run the SAME total weight, regardless of how
    many points it contains.

    boundaries : list of (t_start, t_end) for each run.

    With natural weighting the merged transform is dominated by the largest
    run and the merge is a weak test of coherence. Equalising the runs makes
    an incoherent signal fall to 1/sqrt(n_runs) of a coherent one, which is a
    much sharper discriminator.
    """
    t = np.asarray(t, float)
    w = np.zeros(t.size)
    for lo, hi in boundaries:
        m = (t >= lo) & (t <= hi)
        if m.sum():
            w[m] = 1.0 / m.sum()
    if np.any(w == 0):
        raise ValueError(f"{int((w == 0).sum())} points fall outside all "
                         "supplied run boundaries")
    return w / w.sum()


def coherence_ratios(weights_per_run):
    """
    Predicted merged/single-run amplitude ratio under the two hypotheses.

    coherent   : the signal keeps its phase across every run, so the runs add
                 in phase and the merged amplitude equals the intrinsic one.
    incoherent : each run carries the signal at a random phase, so they add
                 in quadrature.

    Pass the fractional weight of each run (summing to 1).
    """
    w = np.asarray(weights_per_run, float)
    w = w / w.sum()
    return dict(coherent=1.0, incoherent=float(np.sqrt((w ** 2).sum())))


def spectral_window(t, f0, half_width=None, oversample=20, n=None,
                    weights=None):
    """
    Spectral window of the sampling, centred on f0 and normalised to 1.

    This is the pattern that a single perfectly coherent sinusoid at f0
    would produce. Overlay it on the observed spectrum: any candidate
    peak that falls on a sidelobe is an alias, not an independent mode.
    """
    t = np.asarray(t, float)
    if half_width is None:
        half_width = 20.0 * rayleigh(t)
    if n is None:
        n = int(2 * half_width / (rayleigh(t) / oversample))
    freq = np.linspace(f0 - half_width, f0 + half_width, n)
    unit = np.sin(2.0 * np.pi * f0 * (t - t.mean()))
    _, amp = amplitude_spectrum(t, unit, freq=freq, subtract_mean=True,
                                weights=weights)
    return freq, amp / amp.max()


# --------------------------------------------------------------------------
# Simultaneous multi-sinusoid fit (variable projection)
# --------------------------------------------------------------------------

def _design(tt, freqs):
    """[1, cos(2pi f1 t), sin(2pi f1 t), cos(2pi f2 t), ...]"""
    ph = 2.0 * np.pi * np.outer(tt, np.asarray(freqs, float))
    X = np.empty((tt.size, 1 + 2 * len(freqs)))
    X[:, 0] = 1.0
    X[:, 1::2] = np.cos(ph)
    X[:, 2::2] = np.sin(ph)
    return X


def _linear_solve(tt, y, freqs, check_rank=True):
    X = _design(tt, freqs)
    coef, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    if check_rank and rank < X.shape[1]:
        # Degenerate: two frequencies have collided and the columns are
        # linearly dependent. lstsq will happily return two enormous
        # cancelling amplitudes. Refuse.
        raise np.linalg.LinAlgError(
            f"design matrix rank-deficient ({rank} < {X.shape[1]}): "
            "two frequencies have converged to the same value")
    return coef, y - X @ coef


def fit_sinusoids(t, y, freqs_init, refine_freqs=True, t0=None,
                  anchors=None, bound_rayleigh=0.5, flim=None,
                  min_sep=1.0, bounds_frac=None, max_nfev=None,
                  amp_sanity_factor=20.0, tol=1e-14):
    """
    Fit y = Z + sum A_i sin(2pi(f_i (t-t0) + phi_i)), optimising ALL
    frequencies simultaneously by variable projection.

    anchors : reference frequencies for the search bounds. Each f_i is
        constrained to anchors[i] +/- bound_rayleigh/T. Defaults to
        freqs_init.

        *** THIS MUST BE FIXED ACROSS ITERATIONS OF A PRE-WHITENING LOOP. ***

        If the bounds are re-centred on the current best value at every
        iteration, each frequency performs a random walk and can drift
        arbitrarily far from where it was selected -- out of the search
        window entirely, or onto a neighbour. Two frequencies that collide
        make the design matrix rank-deficient, and the linear solve then
        returns a pair of huge amplitudes that cancel to numerical precision.
        Amplitudes of order 10^5 mmag are the signature.

        With anchors >= 1.5 Rayleigh apart (the selection criterion) and
        bound_rayleigh = 0.5, frequencies can never come within 0.5 Rayleigh
        of one another, so the collision is impossible by construction.

    flim : (fmin, fmax) hard clamp, so no frequency can leave the window.
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    f0 = np.atleast_1d(np.asarray(freqs_init, float)).copy()
    if t0 is None:
        t0 = t.mean()
    tt = t - t0
    anc = f0.copy() if anchors is None else \
        np.atleast_1d(np.asarray(anchors, float)).copy()

    if refine_freqs and f0.size:
        ray = rayleigh(t)
        if bounds_frac is not None:
            half = anc * bounds_frac
        else:
            # Non-overlapping bounds. It is NOT enough for the anchors to be
            # far apart: if every frequency may move +/- b, two anchors a
            # distance d apart can close to d - 2b. With d = 1.5 Rayleigh and
            # b = 0.5 Rayleigh that leaves 0.5 Rayleigh -- badly collinear,
            # huge cancelling amplitudes, and NOT caught by a rank check
            # because the columns are merely near-dependent, not dependent.
            #
            # So shrink each interval until the guaranteed separation is at
            # least min_sep Rayleigh: half_i = (d_nearest - min_sep) / 2.
            order = np.argsort(anc)
            d = np.full(anc.size, np.inf)
            s = anc[order]
            if s.size > 1:
                gaps = np.diff(s)
                d[order[:-1]] = np.minimum(d[order[:-1]], gaps)
                d[order[1:]] = np.minimum(d[order[1:]], gaps)
            half = np.minimum(bound_rayleigh * ray,
                              np.maximum((d - min_sep * ray) / 2.0, 0.0))
            half = np.maximum(half, 1e-9)

        lo, hi = anc - half, anc + half
        if flim is not None:
            lo = np.maximum(lo, flim[0])
            hi = np.minimum(hi, flim[1])
        lo = np.maximum(lo, 1e-10)
        hi = np.maximum(hi, lo + 1e-12)
        f0 = np.clip(f0, lo, hi)

        def resid_fun(f):
            try:
                return _linear_solve(tt, y, f)[1]
            except np.linalg.LinAlgError:
                return np.full(y.size, 1e6)

        # tol: convergence tolerance. 1e-14 is far below any meaningful
        # frequency precision and makes the optimiser grind -- a 20-frequency
        # refit takes ~19 s at 1e-14 and a fraction of that at 1e-10, with
        # frequencies identical to well within their uncertainties. Use a
        # loose tol inside a pre-whitening loop and a tight one for the single
        # final refit whose numbers you publish.
        sol = least_squares(resid_fun, f0, bounds=(lo, hi),
                            xtol=tol, ftol=tol, gtol=tol,
                            max_nfev=max_nfev)
        f0 = sol.x

    coef, resid = _linear_solve(tt, y, f0)
    offset = coef[0]
    a = coef[1::2]          # cos coefficient
    b = coef[2::2]          # sin coefficient
    amps = np.hypot(a, b)
    if amp_sanity_factor and amps.size:
        rms = np.std(y, ddof=1)
        if np.any(amps > amp_sanity_factor * rms):
            k = np.argsort(f0)
            raise RuntimeError(
                "fitted amplitude exceeds %.1fx the data rms (%.4f mmag). "
                "Two frequencies are unresolved and the solve is degenerate. "
                "Sorted frequencies: %s"
                % (amp_sanity_factor, rms * 1e3,
                   np.array2string(f0[k], precision=6)))
    # A sin(2pi f t + theta) = A sin(theta) cos + A cos(theta) sin
    theta = np.arctan2(a, b)
    phases = np.mod(theta / (2.0 * np.pi), 1.0)   # Period04 fractional phase

    return dict(freqs=f0, amps=amps, phases=phases, offset=offset,
                model=y - resid, resid=resid, t0=t0)


def evaluate_model(t, freqs, amps, phases, offset=0.0, t0=0.0):
    tt = np.asarray(t, float) - t0
    out = np.full(tt.shape, float(offset))
    for f, A, p in zip(freqs, amps, phases):
        out += A * np.sin(2.0 * np.pi * (f * tt + p))
    return out


# --------------------------------------------------------------------------
# Noise, correlation factor, uncertainties
# --------------------------------------------------------------------------

def local_noise(freq_res, amp_res, f0, box=1.0, exclude=0.0, stat="mean"):
    """
    Noise level at f0: the average amplitude of the residual spectrum in a
    box of total width `box` (in the same units as freq_res) centred on f0.

    box     : box WIDTH in d^-1. Breger et al. (1993) use a few d^-1; 1-5 d^-1
              is standard for space photometry. Quote this number in the paper.
    exclude : optionally blank out +/- this distance around f0 so that a
              residual peak does not inflate its own noise estimate. A few
              Rayleigh widths is sensible.
    stat    : "mean" (Breger convention) or "median" (robust). For Rayleigh-
              distributed noise the median sits ~6% below the mean, so the
              median inflates S/N by ~6%. Pick one and say which.
    """
    m = np.abs(freq_res - f0) <= box / 2.0
    if exclude > 0:
        m &= np.abs(freq_res - f0) > exclude
    if not m.any():
        return np.nan
    return np.median(amp_res[m]) if stat == "median" else np.mean(amp_res[m])


def correlation_factor(resid, max_lag=500):
    """
    Schwarzenberg-Czerny (1991) correlation factor
        D = sqrt(1 + 2 * sum_k rho_k)
    where rho_k is the autocorrelation of the residuals, summed to the first
    zero crossing. D = 1 for uncorrelated residuals; oversampled space
    photometry typically gives D of a few. Analytical uncertainties must be
    multiplied by D.
    """
    r = np.asarray(resid, float)
    r = r - r.mean()
    n = r.size
    max_lag = int(min(max_lag, n // 2))
    var = np.dot(r, r) / n
    if var == 0:
        return 1.0
    total = 0.0
    for k in range(1, max_lag + 1):
        rho = np.dot(r[:-k], r[k:]) / ((n - k) * var)
        if rho <= 0:
            break
        total += rho
    return float(np.sqrt(1.0 + 2.0 * total))

def band_D(t, resid, fmin_band):
    """Correlation factor measured only above fmin_band.
    D on the full residual is dominated by low-frequency red noise and
    reports the same value whatever band you extracted from."""
    if fmin_band == 0.0:
        return correlation_factor(resid)
    else:
        npts = int(round((1.0 / fmin_band) / np.median(np.diff(np.sort(t)))))
        slow = uniform_filter1d(resid, size=max(npts, 3), mode="nearest")
        return correlation_factor(resid - slow)

def analytic_errors(t, resid, amps, apply_D=True, sigma=None):
    """
    Montgomery & O'Donoghue (1999) analytical uncertainties.

        sigma(A)   = sqrt(2/N) * sigma_m
        sigma(f)   = sqrt(6/N) * sigma_m / (pi * T * A)
        sigma(phi) = sqrt(2/N) * sigma_m / (2 * pi * A)      [fractional phase]

    *** sigma_m is the RMS OF THE RESIDUAL LIGHT CURVE IN THE TIME DOMAIN. ***

    It is NOT the mean amplitude level of the residual amplitude spectrum.
    Substituting the latter underestimates every uncertainty by a factor
    sqrt(N/pi) -- about 90x for N = 25000. This is the single most common
    way these formulae get misapplied.

    (For white noise the two are related by <A_noise> = sigma_m * sqrt(pi/N),
    so if you only have the spectral noise level, use
    sigma_m = A_noise * sqrt(N/pi).)
    
    sigma may be a scalar or one value per frequency. For red residuals pass
    the frozen continuum evaluated at each frequency, converted to a
    time-domain rms via sigma_m = alpha(nu) * sqrt(N/pi), and leave
    apply_D=False. The continuum and D are two descriptions of the same
    excess low-frequency power; applying both double-counts it.
    """

    t = np.asarray(t, float)
    N = t.size
    T = np.max(t) - np.min(t)
    if sigma is None:
        sigma = np.std(np.asarray(resid, float), ddof=1)
    A = np.atleast_1d(np.asarray(amps, float))
    # sigma may be a scalar (one noise level for the whole band) or an array
    # with one entry per frequency (local continuum level at each candidate).
    sigma = np.broadcast_to(np.atleast_1d(np.asarray(sigma, float)),
                            A.shape).astype(float)

    D = correlation_factor(resid) if apply_D else 1.0

    sig_A = np.sqrt(2.0 / N) * sigma * D
    sig_f = np.sqrt(6.0 / N) * sigma / (np.pi * T * A) * D
    sig_p = np.sqrt(2.0 / N) * sigma / (2.0 * np.pi * A) * D
    return dict(sigma_m=sigma, D=D, freq_err=sig_f, amp_err=sig_A,
                phase_err=sig_p)

def red_noise_draw(t, rn_model, rng, dt):
    """One noise realisation whose amplitude spectrum follows rn_model."""
    N = t.size
    n = int(round(np.ptp(t) / dt)) + 1                  # uniform grid over the full span
    f = np.fft.rfftfreq(n, dt)
    sig = np.zeros(f.size)
    sig[1:] = rn_model(f[1:]) * np.sqrt(N / np.pi)      # continuum -> rms, same as analytic_errors
    c = np.sqrt(n / 2) * sig * (rng.normal(size=f.size) + 1j * rng.normal(size=f.size))
    x = np.fft.irfft(c, n)
    idx = np.clip(np.round((t - t.min()) / dt).astype(int), 0, n - 1)
    return x[idx]                                       # keep only the observed bins


def monte_carlo_errors(t, freqs, amps, phases, offset, t0, resid,
                       n_iter=500, seed=None, progress=False,
                       refit_freqs=False, noise_mode="white",
                       rn_model=None, dt=None):
    """
    Period04-style Monte Carlo. Sample the best-fit model at the observed
    times, add noise, refit, and take the standard deviation of the
    recovered parameters over all realisations.

    noise_mode
        "white" : Gaussian noise with the residual rms.
        "block" : contiguous chunks of the residuals. Keeps the correlation,
                  but after a deep extraction the residuals are depleted at
                  the extracted frequencies, so the errors come out low.
        "red"   : noise drawn from the frozen continuum `rn_model` (needs
                  rn_model; dt defaults to the median cadence).
    """
    rng = np.random.default_rng(seed)
    t = np.asarray(t, float)
    resid = np.asarray(resid, float)
    sigma = np.std(resid, ddof=1)
    if noise_mode == "red":
        if rn_model is None:
            raise ValueError("noise_mode='red' needs rn_model (e.g. rn_frozen['model'])")
        if dt is None:
            dt = np.median(np.diff(np.sort(t)))
    model = evaluate_model(t, freqs, amps, phases, offset, t0)

    # block length: one cycle of the lowest frequency, so a block carries the
    # correlation that a Gaussian draw destroys
    if noise_mode == "block":
        L = int(np.clip(round((1.0 / np.min(freqs)) / np.median(np.diff(np.sort(t)))), 8, t.size // 4))
        n_blocks = int(np.ceil(t.size / L))
    model = evaluate_model(t, freqs, amps, phases, offset, t0)

    F = np.empty((n_iter, len(freqs)))
    A = np.empty_like(F)
    P = np.empty_like(F)
    it = range(n_iter)
    if progress:
        try:
            from tqdm.auto import tqdm
            it = tqdm(it, desc="Monte Carlo")
        except ImportError:
            pass
    X = _design(t - t0, freqs)
    pinv = np.linalg.pinv(X)
    for i in it:
        if noise_mode == "block":
            starts = rng.integers(0, resid.size - L, size=n_blocks)
            noise = np.concatenate([resid[s:s + L] for s in starts])[:t.size]
        elif noise_mode == "red":
            noise = red_noise_draw(t, rn_model, rng, dt)
        else:
            noise = rng.normal(0.0, sigma, t.size)
        ysim = model + noise
        if refit_freqs:
            r = fit_sinusoids(t, ysim, freqs, refine_freqs=True, t0=t0,
                              anchors=freqs)
            F[i], A[i], P[i] = r["freqs"], r["amps"], r["phases"]
        else:
            c = pinv @ ysim
            a_, b_ = c[1::2], c[2::2]
            F[i] = freqs
            A[i] = np.hypot(a_, b_)
            P[i] = np.mod(np.arctan2(a_, b_) / (2 * np.pi), 1.0)

    # phases wrap; use circular std
    ang = 2 * np.pi * P
    R = np.hypot(np.mean(np.cos(ang), axis=0), np.mean(np.sin(ang), axis=0))
    p_err = np.sqrt(-2.0 * np.log(np.clip(R, 1e-12, 1.0))) / (2 * np.pi)

    return dict(freq_err=(F.std(axis=0, ddof=1) if refit_freqs
                          else np.full(len(freqs), np.nan)),
                amp_err=A.std(axis=0, ddof=1),
                phase_err=p_err, n_iter=n_iter, refit_freqs=refit_freqs)


# --------------------------------------------------------------------------
# The pre-whitening loop
# --------------------------------------------------------------------------

def prewhiten(t, y, fmin=0.1, fmax=None, oversample=20, snr_stop=3.5,
              max_freqs=60, noise_box=1.0, noise_stat="mean",
              search_method="dft", verbose=True, resolution_factor=1.5,
              stop_mode="snr", n_extract=25, noise_model="box",
              rn_fmax=None, amp_sanity_factor=20.0, rn_frozen=None,
              refit_max_nfev=None, refit_tol=1e-10,
              f_floor_rayleigh=1.0, f_floor_abs=0.0):
    """
    stop_mode : {"snr", "fixed"}
        "snr"   -- stop when the running S/N of the next peak falls below
                   snr_stop. Fine for white residuals.
        "fixed" -- extract exactly n_extract frequencies regardless, then
                   assess all of them at the end.

        *** USE "fixed" IF THE RESIDUALS ARE RED. ***

        The running S/N is measured against a residual spectrum that still
        contains every signal not yet removed. When the low-frequency region
        carries a strong stochastic component, that estimate is inflated and
        the loop terminates after one or two frequencies while real signals
        remain. Extracting a fixed number and judging them once, at the end,
        against a fitted noise profile decouples the stopping rule from the
        significance criterion -- which is what you want, because they are
        answering different questions.

    noise_model : {"box", "red"}
        "box" -- mean amplitude of the final residual spectrum in a box of
                 width noise_box centred on each frequency (Breger 1993).
        "red" -- fit alpha(nu) = alpha0/(1+(nu/nu_char)^gamma) + C_w to the
                 final residual spectrum and evaluate it at each frequency.
                 Removes the box-width dependence. Use when D >> 1.
    """
    """
    Iterative pre-whitening.

    Strategy (important): the loop runs down to a DELIBERATELY LOW threshold
    (`snr_stop`, default 3.5) using the running residual spectrum. The
    publishable S/N values are then recomputed once, at the end, against the
    FINAL residual spectrum. This matters because the noise level drops as
    signal is removed -- assessing each frequency against a different noise
    realisation is not a consistent criterion. Apply your 5-sigma cut to the
    returned table afterwards.

    fmin : low-frequency cutoff. 0.1 d^-1 is a reasonable default for TESS,
           where residual scattered-light and pointing systematics dominate on
           sector-length timescales. Justify whatever you choose in the text.
    resolution_factor : new peaks must sit at least this many Rayleigh widths
           from every already-extracted frequency (Loumos & Deeming 1978
           recommend 1.5). Prevents the solver from fitting two sinusoids to
           one unresolved peak, which is the classic way an incoherent signal
           generates spurious close pairs.

    Returns (table, result) where `result` is the final fit_sinusoids dict.
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    y = y - np.mean(y)
    N, T = t.size, np.max(t) - np.min(t)
    ray = 1.0 / T
    if fmax is None:
        fmax = 0.5 / np.median(np.diff(np.sort(t)))

    step = ray / oversample
    grid = np.arange(max(fmin, step), fmax + step, step)
    f_floor = max(fmin, f_floor_rayleigh * ray, f_floor_abs)
    if rn_fmax is None:
        rn_fmax = min(0.5 / np.median(np.diff(np.sort(t))), 100.0)

    freqs = []
    anchors = []
    resid = y.copy()
    fit = None
    history = []

    while len(freqs) < max_freqs:
        fg, ag = amplitude_spectrum(t, resid, freq=grid, method=search_method)

        # mask out the immediate neighbourhood of everything already extracted
        ok = np.ones(fg.size, bool)
        ok &= fg >= f_floor
        for f in freqs:
            ok &= np.abs(fg - f) > resolution_factor * ray
        if not ok.any():
            break

        k = np.argmax(np.where(ok, ag, -np.inf))
        f_new, a_new = fg[k], ag[k]
        noise = local_noise(fg, ag, f_new, box=noise_box,
                            exclude=resolution_factor * ray, stat=noise_stat)
        snr_running = a_new / noise if noise and np.isfinite(noise) else np.inf

        if stop_mode == "snr" and snr_running < snr_stop:
            if verbose:
                print(f"  stop: next peak {f_new:.6f} d^-1 has running "
                      f"S/N = {snr_running:.2f} < {snr_stop}")
            break
        if stop_mode == "fixed" and len(freqs) >= n_extract:
            break

        trial = freqs + [f_new]
        anchors.append(f_new)
        # refit_max_nfev caps the in-loop simultaneous refit. Cost of that
        # refit grows as ~k^3 in the number of frequencies extracted so far
        # (a numerical Jacobian needs k+1 solves, each O(N k^2)), so an
        # uncapped loop to n_extract=25 takes minutes per dataset. The
        # in-loop fit only has to be good enough to subtract cleanly before
        # the next peak search -- do ONE uncapped simultaneous refit on the
        # final frequency list afterwards and quote those values.
        fit = fit_sinusoids(t, y, trial, refine_freqs=True,
                            anchors=anchors, bound_rayleigh=0.5,
                            flim=(f_floor, fmax),
                            max_nfev=refit_max_nfev, tol=refit_tol)
        freqs = list(fit["freqs"])
        resid = fit["resid"]
        history.append(dict(n=len(freqs), f=fit["freqs"][-1],
                            A=fit["amps"][-1], snr_running=snr_running,
                            resid_rms=np.std(resid, ddof=1)))
        if verbose:
            print(f"F{len(freqs):<3d} f = {fit['freqs'][-1]:12.6f} d^-1   "
                  f"A = {fit['amps'][-1]*1e3:9.4f} mmag   "
                  f"S/N(run) = {snr_running:5.2f}")

    if fit is None:
        return pd.DataFrame(), None

    # ---- final, consistent significance assessment -----------------------
    data_rms = np.std(y, ddof=1)
    bad = np.asarray(fit["amps"]) > amp_sanity_factor * data_rms
    if bad.any():
        raise RuntimeError(
            f"{bad.sum()} fitted amplitude(s) exceed {amp_sanity_factor}x the "
            f"data rms ({data_rms*1e3:.4f} mmag). This means two frequencies "
            "collided and the design matrix went singular. Reduce "
            "bound_rayleigh or increase resolution_factor.")

    fg, ag = amplitude_spectrum(t, fit["resid"], freq=grid, method=search_method)

    rn = None
    if noise_model == "red" and rn_frozen is not None:
        # Use a profile fitted BEFORE the deep extraction. See the note in
        # fit_red_noise about why re-fitting after extracting many
        # frequencies biases the noise level low.
        rn = rn_frozen
        if verbose:
            print(f"\n  using FROZEN red-noise model: "
                  f"alpha0 = {rn['alpha0']*1e3:.5f} mmag, "
                  f"nu_char = {rn['nu_char']:.4f} d^-1, "
                  f"gamma = {rn['gamma']:.2f}, "
                  f"C_white = {rn['C_white']*1e3:.5f} mmag")
    elif noise_model == "red":
        # Fit over the FULL available range, not just the search window.
        # The profile has four parameters and C_white is constrained almost
        # entirely by the flat high-frequency plateau. Fitting over a narrow
        # low-frequency window leaves it unconstrained, gamma runs to its
        # bound, and the "model" degenerates to a constant -- at which point
        # it is a box by another name.
        f_full = np.arange(max(fmin, step), rn_fmax + step, step)
        gf, ga = amplitude_spectrum(t, fit["resid"], freq=f_full,
                                    method="ls")
        rn = fit_red_noise(gf, ga)
        if verbose:
            print(f"\n  red-noise fit over {f_full[0]:.2f}-{f_full[-1]:.1f} "
                  f"d^-1: alpha0 = {rn['alpha0']*1e3:.5f} mmag, "
                  f"nu_char = {rn['nu_char']:.4f} d^-1, "
                  f"gamma = {rn['gamma']:.2f}, "
                  f"C_white = {rn['C_white']*1e3:.5f} mmag")
            if rn["gamma"] > 11.5 or rn["gamma"] < 0.15:
                print("  *** WARNING: gamma is at its bound -- the red-noise "
                      "fit has FAILED. Do not use these S/N values. Widen "
                      "rn_fmax so the white plateau is included. ***")
            if rn["alpha0"] < 0.5 * rn["C_white"]:
                print("  *** NOTE: alpha0 < C_white/2, i.e. no significant "
                      "red component was found. Use noise_model='box'. ***")

    rows = []
    for i, (f, A, p) in enumerate(zip(fit["freqs"], fit["amps"],
                                      fit["phases"]), start=1):
        nz_box = local_noise(fg, ag, f, box=noise_box,
                             exclude=resolution_factor * ray, stat=noise_stat)
        nz_red = float(rn["model"](f)) if rn is not None else np.nan
        nz = nz_red if noise_model == "red" else nz_box
        rows.append(dict(ID=f"F{i}", freq=f, amp=A, phase=p,
                         period=1.0 / f if f else np.nan,
                         noise_box=nz_box, noise_red=nz_red,
                         noise=nz, snr=A / nz if nz else np.nan,
                         snr_box=A / nz_box if nz_box else np.nan))
    table = pd.DataFrame(rows).sort_values("freq").reset_index(drop=True)
    # NOTE: attrs must hold scalars only. Putting numpy arrays in DataFrame
    # .attrs breaks any operation that concatenates frames (nlargest, concat,
    # merge) because pandas compares attrs with ==.
    table.attrs.update(N=N, T=T, rayleigh=ray, noise_box_width=noise_box,
                       noise_stat=noise_stat, noise_model=noise_model,
                       fmin=fmin, fmax=fmax,
                       resid_rms=float(np.std(fit["resid"], ddof=1)))
    if rn is not None:
        table.attrs.update(rn_alpha0=rn["alpha0"], rn_nu_char=rn["nu_char"],
                           rn_gamma=rn["gamma"], rn_C_white=rn["C_white"])
    fit["red_noise"] = rn
    fit["history"] = pd.DataFrame(history)
    return table, fit


# --------------------------------------------------------------------------
# Injection-recovery
# --------------------------------------------------------------------------

def injection_recovery(t, resid, amplitudes, freq_range, n_trials=50,
                       tol_factor=1.0, oversample=5, noise_box=1.0,
                       noise_stat="mean", seed=None, progress=True,
                       mode="targeted", snr_threshold=None,
                       search_method="ls", noise_fn=None):
    """
    Empirically calibrate the detection threshold.

    Inject a sinusoid of known amplitude at a random frequency into the REAL
    residuals -- real noise colour, real window function -- and try to recover
    it exactly as the pipeline would. Sweep amplitude.

    mode : {"targeted", "blind"}
        "targeted" -- look for the strongest peak WITHIN tol of the injected
            frequency and record its S/N. Answers "if this signal were
            present, what S/N would I assign it?", which is what you need to
            calibrate a threshold.
        "blind" -- require the injected signal to be the global maximum of
            the whole window. Much stricter: in a red-noise regime a real
            but modest signal will lose to a noise peak elsewhere and be
            scored as a failure, which conflates detectability with
            "is it the largest thing present".

    snr_threshold : if given, also report the fraction of trials whose
        recovered S/N exceeds it -- the completeness at that threshold.

    noise_fn : callable mapping frequency -> noise amplitude. Defaults to the
    local box. Pass rn["model"] to score against the frozen red-noise
    profile, which is what you want if the extraction used
    noise_model="red" -- otherwise the calibrated threshold and the
    quoted S/N are measured against different noise definitions.

    Set amplitudes=[0.0] with mode="targeted" to measure the FALSE ALARM
    distribution: what S/N does pure noise get assigned? If that routinely
    exceeds 5 in the low-frequency region, S/N > 5 is not a valid criterion
    for your data and you must raise the threshold to where it holds.

    SPEED: the noise level is computed ONCE from the input residuals rather
    than re-derived inside every trial (it barely changes when one small
    sinusoid is removed), and the search grid uses the FFT-based
    Lomb-Scargle. Together roughly 50x faster than the naive loop, with the
    same answer -- the grid only supplies a starting guess for the non-linear
    refinement, which is exact regardless.
    """
    rng = np.random.default_rng(seed)
    t = np.asarray(t, float)
    resid = np.asarray(resid, float)
    ray = rayleigh(t)
    step = ray / oversample
    lo = max(freq_range[0] - noise_box, step)
    grid = np.arange(lo, freq_range[1] + noise_box + step, step)

    # noise level computed once, from the unmodified residuals.
    # noise_fn : callable f -> noise amplitude. Pass the frozen red-noise
    # model so the calibrated S/N is on the SAME scale as the extraction.
    if noise_fn is None:
        fr0, ar0 = amplitude_spectrum(t, resid, freq=grid, method=search_method)
        noise_at = lambda f: local_noise(fr0, ar0, f, box=noise_box,
                                         exclude=1.5 * ray, stat=noise_stat)
    else:
        noise_at = noise_fn

    inner = (grid >= freq_range[0]) & (grid <= freq_range[1])
    out = []
    for A_inj in np.atleast_1d(amplitudes):
        rec, snrs, ferrs = 0, [], []
        for _ in range(n_trials):
            f_inj = rng.uniform(*freq_range)
            ysim = resid.copy()
            if A_inj > 0:
                ysim = ysim + A_inj * np.sin(
                    2 * np.pi * (f_inj * (t - t.mean()) + rng.uniform(0, 1)))
            _, ag = amplitude_spectrum(t, ysim, freq=grid,
                                       method=search_method)
            if mode == "targeted":
                m = inner & (np.abs(grid - f_inj) <= tol_factor * ray)
                if not m.any():
                    continue
                k = np.flatnonzero(m)[np.argmax(ag[m])]
            else:
                k = np.flatnonzero(inner)[np.argmax(ag[inner])]
            r = fit_sinusoids(t, ysim, [grid[k]], refine_freqs=True,
                              anchors=[grid[k]], bound_rayleigh=1.0)
            f_rec, a_rec = r["freqs"][0], r["amps"][0]
            nz = noise_at(f_rec)
            snr = a_rec / nz if nz and np.isfinite(nz) else np.nan
            if abs(f_rec - f_inj) <= tol_factor * ray:
                rec += 1
                snrs.append(snr)
                ferrs.append(abs(f_rec - f_inj))
        row = dict(amp_injected=A_inj,
                   recovery_fraction=rec / n_trials,
                   median_snr=np.median(snrs) if snrs else np.nan,
                   snr_16=np.percentile(snrs, 16) if snrs else np.nan,
                   snr_84=np.percentile(snrs, 84) if snrs else np.nan,
                   median_freq_err=np.median(ferrs) if ferrs else np.nan)
        if snr_threshold is not None:
            row["frac_above_threshold"] = (
                np.mean(np.array(snrs) >= snr_threshold) if snrs else 0.0)
        out.append(row)
        if progress:
            extra = (f"  frac S/N>{snr_threshold} = "
                     f"{row['frac_above_threshold']*100:5.1f}%"
                     if snr_threshold is not None else "")
            print(f"  A = {A_inj*1e3:8.5f} mmag  ->  recovered "
                  f"{row['recovery_fraction']*100:5.1f}%   median S/N = "
                  f"{row['median_snr']:6.2f} "
                  f"[{row['snr_16']:.2f}, {row['snr_84']:.2f}]{extra}")
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# Cross-dataset matching
# --------------------------------------------------------------------------

def match_frequencies(tables, time_bases, anchor=None, resolution_factor=1.5,
                      required=None):
    """
    Match frequencies across datasets.

    The tolerance for "these are the same peak" is the frequency RESOLUTION
    of the shorter of the two datasets, resolution_factor / T_short --
    NOT the formal (Monte Carlo or analytical) uncertainty.

    Formal errors describe how well the centroid of a peak is determined
    given that the peak is real and coherent. They can be orders of magnitude
    smaller than the width of the peak itself. Matching at N sigma of the
    formal error therefore asks a question the data cannot answer, and will
    reject genuine matches essentially at random.

    tables      : {label: DataFrame with 'freq' and 'amp' columns}
    time_bases  : {label: T in days}
    anchor      : label whose frequencies are iterated over (default: the
                  one with the longest time base)
    required    : labels that must contain a match for the row to be kept
    """
    labels = list(tables)
    if anchor is None:
        anchor = max(labels, key=lambda k: time_bases[k])
    others = [l for l in labels if l != anchor]
    if required is None:
        required = others

    rows = []
    for _, a in tables[anchor].iterrows():
        row = {"anchor_ID": a.get("ID", ""), f"{anchor}_freq": a["freq"],
               f"{anchor}_amp": a["amp"]}
        keep = True
        for lab in others:
            T = min(time_bases[anchor], time_bases[lab])
            tol = resolution_factor / T
            d = tables[lab]
            diff = np.abs(d["freq"].values - a["freq"])
            j = int(np.argmin(diff))
            if diff[j] <= tol:
                row[f"{lab}_freq"] = d["freq"].values[j]
                row[f"{lab}_amp"] = d["amp"].values[j]
                row[f"{lab}_dev"] = diff[j]
                row[f"{lab}_dev_over_tol"] = diff[j] / tol
            else:
                row[f"{lab}_freq"] = np.nan
                row[f"{lab}_amp"] = np.nan
                row[f"{lab}_dev"] = diff[j]
                row[f"{lab}_dev_over_tol"] = diff[j] / tol
                if lab in required:
                    keep = False
            row[f"{lab}_tol"] = tol
        row["consistent"] = keep
        rows.append(row)
    return pd.DataFrame(rows)


def window_match(t, y, f0, half_width=None, weights=None, oversample=20,
                 n=1501):
    """
    Test whether the power near f0 behaves like ONE coherent sinusoid.

    A signal that keeps its phase across every run produces, in the merged
    amplitude spectrum, exactly

        A * |W(f - f0)|

    where W is the merged spectral window -- a deterministic fringe pattern
    fixed by the sampling alone. A signal that decoheres between runs is a sum
    of the individual windows with random relative phases, which does not
    reproduce that pattern.

    So the diagnostic is the SHAPE of the fringes, not the peak height. The
    peak height does not discriminate: an incoherent signal also reaches ~1.0
    at whichever fringe its phases happen to align on. The pattern uses every
    fringe instead of one number.

    Returns:
      r            Pearson correlation between the observed spectrum and the
                   window template over +/- half_width. Near 1 -> coherent.
      f_peak       frequency of the strongest fringe
      offset_ray   f_peak - f0 in Rayleigh units. For a coherent signal this
                   is ~0; for an incoherent one it lands at a random fringe
                   within the envelope.
      amp_peak     amplitude at that fringe

    Interpret r together with offset_ray, and always against a control: run
    the same test on a frequency you believe is noise, and on one you believe
    is real (the rotation signal before you removed it) to calibrate what
    "high" means for YOUR sampling.
    """
    t = np.asarray(t, float)
    ray = rayleigh(t)
    if half_width is None:
        half_width = 0.6 / (np.max(t) - np.min(t)) * 32
    grid = np.linspace(f0 - half_width, f0 + half_width, n)

    _, obs = amplitude_spectrum(t, y, freq=grid, weights=weights)
    k = int(np.argmax(obs))
    f_peak = grid[k]

    _, tmpl = spectral_window(t, f_peak, half_width=half_width, n=n,
                              weights=weights)
    tmpl = tmpl * obs[k]

    a, b = obs - obs.mean(), tmpl - tmpl.mean()
    denom = np.sqrt(np.dot(a, a) * np.dot(b, b))
    r = float(np.dot(a, b) / denom) if denom > 0 else np.nan

    return dict(r=r, f_peak=f_peak, offset_ray=(f_peak - f0) / ray,
                amp_peak=float(obs[k]), half_width=half_width,
                grid=grid, observed=obs, template=tmpl)


def fit_red_noise(freq, amp, n_bins=200, fmin=None, fmax=None, p0=None,
                  mask_freqs=None, mask_width=None, bin_stat="median"):
    """
    Fit a red-noise (stochastic low-frequency variability) profile to a
    residual amplitude spectrum:

        alpha(nu) = alpha0 / (1 + (nu/nu_char)^gamma) + C_w

    This is the standard form used for SLF variability (Bowman et al. 2019,
    2020, A&A). alpha0 is the amplitude at zero frequency, nu_char the
    characteristic frequency at which the profile turns over, gamma the
    logarithmic slope, and C_w the white-noise floor.

    WHY THIS MATTERS: a flat local box assumes the noise is white within the
    box. If the residuals are red, a box centred at low frequency is dominated
    by the steeply rising red component and a WIDE box averages that together
    with the flat high-frequency floor. The first over-estimates the noise,
    the second under-estimates it, and neither is defensible. Fitting the
    profile removes the box-width dependence entirely.

    Check D = correlation_factor(residuals) first. D near 1 means the
    residuals are white and a box is fine. D of order 10 means they are not.

    Fit is done on logarithmically binned means to avoid the chi^2_2 scatter
    of individual bins dominating the least-squares.

    Returns dict with the parameters and a callable `model(nu)`.
    """
    freq = np.asarray(freq, float)
    amp = np.asarray(amp, float)
    m = np.isfinite(freq) & np.isfinite(amp) & (freq > 0)
    if mask_freqs is not None and len(mask_freqs):
        # Exclude the neighbourhood of known/candidate peaks before fitting
        # the continuum.
        #
        # This resolves a bias that runs in BOTH directions otherwise:
        #   - fit after a deep extraction  -> sinusoids have absorbed red
        #     power, continuum too LOW, every S/N too high
        #   - fit after removing only the dominant signal -> the candidate
        #     peaks themselves inflate the binned continuum, too HIGH, real
        #     signals rejected
        # Masking the peaks and fitting only the surrounding continuum avoids
        # both. Measured spread between the two unmasked extremes on real
        # data: a factor of 4.
        if mask_width is None:
            mask_width = 5.0 * (freq[1] - freq[0]) if freq.size > 1 else 0.0
        for f0 in np.atleast_1d(mask_freqs):
            m &= np.abs(freq - f0) > mask_width
    if fmin is not None:
        m &= freq >= fmin
    if fmax is not None:
        m &= freq <= fmax
    f, a = freq[m], amp[m]

    # bin_stat="median" is the ROBUST option and should be the default.
    #
    # The mean amplitude in a bin is pulled up by any peak inside it, so
    # fitting the binned mean makes the continuum track the signal -- which
    # is exactly the failure that made an unmasked fit come out ~4x too high
    # at low frequency. Peaks occupy a small fraction of a bin, so the median
    # barely moves. For Rayleigh-distributed noise
    #     median = mean * sqrt(2 ln2) / sqrt(pi/2) = 0.9394 * mean
    # so dividing by that factor converts the robust estimate back to the
    # mean-amplitude scale that S/N is defined against.
    RAYLEIGH_MEDIAN_OVER_MEAN = np.sqrt(2 * np.log(2)) / np.sqrt(np.pi / 2)
    edges = np.logspace(np.log10(f.min()), np.log10(f.max()), n_bins + 1)
    idx = np.digitize(f, edges) - 1
    fb, ab = [], []
    for i in range(n_bins):
        sl = idx == i
        if sl.sum() >= 3:
            fb.append(np.mean(f[sl]))
            if bin_stat == "median":
                ab.append(np.median(a[sl]) / RAYLEIGH_MEDIAN_OVER_MEAN)
            else:
                ab.append(np.mean(a[sl]))
    fb, ab = np.array(fb), np.array(ab)

    if p0 is None:
        p0 = [ab.max(), 1.0, 2.0, np.median(ab[fb > np.median(fb)])]

    def model_from(p, nu):
        a0, nuc, g, cw = p
        return a0 / (1.0 + (nu / max(nuc, 1e-8)) ** g) + cw

    def cost(p):
        if p[0] <= 0 or p[1] <= 0 or p[3] <= 0:
            return np.full(fb.size, 1e6)
        return np.log(model_from(p, fb)) - np.log(ab)

    sol = least_squares(cost, p0,
                        bounds=([1e-12, 1e-6, 0.1, 1e-12],
                                [np.inf, np.inf, 12.0, np.inf]),
                        xtol=1e-14, ftol=1e-14)
    a0, nuc, g, cw = sol.x
    return dict(alpha0=a0, nu_char=nuc, gamma=g, C_white=cw,
                model=lambda nu, p=sol.x: model_from(p, np.asarray(nu, float)),
                binned_freq=fb, binned_amp=ab, success=sol.success)


def iterate_red_noise(t, resid, fmin=0.1, fmax=None, oversample=6,
                      snr_threshold=4.0, mask_width_rayleigh=3.0,
                      max_iter=6, verbose=True):
    """
    Self-consistent red-noise continuum.

    Alternates between (a) fitting the continuum with the current peak list
    masked out, and (b) re-identifying which peaks exceed snr_threshold
    against that continuum -- until the peak list stops changing.

    This is the defensible version of the noise model: it neither absorbs
    the signal (as fitting after a deep extraction does) nor lets the signal
    inflate the noise (as fitting after minimal extraction does). Report the
    converged parameters and the number of iterations.

    Returns (red_noise_dict, peak_freqs, peak_amps).
    """
    t = np.asarray(t, float)
    ray = rayleigh(t)
    if fmax is None:
        fmax = min(0.5 / np.median(np.diff(np.sort(t))), 100.0)
    f, a = amplitude_spectrum(t, resid, fmin=fmin, fmax=fmax,
                              oversample=oversample, method="ls")
    mask_w = mask_width_rayleigh * ray

    peaks = np.array([])
    rn = None
    for it in range(max_iter):
        rn = fit_red_noise(f, a, mask_freqs=peaks, mask_width=mask_w)
        model = rn["model"](f)
        above = a > snr_threshold * model
        # collapse contiguous runs to one peak each
        new = []
        i = 0
        idx = np.flatnonzero(above)
        while i < idx.size:
            j = i
            while j + 1 < idx.size and idx[j + 1] == idx[j] + 1:
                j += 1
            seg = idx[i:j + 1]
            new.append(f[seg[np.argmax(a[seg])]])
            i = j + 1
        new = np.array(new)
        if verbose:
            print(f"    iter {it+1}: alpha0 = {rn['alpha0']*1e3:.5f} mmag, "
                  f"nu_char = {rn['nu_char']:.4f}, gamma = {rn['gamma']:.2f}, "
                  f"C_w = {rn['C_white']*1e3:.5f}, "
                  f"{new.size} peaks above {snr_threshold}x")
        if new.size == peaks.size and (new.size == 0 or
                                       np.allclose(np.sort(new),
                                                   np.sort(peaks), atol=ray)):
            break
        peaks = new
    amps = np.array([a[np.argmin(np.abs(f - p))] for p in peaks]) \
        if peaks.size else np.array([])
    return rn, peaks, amps


def block_bootstrap_errors(t, y, freqs, block_days=None, n_iter=300,
                           seed=None, progress=False, refit_freqs=False):
    """
    refit_freqs : if False (default) the frequencies are held fixed and only
        the linear amplitude/phase solve is repeated. This is roughly 100x
        faster -- one lstsq instead of a bounded non-linear optimisation per
        realisation -- and the amplitude uncertainties are essentially
        unchanged, because amplitude and frequency are close to uncorrelated
        for well-resolved peaks. Set True only if you specifically need
        bootstrap FREQUENCY errors, and expect it to take hours for ~20
        frequencies on 20000 points.
    """
    rng = np.random.default_rng(seed)
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    base = fit_sinusoids(t, y, freqs, refine_freqs=True)
    resid, model, t0 = base["resid"], base["model"], base["t0"]
    tt = t - t0

    if block_days is None:
        block_days = 10.0 * np.median(np.diff(np.sort(t)))
    span = np.max(t) - np.min(t)
    n_blocks = max(int(np.ceil(span / block_days)), 2)
    bin_id = np.minimum(((t - t.min()) / block_days).astype(int), n_blocks - 1)
    blocks = [np.flatnonzero(bin_id == b) for b in range(n_blocks)]
    blocks = [b for b in blocks if b.size]

    X = _design(tt, base["freqs"])
    pinv = np.linalg.pinv(X)

    F = np.empty((n_iter, len(freqs)))
    A = np.empty_like(F)
    it = range(n_iter)
    if progress:
        try:
            from tqdm.auto import tqdm
            it = tqdm(it, desc="Block bootstrap")
        except ImportError:
            pass
    for i in it:
        pieces = []
        for b in blocks:
            src = blocks[rng.integers(len(blocks))]
            take = src[:b.size] if src.size >= b.size else np.resize(src, b.size)
            pieces.append(resid[take])
        ysim = model + np.concatenate(pieces)[:t.size]
        if refit_freqs:
            r = fit_sinusoids(t, ysim, base["freqs"], refine_freqs=True,
                              t0=t0, anchors=base["freqs"])
            F[i], A[i] = r["freqs"], r["amps"]
        else:
            c = pinv @ ysim
            F[i] = base["freqs"]
            A[i] = np.hypot(c[1::2], c[2::2])

    return dict(freq_err=F.std(axis=0, ddof=1) if refit_freqs
                else np.full(len(freqs), np.nan),
                amp_err=A.std(axis=0, ddof=1),
                block_days=block_days, n_blocks=len(blocks), n_iter=n_iter,
                refit_freqs=refit_freqs)


def false_alarm_probability(snr, n_independent):
    """
    Probability that WHITE noise produces at least one peak of this S/N
    anywhere in a band containing n_independent resolution elements.

    For white noise the amplitude spectrum is Rayleigh distributed, and if
    S/N is defined against the MEAN amplitude then
        P(single bin > k) = exp(-pi k^2 / 4)
        P(any of N bins)  = 1 - (1 - P)^N
    The N factor is the trials correction for having searched the band, so
    this is directly comparable to a threshold applied after a blind search.

    n_independent = (fmax - fmin) * T.

    *** ONLY VALID WHERE THE RESIDUALS ARE WHITE. *** Check that the local
    noise is close to C_white from fit_red_noise before using it. In a
    red-noise region this will badly understate the false-alarm rate; use
    injection_recovery with amplitudes=[0.0] there instead.
    """
    p = np.exp(-np.pi * np.asarray(snr, float) ** 2 / 4.0)
    return 1.0 - (1.0 - p) ** n_independent


def snr_for_fap(fap, n_independent):
    """S/N threshold giving the requested band-wide false-alarm probability."""
    p = 1.0 - (1.0 - fap) ** (1.0 / n_independent)
    return float(np.sqrt(-4.0 / np.pi * np.log(p)))


def prune_unresolved(table, T, resolution_factor=1.5, amp_col="amp"):
    """
    Drop frequencies closer together than resolution_factor / T, keeping the
    higher-amplitude member of each unresolved group.

    Two peaks separated by less than the resolution limit are not two
    detections. Reporting both is a claim the data cannot support, and
    fitting both makes the least-squares problem ill-conditioned. Apply this
    to the final list before quoting anything.
    """
    tol = resolution_factor / T
    d = table.sort_values("freq").reset_index(drop=True)
    keep, dropped = [], []
    for i in range(len(d)):
        if not keep:
            keep.append(i)
            continue
        j = keep[-1]
        if d.loc[i, "freq"] - d.loc[j, "freq"] < tol:
            if d.loc[i, amp_col] > d.loc[j, amp_col]:
                dropped.append(d.loc[j, "freq"])
                keep[-1] = i
            else:
                dropped.append(d.loc[i, "freq"])
        else:
            keep.append(i)
    out = d.loc[keep].reset_index(drop=True)
    out.attrs["dropped_unresolved"] = dropped
    return out


def snr_box_sensitivity(t, resid, table, boxes=(0.5, 1.0, 2.0, 5.0),
                        fmin=0.1, fmax=None, oversample=20,
                        noise_stat="mean"):
    """
    How much does the quoted S/N depend on the width of the noise box?

    Run this before you commit to a threshold. S/N is not a property of the
    data alone -- it is a property of the data and your box. A candidate that
    sits at 5.4 with a 1 d^-1 box may sit at 4.6 with a 5 d^-1 box, and a
    referee is entitled to ask. Report the box width you used and show that
    your surviving candidates are not sensitive to a reasonable change in it.
    """
    ray = rayleigh(t)
    fg, ag = amplitude_spectrum(t, resid, fmin=fmin, fmax=fmax,
                                oversample=oversample)
    out = {"ID": table["ID"].values, "freq": table["freq"].values,
           "amp": table["amp"].values}
    for b in boxes:
        out[f"snr_box{b}"] = [
            row["amp"] / local_noise(fg, ag, row["freq"], box=b,
                                     exclude=1.5 * ray, stat=noise_stat)
            for _, row in table.iterrows()]
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# Convenience: harmonic / combination test
# --------------------------------------------------------------------------

def combination_search(table, tolerance, max_order=2, base_ids=None):
    """
    Flag frequencies consistent with n*f_i, or with f_i +/- f_j, within
    `tolerance` (use the Rayleigh resolution, not the formal error).

    Reports the deviation in units of the tolerance so you can state, e.g.,
    "F15 lies 2.6 resolution elements from 2*F1 and is therefore NOT
    consistent with the first harmonic" -- which is a real result, not a
    rounding issue.
    """
    f = table["freq"].values
    ids = table["ID"].values if "ID" in table else np.arange(len(f)).astype(str)
    if base_ids is None:
        base_ids = list(ids)
    out = []
    for k, (idk, fk) in enumerate(zip(ids, f)):
        best = None
        for i, (idi, fi) in enumerate(zip(ids, f)):
            # a frequency may not be a combination involving itself
            if i == k or idi not in base_ids:
                continue
            for n in range(2, max_order + 1):
                d = abs(fk - n * fi)
                if best is None or d < best[1]:
                    best = (f"{n}*{idi}", d)
            for j, (idj, fj) in enumerate(zip(ids, f)):
                if j in (i, k) or idj not in base_ids:
                    continue
                for sgn, s in ((1, "+"), (-1, "-")):
                    d = abs(fk - (fi + sgn * fj))
                    if best is None or d < best[1]:
                        best = (f"{idi}{s}{idj}", d)
        if best is None:
            best = ("-", np.inf)
        out.append(dict(ID=idk, freq=fk, best_combination=best[0],
                        deviation=best[1],
                        dev_over_tol=best[1] / tolerance,
                        consistent=best[1] <= tolerance))
    return pd.DataFrame(out)
