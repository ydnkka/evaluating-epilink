from __future__ import annotations

"""
Transmission heterogeneity estimators.

Utilities for fitting a negative binomial offspring distribution and summarizing
transmission concentration from individual-level offspring counts.
"""

import warnings

import numpy as np
import numpy.typing as npt
from scipy import optimize, stats


def _check_counts(counts: npt.ArrayLike) -> np.ndarray:
    r"""
    Validate offspring counts.

    Parameters
    ----------
    counts : array-like
        Offspring counts for each individual.

    Returns
    -------
    numpy.ndarray
        1D array of non-negative integer counts.

    Raises
    ------
    ValueError
        If the input is empty, non-finite, negative, or non-integer valued.
    """
    x = np.atleast_1d(np.asarray(counts, dtype=float))
    if x.size == 0:
        raise ValueError("Empty offspring array.")
    if np.any(~np.isfinite(x)):
        raise ValueError("Counts contain non-finite values.")
    if np.any(x < 0):
        raise ValueError("Counts must be non-negative.")
    # Ensure integer-valued: allow float inputs if they are integers numerically
    if np.any(np.abs(x - np.rint(x)) > 1e-12):
        raise ValueError("Counts must be integer-valued.")
    return np.rint(x).astype(int)


def _moments_nb(
    counts: np.ndarray,
    *,
    mean: float | None = None,
    var: float | None = None,
    tol: float = 1e-12,
) -> tuple[float, float]:
    r"""
    Method-of-moments fit for a negative binomial model.

    Uses the mean-dispersion parameterization:
        :math:`\mu = R`,
        :math:`\mathrm{Var}(X) = R + R^2/k`,
        :math:`k > 0`.
    The dispersion estimate is
        :math:`k = R^2 / (\mathrm{Var}(X) - R)`.

    Parameters
    ----------
    counts : numpy.ndarray
        1D array of non-negative integer counts.
    mean : float, optional
        Pre-computed sample mean. If not provided, it is computed from ``counts``.
    var : float, optional
        Pre-computed sample variance (ddof=1). If not provided, it is computed
        from ``counts``.
    tol : float, optional
        Tolerance for the variance <= mean check.

    Returns
    -------
    mean : float
        Sample mean (R).
    dispersion_k : float
        Method-of-moments dispersion estimate (k). Returns NaN if not
        identifiable.
    """
    x = np.asarray(counts)
    n = x.size
    R = float(np.mean(x) if mean is None else mean)

    if n < 2:
        warnings.warn(
            "Method-of-moments: n < 2, cannot estimate dispersion k; returning k = NaN.",
            RuntimeWarning
        )
        return R, np.nan

    var = float(np.var(x, ddof=1) if var is None else var)

    if var <= R + tol:
        # Under-dispersed or Poisson boundary; NB dispersion not identifiable
        warnings.warn(
            "Method-of-moments: sample variance <= mean; NB dispersion k not identifiable. "
            "Returning k = NaN (treat as Poisson-like).",
            RuntimeWarning
        )
        return R, np.nan

    k = (R ** 2) / (var - R)
    if not np.isfinite(k) or k <= 0:
        warnings.warn("Method-of-moments: computed invalid k; returning k = NaN.", RuntimeWarning)
        k = np.nan
    return R, float(k)


def _fit_nb_mle(counts: np.ndarray, eps: float = 1e-9) -> tuple[float, float]:
    r"""
    Fit a negative binomial model by maximum likelihood.

    Mean-dispersion parameterization:
        :math:`\mu = R`,
        :math:`\mathrm{Var}(X) = R + R^2/k`,
        :math:`k > 0`.
    ``scipy.stats.nbinom`` uses parameters :math:`(r=k, p=k/(k+R))`.

    Parameters
    ----------
    counts : numpy.ndarray
        1D array of non-negative integer counts.
    eps : float, optional
        Small positive value for numerical stability.

    Returns
    -------
    mean : float
        Estimated mean (R).
    dispersion_k : float
        Estimated dispersion (k).

    Raises
    ------
    RuntimeError
        If the optimization fails.

    Notes
    -----
    This function expects validated integer counts.
    """
    x = np.asarray(counts, dtype=int)
    if x.size == 0:
        raise ValueError("Empty offspring array.")
    if np.any(x < 0):
        raise ValueError("Counts must be non-negative integers.")

    # Initialise with method-of-moments
    Rt = max(x.mean(), eps)
    var = x.var(ddof=1) if x.size > 1 else Rt + 1.0
    k0 = max((Rt ** 2) / max(var - Rt, eps), 0.3)

    def nll(params: np.ndarray) -> float:
        R, logk = params
        if R <= 0:
            return np.inf
        k = np.exp(logk)
        p = k / (k + R)
        p = np.clip(p, 1e-12, 1 - 1e-12)
        ll = stats.nbinom.logpmf(x, k, p)
        if not np.all(np.isfinite(ll)):
            return np.inf
        return -float(np.sum(ll))

    res = optimize.minimize(
        nll,
        x0=np.array([Rt, np.log(k0)]),
        method="L-BFGS-B",
        bounds=[(1e-12, None), (np.log(1e-8), np.log(1e8))]
    )
    if not res.success:
        raise RuntimeError(f"NB MLE failed: {res.message}")

    R_hat = float(res.x[0])
    k_hat = float(np.exp(res.x[1]))
    return R_hat, k_hat


def _fit_nb_safely(counts: np.ndarray, tol: float = 1e-12) -> tuple[float, float, str, str]:
    r"""
    Fit a negative binomial model with robust fallbacks.

    The fit prefers MLE when :math:`\mathrm{Var}(X) > \mu`; otherwise it
    falls back to method-of-moments. Dispersion estimates can be NaN when
    the model is not identifiable.

    Parameters
    ----------
    counts : numpy.ndarray
        1D array of non-negative integer counts.
    tol : float, optional
        Tolerance for the variance <= mean check.

    Returns
    -------
    mean : float
        Estimated mean (R).
    dispersion_k : float
        Estimated dispersion (k). NaN if not identifiable.
    method : str
        Fitting method used ("mle", "moments", "moments-fallback").
    notes : str
        Diagnostic note for edge cases or fallback paths.
    """
    x = np.asarray(counts, dtype=int)
    n = x.size
    R_emp = float(np.mean(x))

    # Handle trivial cases
    if n < 2:
        R_mom, k_mom = _moments_nb(x, mean=R_emp, tol=tol)
        return R_mom, k_mom, "moments", "n<2"

    var = float(np.var(x, ddof=1))

    if var <= R_emp + tol:
        # NB dispersion not identifiable; MoM returns NaN for k
        R_mom, k_mom = _moments_nb(x, mean=R_emp, var=var, tol=tol)
        return R_mom, k_mom, "moments", "variance<=mean"

    # Try MLE, fall back to MoM if optimisation fails
    try:
        R_mle, k_mle = _fit_nb_mle(x)
        return R_mle, k_mle, "mle", ""
    except Exception as e:
        warnings.warn(f"MLE failed ({e}); falling back to method-of-moments.", RuntimeWarning)
        R_mom, k_mom = _moments_nb(x, tol=tol)
        return R_mom, k_mom, "moments-fallback", "mle-failed"


def _prop_for_80_percent(counts: np.ndarray) -> float:
    r"""
    Minimum fraction of individuals accounting for 80% of transmissions.

    Parameters
    ----------
    counts : numpy.ndarray
        1D array of non-negative integer counts.

    Returns
    -------
    float
        Fraction of individuals needed to reach 80% of total transmissions.
    """
    x = np.asarray(counts, dtype=int)
    n = x.size
    if n == 0:
        return np.nan
    tot = x.sum()
    if tot == 0:
        return 1.0
    order = np.sort(x)[::-1]
    cum = np.cumsum(order)
    idx = np.searchsorted(cum, 0.8 * tot, side="left")
    return (idx + 1) / n


def _bootstrap_nb(
    counts: np.ndarray,
    bootstrap: int = 200,
    seed: int | None = 123,
    tol: float = 1e-12,
) -> dict[str, object]:
    r"""
    Non-parametric bootstrap confidence intervals for ``R`` and ``k``.

    Uses the same safe fitter as the main estimate.

    Parameters
    ----------
    counts : numpy.ndarray
        1D array of non-negative integer counts.
    bootstrap : int, optional
        Number of bootstrap replicates.
    seed : int or None, optional
        Seed for the RNG.
    tol : float, optional
        Tolerance for variance <= mean checks in fitting.

    Returns
    -------
    dict
        Dictionary with the following keys:
        - ``R_samples``: array of bootstrap means.
        - ``k_samples``: array of bootstrap dispersions (finite only).
        - ``R_CI95``: 95% CI tuple for the mean.
        - ``k_CI95``: 95% CI tuple for the dispersion.
        - ``kept``: count of successful bootstrap estimates.
        - ``kept_R``: count of finite mean estimates.
        - ``kept_k``: count of finite dispersion estimates.
    """
    x = np.asarray(counts, dtype=int)
    rng = np.random.default_rng(seed)
    n = len(x)
    if n < 2 or bootstrap <= 0:
        return {
            "R_samples": np.array([]),
            "k_samples": np.array([]),
            "R_CI95": (np.nan, np.nan),
            "k_CI95": (np.nan, np.nan),
            "kept": 0,
            "kept_R": 0,
            "kept_k": 0,
        }

    Rs = np.full(bootstrap, np.nan, dtype=float)
    ks = np.full(bootstrap, np.nan, dtype=float)
    for i in range(bootstrap):
        sample = x[rng.integers(0, n, n)]
        Rb, kb, _, _ = _fit_nb_safely(sample, tol=tol)
        Rs[i] = Rb
        ks[i] = kb

    Rs = Rs[np.isfinite(Rs)]
    ks = ks[np.isfinite(ks)]

    def _ci95(arr):
        if arr.size == 0:
            return np.nan, np.nan
        return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

    out = {
        "R_samples": Rs,
        "k_samples": ks,
        "R_CI95": _ci95(Rs),
        "k_CI95": _ci95(ks),
        "kept": int(max(len(Rs), len(ks))),
        "kept_R": int(len(Rs)),
        "kept_k": int(len(ks)),
    }
    return out


def heterogeneity(
    offspring_dist: npt.ArrayLike,
    bootstrap: int = 200,
    bootstrap_seed: int | None = 123,
    superspreading_quantile: float = 0.99,
    tol: float = 1e-12
) -> dict[str, object]:
    r"""
    Estimate transmission heterogeneity from offspring counts.

    Parameters
    ----------
    offspring_dist : array-like of non-negative ints
        Offspring counts for ALL cases (including zeros).
    bootstrap : int
        Number of non-parametric bootstrap replicates for CI (0 to disable).
    bootstrap_seed : int or None
        RNG seed for bootstrap.
    superspreading_quantile : float in (0,1)
        Quantile :math:`q` for the superspreading threshold (default 0.99).
    tol : float
        Tolerance for variance <= mean checks.

    Returns
    -------
    dict
        Dictionary with keys:
        - ``meanRt``: mean of the offspring distribution.
        - ``disp_k``: dispersion parameter (k) of the negative binomial.
        - ``RtCI95``: 95% CI tuple for the mean (NaN if bootstrapping disabled).
        - ``kCI95``: 95% CI tuple for the dispersion (NaN if not identifiable).
        - ``superspreading_threshold``: Poisson :math:`q`-quantile threshold.
        - ``prop_80_percent_transmitters``: fraction accounting for 80% of transmission.
        - ``pct_zero_transmitters``: percentage with zero offspring.
        - ``pct_superspreaders``: percentage above the threshold.
        - ``meta``: diagnostic metadata.
    """
    x = _check_counts(offspring_dist)
    n = x.size

    if not (0.0 < float(superspreading_quantile) < 1.0):
        raise ValueError("superspreading_quantile must be in (0, 1).")
    q = float(superspreading_quantile)

    mean_Rt_emp = float(np.mean(x))
    pct_zero = float(np.mean(x == 0) * 100.0)

    # If mean = 0 (no transmissions), simple deterministic outputs
    if mean_Rt_emp == 0.0:
        return {
            "disp_k": np.nan,
            "meanRt": 0.0,
            "RtCI95": (np.nan, np.nan),
            "kCI95": (np.nan, np.nan),
            "superspreading_threshold": 0.0,
            "prop_80_percent_transmitters": 1.0,
            "pct_zero_transmitters": pct_zero,
            "pct_superspreaders": 0.0,
            "meta": {
                "n": int(n),
                "total_transmissions": 0,
                "bootstrap_kept": 0,
                "bootstrap_kept_R": 0,
                "bootstrap_kept_k": 0,
                "fit_method": "degenerate",
                "fit_notes": "mean=0",
                "quantile": q,
            }
        }

    # Fit NB safely (MLE if possible, otherwise MoM with warnings)
    R_hat, k_hat, method, notes = _fit_nb_safely(x, tol=tol)
    sse_thr = int(stats.poisson.ppf(q, R_hat))

    # Concentration metrics
    prop80 = float(_prop_for_80_percent(x))
    pct_superspreaders = float((x >= sse_thr).mean() * 100.0)

    # Bootstrap CIs
    R_CI = (np.nan, np.nan)
    k_CI = (np.nan, np.nan)
    kept = 0
    kept_R = 0
    kept_k = 0
    if isinstance(bootstrap, int) and bootstrap > 0 and n >= 2:
        boot = _bootstrap_nb(x, bootstrap=bootstrap, seed=bootstrap_seed, tol=tol)
        R_CI, k_CI = boot["R_CI95"], boot["k_CI95"]
        kept = boot["kept"]
        kept_R = boot["kept_R"]
        kept_k = boot["kept_k"]

    return {
        "disp_k": float(k_hat) if np.isfinite(k_hat) else np.nan,
        "meanRt": float(R_hat),
        "RtCI95": R_CI,
        "kCI95": k_CI,
        "superspreading_threshold": float(sse_thr),
        "prop_80_percent_transmitters": prop80,
        "pct_zero_transmitters": pct_zero,
        "pct_superspreaders": pct_superspreaders,
        "meta": {
            "n": int(n),
            "total_transmissions": int(x.sum()),
            "bootstrap_kept": int(kept),
            "bootstrap_kept_R": int(kept_R),
            "bootstrap_kept_k": int(kept_k),
            "fit_method": method,
            "fit_notes": notes,
            "quantile": q,
        }
    }


if __name__ == '__main__':
    # Example usage
    example_data = [0, 1, 0, 2, 3, 0, 0, 5, 1, 0, 0, 4, 2, 0]
    results = heterogeneity(example_data, bootstrap=100)
    for key, value in results.items():
        print(f"{key}: {value}")
