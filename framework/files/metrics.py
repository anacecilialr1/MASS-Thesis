"""Evaluation.

The internship's results were qualitative ("the band widens appropriately", "the
median tracks the truth"). That is a fine internship report and a weak thesis. You
have the latent driver in hand for every simulated curve, so everything below is
computable -- use it.

Metrics that matter, roughly in order of how much a referee will care:

  1. CALIBRATION / COVERAGE. Does the 68% credible band contain the true driver 68%
     of the time? Plot empirical coverage vs nominal (a reliability diagram). A
     probabilistic method that is not calibrated has no claim; this is the single
     most important figure and the internship did not have it.

  2. ERROR vs GAP SIZE. Bin by gap width in units of tau_DRW (NOT days -- the
     physics is scale-free in dt/tau, and it makes results comparable across the
     LogUniform tau range). Expect degradation as dt/tau -> 1; the question is
     whether beta_skip > 0 flattens that curve relative to beta_skip = 0.

  3. CRPS. Proper scoring rule; rewards sharpness AND calibration together, so it
     is the honest single number for comparing methods.

  4. BAND ABLATION. Same trained model, vary available bands at test time:
     {g} vs {g,r} vs {u,g,r} vs {ugriz}. This is the multiband claim, and because
     it is one model it cannot be confounded by training differences.

  5. BASELINES. You need them or there is no result:
       - linear / spline interpolation (the strawman the intro dismisses)
       - a GP with the TRUE DRW kernel and TRUE tau, sigma. This is the strong one:
         it is the Bayes-optimal filler for data your simulator literally generated
         from a DRW. Do NOT expect to beat it single-band -- you should not, and
         claiming otherwise would be suspicious. The interesting questions are
         (a) how close do you get, and (b) does multiband + learned alignment let
         you match a GP that has been HANDED tau and sigma. Frame it that way.
"""

from __future__ import annotations

import jax.numpy as jnp


def coverage(y_true, samples, level: float = 0.68):
    """Fraction of true values inside the central `level` credible interval.

    samples: (n_samples, M); y_true: (M,). Returns scalar.
    """
    lo = jnp.percentile(samples, 100 * (1 - level) / 2, axis=0)
    hi = jnp.percentile(samples, 100 * (1 + level) / 2, axis=0)
    return jnp.mean((y_true >= lo) & (y_true <= hi))


def reliability(y_true, samples, levels=jnp.linspace(0.05, 0.95, 19)):
    """Empirical vs nominal coverage -> the calibration figure."""
    return levels, jnp.array([coverage(y_true, samples, float(l)) for l in levels])


def crps(y_true, samples):
    """Continuous Ranked Probability Score, empirical form.

        CRPS = E|X - y| - 0.5 E|X - X'|

    Lower is better. Averaged over M points.
    """
    x = samples                                   # (n, M)
    term1 = jnp.mean(jnp.abs(x - y_true[None, :]), axis=0)
    term2 = 0.5 * jnp.mean(jnp.abs(x[:, None, :] - x[None, :, :]), axis=(0, 1))
    return jnp.mean(term1 - term2)


def error_vs_gap(y_true, samples, t, gap_edges, tau):
    """Bin reconstruction error by gap width in units of dt/tau.

    TODO: for each interior point, compute the width of the gap it sits in, divide
    by tau, bin, and report median |error| and coverage per bin.
    """
    raise NotImplementedError


def drw_gp_baseline(t_obs, y_obs, yerr, t_pred, tau, sigma):
    """Bayes-optimal reference: GP with the TRUE DRW kernel and TRUE parameters.

    k(dt) = sigma^2 exp(-|dt| / tau)

    Use celerite/tinygp (already a dependency via EzTao) rather than a dense
    solve -- O(N) vs O(N^3), and you will run this on every validation curve.

    Returns predictive (mean, std) at t_pred.
    """
    raise NotImplementedError
