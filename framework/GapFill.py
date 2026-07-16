"""Gap filling with the trained bridge, plus the DRW-GP baseline.
"""
from __future__ import annotations
import numpy as np, jax, jax.numpy as jnp
from jax import random


def fillGaps(bridge, t_obs, y_obs, n_samples=200, grid_per_gap=20, seed=0):
    """Monte-Carlo reconstructions between consecutive observations.
    All inputs in NORMALIZED units. Returns (t_rec, s_rec (n_samples, M))."""
    key = random.PRNGKey(seed)
    t_all, s_all = [np.atleast_1d(t_obs[0])], [np.full((n_samples, 1), y_obs[0])]

    for i in range(len(t_obs) - 1):
        ta, tb = float(t_obs[i]), float(t_obs[i + 1])
        ya, yb = float(y_obs[i]), float(y_obs[i + 1])
        tg = np.linspace(ta, tb, grid_per_gap + 2)[1:-1]
        if tg.size:
            key, sub = random.split(key)
            ks = random.split(sub, n_samples * tg.size).reshape(n_samples, tg.size, -1)

            def one(k, t):
                return bridge(x_init=jnp.array([ya]), t_init=jnp.array(ta),
                              x_final=jnp.array([yb]), t_final=jnp.array(tb),
                              t=t).sample(k)[0]
            path = jax.vmap(lambda kp: jax.vmap(one)(kp, jnp.asarray(tg)))(ks)
            t_all.append(tg); s_all.append(np.asarray(path))
        t_all.append(np.atleast_1d(tb)); s_all.append(np.full((n_samples, 1), yb))

    return np.concatenate(t_all), np.concatenate(s_all, axis=1)


def drwGPBaseline(t_obs, y_obs, yerr, t_pred, tau, sigma):
    """Bayes-optimal reference: a GP with the TRUE DRW kernel and TRUE parameters.
    This is the ceiling since the data were generated from exactly this process
    and it has been handed the parameters, just see how close we get.
    All inputs in the SAME units (normalized, consistently).
    """
    t_obs, y_obs = np.asarray(t_obs, float), np.asarray(y_obs, float)
    k = lambda a, b: sigma**2 * np.exp(-np.abs(a[:, None] - b[None, :]) / tau)
    L = np.linalg.cholesky(k(t_obs, t_obs) + np.diag(np.asarray(yerr, float)**2))
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_obs))
    Ks = k(np.asarray(t_pred, float), t_obs)
    v = np.linalg.solve(L, Ks.T)
    return Ks @ alpha, np.sqrt(np.maximum(sigma**2 - np.sum(v**2, axis=0), 0.0))


def gapWidths(t_obs, t_eval):
    """Width of the observational gap containing each t_eval; NaN outside the
    observed span (extrapolation -> np.isfinite() doubles as the eval mask)."""
    t_obs, t_eval = np.asarray(t_obs), np.asarray(t_eval)
    i = np.searchsorted(t_obs, t_eval)
    inside = (i > 0) & (i < t_obs.size)
    w = np.full(t_eval.shape, np.nan)
    w[inside] = t_obs[i[inside]] - t_obs[i[inside] - 1]
    return w


def coverage(y_true, samples, level=0.68):
    lo = np.percentile(samples, 100 * (1 - level) / 2, axis=0)
    hi = np.percentile(samples, 100 * (1 + level) / 2, axis=0)
    return float(np.mean((y_true >= lo) & (y_true <= hi)))


def plotGapFill(t_true, y_true, t_obs, y_obs, yerr, t_rec, recons, ax=None,
                label="NSF bridge", xlabel="time / $t_{max}$"):
    """
    recons : (n_samples, M) Monte-Carlo samples, OR a (median, lo, hi) tuple.

    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 3.4))
    if isinstance(recons, tuple):
        med, lo, hi = recons
    else:
        lo, med, hi = np.percentile(np.asarray(recons), [16, 50, 84], axis=0)

    ax.plot(t_true, y_true, color="0.75", lw=0.9, zorder=1, label="latent driver (truth)")
    ax.fill_between(t_rec, lo, hi, color="tab:blue", alpha=0.22, lw=0, zorder=2,
                    label=rf"{label} $\pm1\sigma$")
    ax.plot(t_rec, med, color="tab:red", lw=1.0, zorder=3, label=f"{label} median")
    ax.errorbar(t_obs, y_obs, yerr=yerr, fmt="o", ms=2.2, lw=0.7, color="k",
                capsize=0, zorder=4, label="observed")
    ax.set_xlabel(xlabel); ax.set_ylabel(r"$\Delta$mag / $\sigma$")
    ax.set_xlim(float(t_true[0]), float(t_true[-1]))
    ax.invert_yaxis()
    ax.legend(ncol=4, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    return ax
