"""Gap filling with the trained bridge, plus the DRW-GP baseline.
"""
from __future__ import annotations
import numpy as np, jax, jax.numpy as jnp
from jax import random


def fillGaps(bridge, t_obs, y_obs, n_samples=2000, grid_per_gap=40, seed=0):
    """Monte-Carlo reconstructions between observations,
       returns (t_rec, s_rec (n_samples, M))."""
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
                label="NSF bridge", xlabel="time / $t_{max}$",
                tau=None, sigma=None, t_scale=None):
    """
    recons : (n_samples, M) Monte-Carlo samples  -> nested 68/95/99.7% percentile bands
             OR a (median, sd) tuple             -> Gaussian mu +- {1,2,3} sd (GP path)
    tau, sigma : driver params of THIS curve, printed in the truth label. tau is
                 shown in physical days if t_scale is given, else in normalized units.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 3.4))

    # truth of the latent driver
    if tau is not None and sigma is not None:
        tau_disp = tau * t_scale if t_scale is not None else tau
        unit = "d" if t_scale is not None else r"$t_{max}$"
        truth_lab = rf"latent driver: $\sigma={sigma:.3f}$, $\tau={tau_disp:.0f}$ {unit}"
    else:
        truth_lab = "latent driver (truth)"
    ax.plot(t_true, y_true, color="0.75", lw=0.9, zorder=1, label=truth_lab)

    if isinstance(recons, tuple):   # GP: Gaussian
        med, sd = recons
        for k, a in [(3, 0.15), (2, 0.20), (1, 0.30)]:
            ax.fill_between(t_rec, med - k*sd, med + k*sd, color="cornflowerblue",
                            alpha=a, lw=0, zorder=2)
    else:                         # NSF: use percentiles
        s = np.asarray(recons)
        for plo, phi, a in [(0.15, 99.85, 0.15), (2.5, 97.5, 0.20), (16, 84, 0.30)]:
            blo, bhi = np.percentile(s, [plo, phi], axis=0)
            ax.fill_between(t_rec, blo, bhi, color="cornflowerblue", alpha=a, lw=0, zorder=2)
        med = np.percentile(s, 50, axis=0)

    ax.plot(t_rec, med, color="tab:red", lw=1.0, zorder=3, label=f"{label} median")
    ax.errorbar(t_obs, y_obs, yerr=yerr, fmt="o", ms=2.2, lw=0.7, color="k",
                capsize=0, zorder=4, label="observed")

    for a, lb in [(0.30, rf"{label} 68%"), (0.20, "95%"), (0.15, "99.7%")]:
        ax.plot([], [], color="cornflowerblue", lw=6, alpha=a, label=lb)

    ax.set_xlabel(xlabel); ax.set_ylabel(r"$\Delta$mag / $\sigma$")
    ax.set_xlim(float(t_true[0]), float(t_true[-1]))
    ax.invert_yaxis()
    ax.legend(ncol=3, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    return ax

def evalCurve(bridge, D, V, i, tau_n=None, sig_n=None,
              n_samples=200, grid_per_gap=12, seed=0, near_tol=1e-6):
    """
    Gap-fill validation curve i and score it only on GAP INTERIORS.
      1. Per-curve (tau, sigma): the GP baseline gets THIS curve's parameters, read
         from V["tau"][i] / V["sigma"][i]. Pass tau_n/sig_n to override.
      2. Interior mask: exclude the pinned observation epochs, where both models nail
         the value and the band collapses to ~yerr. interior = inside the observed
         span (gapWidths finite) AND not sitting on an observation.
    """
    m = np.asarray(V["mask"][i])
    t_o = np.asarray(V["t"][i])[m]; y_o = np.asarray(V["y"][i])[m]; e_o = np.asarray(V["yerr"][i])[m]
    t_tr = np.asarray(V["t_true"]);  y_tr = np.asarray(V["y_true"][i])

    tau_i = float(V["tau"][i]) if tau_n is None else tau_n          # FIX 1
    sig_i = float(V["sigma"][i]) if sig_n is None else sig_n

    t_rec, s_rec = fillGaps(bridge, t_o, y_o, n_samples=n_samples,
                            grid_per_gap=grid_per_gap, seed=seed)
    mu, sd = drwGPBaseline(t_o, y_o, e_o, t_rec, tau_i, sig_i)
    y_ref = np.interp(t_rec, t_tr, y_tr)

    w = gapWidths(t_o, t_rec)                                       # NaN outside span
    near = np.min(np.abs(t_rec[:, None] - t_o[None, :]), axis=1) < near_tol
    interior = np.isfinite(w) & ~near                              # FIX 2

    rng = np.random.default_rng(seed)
    gp_s = mu[None, :] + sd[None, :] * rng.normal(size=(s_rec.shape[0], mu.size))

    def score(samples, med):
        return (coverage(y_ref[interior], samples[:, interior]),
                float(np.sqrt(np.mean((med[interior] - y_ref[interior]) ** 2))))

    nsf_cov, nsf_rmse = score(s_rec, np.median(s_rec, 0))
    gp_cov, gp_rmse = score(gp_s, mu)
    return dict(t_rec=t_rec, s_rec=s_rec, mu=mu, sd=sd, y_ref=y_ref,
                interior=interior, gap_tau=w / tau_i,
                nsf_cov=nsf_cov, nsf_rmse=nsf_rmse, gp_cov=gp_cov, gp_rmse=gp_rmse,
                tau=tau_i, sigma=sig_i)
