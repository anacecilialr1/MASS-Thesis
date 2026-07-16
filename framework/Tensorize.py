"""Pad ragged single-band light curves into batched arrays, and sample triplets."""
from __future__ import annotations
import numpy as np, jax.numpy as jnp
from jax import random


def padDataset(curves, T_max=None):
    """List of makeSingleBand dicts -> (N, T_max) arrays + validity mask."""
    if T_max is None:
        T_max = max(c["t_obs"].size for c in curves)
    N = len(curves)
    t = np.zeros((N, T_max), np.float32)
    y = np.zeros((N, T_max), np.float32)
    e = np.ones((N, T_max), np.float32)      # 1.0, NOT 0.0: we divide by yerr, and a
    m = np.zeros((N, T_max), bool)           # 0/0 in a masked slot poisons gradients
    for i, c in enumerate(curves):
        T = c["t_obs"].size
        if T > T_max:
            raise ValueError(f"curve {i} has {T} epochs > T_max={T_max}")
        t[i, :T], y[i, :T], e[i, :T], m[i, :T] = c["t_obs"], c["y_obs"], c["yerr"], True
    return dict(t=jnp.asarray(t), y=jnp.asarray(y), yerr=jnp.asarray(e), mask=jnp.asarray(m))


def validIndices(mask, T_max):
    """Valid indices first, then filler. JAX-safe (static shape)."""
    return jnp.nonzero(mask, size=T_max, fill_value=0)[0], jnp.sum(mask).astype(jnp.int32)


def sampleTriplets(key, idx_full, K, n=64, min_gap=2):
    """
    Sample (i, j, k), i < j < k, from the valid indices.

    The SPAN k-i is drawn LOG-uniformly, so triplets cover short and long dt alike.
    The internship sampler drew j-i and k-j from narrow fixed ranges (min_gap..+6,
    min_gap..+10), so k-i never exceeded ~20 of ~85 indices: the objective only ever
    constrained short transitions, which is exactly why the reconstructed median
    flattened across wide gaps.
    """
    k1, k2, k3 = random.split(key, 3)
    K = K.astype(jnp.int32)
    lo_span = 2 * min_gap

    i_pos = random.randint(k1, (n,), 0, jnp.maximum(K - lo_span, 1))
    room = jnp.maximum(K - 1 - i_pos, lo_span)
    u = random.uniform(k2, (n,))
    span = jnp.round(lo_span * jnp.exp(u * jnp.log(room / lo_span))).astype(jnp.int32)
    k_pos = jnp.minimum(i_pos + span, K - 1)

    v = random.uniform(k3, (n,))
    j_pos = i_pos + min_gap + jnp.floor(v * jnp.maximum(k_pos - i_pos - lo_span, 1)).astype(jnp.int32)
    j_pos = jnp.clip(j_pos, i_pos + min_gap, jnp.maximum(k_pos - min_gap, i_pos + min_gap))
    return idx_full[i_pos], idx_full[j_pos], idx_full[k_pos]


def loadDataset(path, t_scale=None, y_scale=None):
    """
    Read an .npz written by LightCurves.saveDataset. Never imports eztao.
    NORMALISES BY DEFAULT, and this is not optional. The flow conditions on
    (x, dt); feeding it dt ~ 1000 with x ~ 0.1 is badly conditioned and it will
    silently learn nonsense -- the transition mean grows instead of decaying, the
    sd grows instead of saturating, and the training loss sits ~50 instead of ~2.

    tau comes back already in normalised units, so the Stage 0 check against the
    analytic DRW needs no bookkeeping.
    """
    d = np.load(path)
    t_scale = float(d["t_true"][-1]) if t_scale is None else float(t_scale)
    y_scale = float(np.median(d["sigma"])) if y_scale is None else float(y_scale)

    out = dict(
        t=jnp.asarray(d["t_obs"] / t_scale), y=jnp.asarray(d["y_obs"] / y_scale),
        yerr=jnp.asarray(d["yerr"] / y_scale), mask=jnp.asarray(d["mask"]),
        t_true=jnp.asarray(d["t_true"] / t_scale),
        y_true=jnp.asarray(d["y_true"] / y_scale),
        tau=jnp.asarray(d["tau"] / t_scale),        # -> tau_n
        sigma=jnp.asarray(d["sigma"] / y_scale),    # -> 1.0
        t_scale=t_scale, y_scale=y_scale,
    )
    
    ymax = float(jnp.max(jnp.abs(out["y"][out["mask"]])))
    tmax = float(jnp.max(out["t"]))
    if not (0.5 < ymax < 20.0):
        raise ValueError(f"y looks unnormalised (max|y| = {ymax:.3g}); expected O(1)")
    if not (0.1 < tmax < 10.0):
        raise ValueError(f"t looks unnormalised (max t = {tmax:.3g}); expected O(1)")
    return out
