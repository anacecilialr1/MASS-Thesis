"""Turn simulator output into padded token streams for batched training.

THE KEY DESIGN DECISION lives here.

Do NOT store per-band arrays as separate objects. Store ONE time-sorted stream of
tokens per curve, each token tagged with its band:

    (t, y, yerr, band_idx)   sorted by t, padded to T_max, with a validity mask.

Consequences:
  * single band == the same layout with every non-driver token masked out;
  * band dropout == flipping bits in the mask, band-wise;
  * the encoder sees one sequence regardless of how many bands exist;
  * the NSF transition never sees a band at all.

That is what makes Stage 2 a config flag instead of a rewrite.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from jax import random


# --------------------------------------------------------------------------- #
# building the stream
# --------------------------------------------------------------------------- #
def flatten_curve(sim: dict, bands: str, band_to_idx: dict[str, int]):
    """Merge every band of ONE simulated curve into a single time-sorted stream.

    Parameters
    ----------
    sim : dict
        Output of MultibandLCGenerator.makeMultiband().
    bands : str
        Bands to include, e.g. "ugriz". Must be keys of sim["bands"].
    band_to_idx : dict
        Stable band -> integer index map (share it across the whole dataset!).

    Returns
    -------
    t, y, yerr, band : ndarray, each shape (sum_b T_b,), sorted by t.
    """
    t, y, e, b = [], [], [], []
    for band in bands:
        d = sim["bands"][band]
        n = d["t_obs"].shape[0]
        t.append(np.asarray(d["t_obs"]))
        y.append(np.asarray(d["y_obs"]))
        e.append(np.asarray(d["yerr"]))
        b.append(np.full(n, band_to_idx[band], dtype=np.int32))

    t = np.concatenate(t)
    y = np.concatenate(y)
    e = np.concatenate(e)
    b = np.concatenate(b)

    order = np.argsort(t, kind="stable")   # stable -> reproducible tie-breaking
    return t[order], y[order], e[order], b[order]


def pad_stream(t, y, e, b, T_max: int):
    """Pad one stream to T_max and return the validity mask.

    yerr is padded with 1.0, NOT 0.0: the emission divides by yerr, and in JAX a
    0/0 or inf in a masked-out slot still poisons gradients through the whole
    batch. Masking the LOSS is not enough -- keep the padded values finite.
    """
    T = int(t.shape[0])
    if T > T_max:
        raise ValueError(f"stream length {T} exceeds T_max={T_max}; raise T_max")
    pad = T_max - T
    return (
        np.pad(t, (0, pad)),
        np.pad(y, (0, pad)),
        np.pad(e, (0, pad), constant_values=1.0),
        np.pad(b, (0, pad)),
        np.arange(T_max) < T,
    )


def build_dataset(sims: list[dict], bands: str, T_max: int) -> dict:
    """Stack many curves into (N, T_max) arrays.

    Returns a dict of jnp arrays:
        t, y, yerr : (N, T_max) float32
        band       : (N, T_max) int32
        mask       : (N, T_max) bool     -- padding validity (dropout applied later)
    plus per-curve metadata needed by alignment/emission/eval.
    """
    band_to_idx = {b: i for i, b in enumerate(bands)}
    T, Y, E, B, M = [], [], [], [], []
    lags, scales, mus = [], [], []

    for sim in sims:
        t, y, e, b = flatten_curve(sim, bands, band_to_idx)
        tp, yp, ep, bp, m = pad_stream(t, y, e, b, T_max)
        T.append(tp); Y.append(yp); E.append(ep); B.append(bp); M.append(m)
        # per-curve, per-band truth: (n_bands,) each. Oracle alignment / eval only.
        lags.append([sim["bands"][bd]["lag"] for bd in bands])
        scales.append([sim["bands"][bd]["S"] for bd in bands])
        mus.append([sim["bands"][bd]["mu"] for bd in bands])

    return dict(
        t=jnp.asarray(np.stack(T), jnp.float32),
        y=jnp.asarray(np.stack(Y), jnp.float32),
        yerr=jnp.asarray(np.stack(E), jnp.float32),
        band=jnp.asarray(np.stack(B), jnp.int32),
        mask=jnp.asarray(np.stack(M)),
        # --- truth / oracle channels: NEVER feed these to the model unless
        #     alignment_mode == "oracle". Kept for eval and ablations.
        lag_true=jnp.asarray(np.array(lags), jnp.float32),      # (N, n_bands)
        S_true=jnp.asarray(np.array(scales), jnp.float32),      # (N, n_bands)
        mu_true=jnp.asarray(np.array(mus), jnp.float32),        # (N, n_bands)
        band_to_idx=band_to_idx,
    )


# --------------------------------------------------------------------------- #
# band tokens
# --------------------------------------------------------------------------- #
def band_features(band_idx, wavelengths_by_idx, lambda_ref: float):
    """The band token fed to the encoder and emission.

    Use log(lambda_b / lambda_ref), NOT a categorical band id.

    Reason: both MacLeod relations are functions of lambda, so a wavelength token
    lets the network learn a CONTINUOUS relation (and in principle interpolate to
    bands it never saw), whereas a band id can only be memorised. The driver band
    maps to exactly 0.0, which is the right inductive bias -- it is the reference.
    """
    lam = jnp.asarray(wavelengths_by_idx)[band_idx]
    return jnp.log(lam / lambda_ref)


# --------------------------------------------------------------------------- #
# band dropout  <-- the staging knob
# --------------------------------------------------------------------------- #
def band_dropout(mask, band, key, keep_prob: float, n_bands: int,
                 driver_idx: int, always_keep_driver: bool = True):
    """Drop ENTIRE bands (not individual epochs) from the validity mask.

        keep_prob = 0.0  -> only the driver survives          == Stage 1
        keep_prob = 1.0  -> every band survives               == Stage 2
        keep_prob = 0.5  -> a random subset per curve         == Stage 2, robust

    Dropping whole bands (rather than scattered points) is the point: it makes the
    encoder's input distribution at train time match the test-time reality where a
    band is simply *not observed for this object*. A model trained only on ugriz
    is off-distribution when handed g alone.

    Shapes: mask (T,), band (T,) -> returns (T,).  vmap over the batch.
    """
    keep = random.bernoulli(key, keep_prob, (n_bands,))
    if always_keep_driver:
        keep = keep.at[driver_idx].set(True)
    return mask & keep[band]


def valid_indices(mask, T_max: int):
    """Valid indices first, then filler -- JAX-safe (no dynamic shapes).

    Returns (idx_full, K) with K = number of valid entries. Feed idx_full/K to the
    triplet sampler. Same trick as the internship code, kept because it works.
    """
    idx_full = jnp.nonzero(mask, size=T_max, fill_value=0)[0]
    K = jnp.sum(mask).astype(jnp.int32)
    return idx_full, K
