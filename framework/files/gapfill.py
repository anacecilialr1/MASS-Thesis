"""Probabilistic gap filling with the trained Latent NSF.

The recipe:

  1. encode ALL available observations (whatever bands you have) -> latent
     posterior over the driver, on the driver-frame time axis;
  2. find gaps: consecutive driver-frame times with large dt;
  3. inside each gap, sample the BRIDGE conditioned on the latent states at the two
     endpoints, on a dense grid;
  4. decode to whichever band you want to reconstruct, via the emission;
  5. repeat -> Monte Carlo median + credible band.

Two things worth noticing:

  * The endpoints are POSTERIOR SAMPLES, not observed fluxes. So endpoint noise
    propagates into the reconstruction instead of being silently treated as truth.
    That is the concrete payoff of going latent, and it should show up as a
    credible band that does not pinch to zero width at observed epochs -- the
    internship's did, which was wrong.

  * With multiband, a gap in band i is often NOT a gap in driver time, because
    band j has epochs there. Once merged in driver frame, cross-band support is
    automatic: step 2 simply finds fewer/shorter gaps. Nothing special to code.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import random


def find_gaps(t_drv, mask, min_gap_days: float):
    """Consecutive valid driver-frame times separated by more than min_gap_days.

    Returns index pairs (i, k) bracketing each gap. NOTE: t_drv is sorted by
    OBSERVATION time; after subtracting per-band lags it may no longer be sorted.
    Re-sort on the valid subset before diffing.
    """
    raise NotImplementedError


def fill_gaps(model, y, yerr, band_feat, t_obs, mask, target_band_feat,
              key, n_samples=200, grid_per_gap=40, lag_oracle=None):
    """Monte Carlo reconstructions for one curve.

    Returns
    -------
    t_rec  : (M,)             driver-frame grid (observed + interior points)
    s_rec  : (n_samples, M)   reconstructions decoded into the target band
    """
    mean, std, t_drv = model.encode(y, yerr, band_feat, t_obs, mask, lag_oracle)

    # TODO
    #  - draw n_samples latent endpoint sets from N(mean, std^2) at valid times
    #  - for each consecutive valid pair (a, b): build an interior grid, sample
    #    model.bridge_dist(x_a, t_a, x_b, t_b, t) at each interior t
    #  - decode: model.emission.mean(x, target_band_feat)
    #  - concatenate; return
    #
    # Reuse the internship's fill_gaps structure -- it was sound. The changes are
    # (a) endpoints are latent samples not raw y, (b) the decode step at the end.
    raise NotImplementedError


def quantiles(s_rec, qs=(16, 50, 84)):
    return jnp.percentile(s_rec, jnp.asarray(qs), axis=0)
