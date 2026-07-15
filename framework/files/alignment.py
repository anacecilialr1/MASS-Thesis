"""Interband lag: mapping observation time -> driver-frame time.

    t_drv = t_obs - lag(band)

The lag is the ONE genuinely awkward part of multiband, because it is a shift in
*time* rather than a transform of the *value*. Amplitude and offset are pure
value transforms and the emission absorbs them trivially; the lag is not.

Three modes, deliberately pluggable so you can climb the ladder:

  "zero"    lag = 0 everywhere. Stage 1. Also always true for the driver band.

  "oracle"  lag taken from the simulator. This is the DRIVER-FRAME MERGE: you hand
            the model the alignment it would not have on real data. Not a method --
            an UPPER BOUND. Report it as such; the gap between oracle and learned
            is itself a result.

  "learned" lag = A * (lambda_b / lambda_ref)**p, with A and p learned globally
            across the training set. Physically motivated: thin-disk reprocessing
            predicts tau_lag ~ lambda^(4/3), so p should drift toward ~1.33 if the
            data support it -- which makes p a READABLE diagnostic, not just a
            weight. Initialise p at 4/3 and let it move.

Note on honesty: "oracle" is only legitimate as a control. On real AGN data S, mu
and lag are unknown -- EzTaoX has to fit them by MCMC. A result that depends on
oracle alignment does not transfer to real light curves.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp


class Alignment(eqx.Module):
    """Produce a per-token lag from its band token."""

    mode: str = eqx.field(static=True)
    log_A: jax.Array          # learned: overall lag scale [days] at lambda_ref
    p: jax.Array              # learned: lag-wavelength exponent

    def __init__(self, mode: str = "zero", A_init: float = 1.0, p_init: float = 4.0 / 3.0):
        if mode not in ("zero", "oracle", "learned"):
            raise ValueError(f"unknown alignment mode: {mode}")
        self.mode = mode
        self.log_A = jnp.asarray(jnp.log(A_init))
        self.p = jnp.asarray(p_init)

    def __call__(self, band_feat, lag_oracle=None):
        """
        Parameters
        ----------
        band_feat : (T,) log(lambda_b / lambda_ref) per token (see tensorize.band_features)
        lag_oracle : (T,) or None
            True per-token lag; required iff mode == "oracle".

        Returns
        -------
        lag : (T,) days
        """
        if self.mode == "zero":
            return jnp.zeros_like(band_feat)

        if self.mode == "oracle":
            if lag_oracle is None:
                raise ValueError('alignment_mode="oracle" requires lag_oracle')
            return lag_oracle

        # learned: lag = A * (lambda/lambda_ref)^p = A * exp(p * band_feat)
        # band_feat == 0 for the driver band -> lag == A there. If you want the
        # driver pinned at lag=0 exactly, subtract A:  A*(exp(p*bf) - 1).
        return jnp.exp(self.log_A) * jnp.expm1(self.p * band_feat)

    def to_driver_time(self, t_obs, band_feat, lag_oracle=None):
        """t_obs -> t_drv. This is what the encoder and the transition should see."""
        return t_obs - self(band_feat, lag_oracle)
