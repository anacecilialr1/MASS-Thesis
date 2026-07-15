"""Latent NSF for AGN light curves: assembles the four modules.

Kiyohara+25 factorise Latent NSF into four separately-parameterised pieces (their
appendix reports them separately: stochastic flow, bridge model, decoder,
posterior). We keep that separation, because it is exactly what buys the staging:

    transition  p_theta(x_t | x_s, dt)   <- the NSF. BAND-AGNOSTIC. Never changes
                                            between Stage 1 and Stage 2.
    bridge      b_xi(x_j | x_i, x_k)     <- auxiliary for the flow loss AND the
                                            thing that actually fills gaps.
    emission    p_psi(o | x, band)       <- band scaling/offset + known yerr.
    encoder     q_phi(x | o_{<=t})       <- ingests the merged token stream.

Bands touch the last two only. Adding bands widens the encoder input and conditions
the emission; the flow/bridge code is untouched.

d_latent = 1 is the default and is a deliberate choice: the truth IS a 1-D DRW, so
the latent is directly comparable to the known driver. You can plot learned latent
vs true driver on the same axes. Do not raise it without a reason.

--- ON THE BRIDGE -----------------------------------------------------------------
Filling a gap means conditioning on BOTH endpoints -- that is the definition of a
bridge. The transition alone conditions only on the left endpoint, which is
forecasting: it drifts and its band widens monotonically. So the bridge is still
the sampler here. What changed vs. the internship is that it now operates on LATENT
driver states inside a model where the flow loss ties it to the transition, rather
than directly on noisy fluxes.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random

from jax_nsf.models.stochastic_flows import AffineCouplingStochasticFlow
from jax_nsf.models.bridge_models import AffineCouplingBridgeModel

from .alignment import Alignment
from .emission import make_emission


class Encoder(eqx.Module):
    """q_phi(x_t | o_{<=t}) over the merged token stream.

    Input per token: [y, yerr, band_feat, dt, (t)]
      - band_feat is what lets the encoder tell bands apart. Feed it ALWAYS, even
        in Stage 1 (it is identically 0 there) -- keeping the input width fixed is
        half of why Stage 2 is a flag.
      - yerr is fed so the encoder can down-weight noisy epochs itself.
      - absolute t only for non-autonomous systems; DRW is autonomous, so omit it
        (Kiyohara Eq. 15 makes the same distinction).

    Kiyohara run the GRU REVERSED in time then restore order, so the posterior at
    t_i sees the future too -- which matters a lot for interpolation. Keep that.
    """

    gru: eqx.nn.GRUCell
    head: eqx.nn.MLP
    d_latent: int = eqx.field(static=True)

    def __init__(self, d_latent: int, hidden: int, key, d_in: int = 4):
        k1, k2 = random.split(key)
        self.gru = eqx.nn.GRUCell(d_in, hidden, key=k1)
        self.head = eqx.nn.MLP(hidden, 2 * d_latent, hidden, 2, jax.nn.silu, key=k2)
        self.d_latent = d_latent

    def __call__(self, y, yerr, band_feat, t, mask):
        """(T,) inputs -> (mean, std) each (T, d_latent)."""
        dt = jnp.diff(t, prepend=t[:1])
        inp = jnp.stack([y, yerr, band_feat, dt], axis=-1)          # (T, 4)
        # zero out padded/dropped tokens so they cannot move the hidden state
        inp = inp * mask[:, None]

        def step(h, u):
            h = self.gru(u, h)
            return h, h

        h0 = jnp.zeros(self.gru.hidden_size)
        _, hs = jax.lax.scan(step, h0, inp[::-1])                   # reverse in time
        hs = hs[::-1]                                               # restore order
        out = jax.vmap(self.head)(hs)
        mean, raw = jnp.split(out, 2, axis=-1)
        return mean, jax.nn.softplus(raw) + 1e-4


class LatentNSF(eqx.Module):
    """The full model. Stage 1 and Stage 2 differ only by config."""

    encoder: Encoder
    emission: eqx.Module
    alignment: Alignment
    flow: AffineCouplingStochasticFlow          # p_theta(x_t | x_s, dt)
    bridge: AffineCouplingBridgeModel           # b_xi(x_j | x_i, x_k)

    def __init__(self, cfg, key):
        k_enc, k_emi, k_flow, k_bridge = random.split(key, 4)
        d = cfg.d_latent

        self.encoder = Encoder(d, cfg.gru_hidden, k_enc)
        self.emission = make_emission(cfg, d, k_emi)
        self.alignment = Alignment(cfg.alignment_mode)

        # NOTE: signatures below follow the jax_nsf API as used in the working
        # internship code. The package README warns "the API may change" -- if an
        # arg name errors, check the installed source; the surrounding logic holds.
        self.flow = AffineCouplingStochasticFlow(
            input_dim=d,
            autonomous_sde=cfg.autonomous_sde,
            mvn_width_size=cfg.mvn_width_size,
            mvn_depth=cfg.mvn_depth,
            mvn_activation=jax.nn.tanh,
            conditioner_width_size=cfg.conditioner_width_size,
            conditioner_depth=cfg.conditioner_depth,
            conditioner_activation=jax.nn.tanh,
            num_flow_layers=cfg.num_flow_layers,
            key=k_flow,
            affine_coupling_scale_fn="tanh_exp",
        )
        self.bridge = AffineCouplingBridgeModel(
            state_dim=d,
            autonomous_sde=cfg.autonomous_sde,
            mvn_width_size=cfg.mvn_width_size,
            mvn_depth=cfg.mvn_depth,
            mvn_activation=jax.nn.tanh,
            conditioner_width_size=cfg.conditioner_width_size,
            conditioner_depth=cfg.conditioner_depth,
            conditioner_activation=jax.nn.tanh,
            num_flow_layers=cfg.num_flow_layers,
            key=k_bridge,
            affine_coupling_scale_fn="tanh_exp",
        )

    # ---------------------------------------------------------------- encode
    def encode(self, y, yerr, band_feat, t_obs, mask, lag_oracle=None):
        """Observations -> latent posterior, on the DRIVER-FRAME time axis.

        Everything after this point (transition, bridge, triplets) lives in driver
        time. That is the whole trick: the dynamics see one process, not N bands.
        """
        t_drv = self.alignment.to_driver_time(t_obs, band_feat, lag_oracle)
        mean, std = self.encoder(y, yerr, band_feat, t_drv, mask)
        return mean, std, t_drv

    # ------------------------------------------------------------ transition
    def transition(self, x_s, t_s, t_t):
        """p_theta(x_t | x_s, t_t - t_s). Identity at dt=0 by construction."""
        return self.flow(x_init=x_s, t_init=t_s, t_final=t_t)

    def bridge_dist(self, x_i, t_i, x_k, t_k, t_j):
        """b_xi(x_j | x_i, x_k). Endpoint-conditioned -> this is the gap filler."""
        return self.bridge(x_init=x_i, t_init=t_i, x_final=x_k, t_final=t_k, t=t_j)


def check_transition_against_drw(model, tau, sigma, mu, x_s, dts, key, n=2000):
    """STAGE 0 VALIDATION -- do this before anything else.

    You know the true DRW transition in closed form:

        X_{i+1} | X_i ~ N( mu + e^{-dt/tau} (X_i - mu),  sigma^2 (1 - e^{-2 dt/tau}) )

    so you can check the LEARNED p_theta(x_t | x_s, dt) against it directly, as a
    function of dt. Kiyohara could not do this on Lorenz/mocap -- it is a free
    validation figure that your problem hands you. If the learned mean does not
    decay like e^{-dt/tau} and the variance does not saturate at sigma^2, stop:
    nothing downstream will be trustworthy.

    Returns (learned_mean, learned_var, true_mean, true_var) over dts.
    """
    a = jnp.exp(-dts / tau)
    true_mean = mu + a * (x_s - mu)
    true_var = sigma**2 * (1.0 - a**2)

    def one(dt, k):
        dist = model.transition(jnp.atleast_1d(x_s), 0.0, float(dt))
        xs = jax.vmap(lambda kk: dist.sample(kk))(random.split(k, n))
        return xs.mean(), xs.var()

    keys = random.split(key, dts.shape[0])
    learned_mean, learned_var = jax.vmap(one)(dts, keys)
    return learned_mean, learned_var, true_mean, true_var
