"""Observation model  p_psi(o | x, band).

This is where band physics lives, and it is the ONLY place (besides the encoder)
that knows bands exist. The transition kernel stays band-agnostic.

Two variants:

  structured=True  (default)
      mu = mu_b + S_b * x        with (S_b, mu_b) predicted from the band token.
      This is EXACTLY the EzTaoX model, Eq. 5: bands are scaled + offset views of
      one shared latent. It matches your generative truth, it is easy to learn,
      and S_b is readable -- you can plot learned S vs the MacLeod
      sigma ~ lambda^-0.479 prediction as a diagnostic figure.

  structured=False
      mu = MLP(x, band_token).   Strictly more flexible. Worth running once as an
      ablation to show the affine restriction costs nothing on DRW data (it should
      not -- the truth IS affine). If it helps, something is off in your sim.

Noise: you KNOW yerr per epoch (the simulator produced it), so use it directly
rather than learning a std. This removes a nuisance parameter the NSF paper had to
carry (their configs use "fixed std" or "trainable std" because their benchmarks
have no per-epoch errors). Optionally add a learned per-band jitter in quadrature,
which is what EzTaoX does on real ZTF data for residual pipeline error.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random


class AffineEmission(eqx.Module):
    """p_psi(o | x, band) = N(mu_b + S_b * x,  yerr^2 + jitter_b^2)."""

    scale_net: eqx.nn.MLP
    offset_net: eqx.nn.MLP
    log_jitter: jax.Array
    learn_jitter: bool = eqx.field(static=True)

    def __init__(self, d_latent: int, key, width: int = 32, depth: int = 2,
                 learn_jitter: bool = False, jitter_init: float = 1e-3):
        k1, k2 = random.split(key)
        # band token (scalar log-wavelength ratio) -> S_b and mu_b
        self.scale_net = eqx.nn.MLP(1, d_latent, width, depth, jax.nn.silu, key=k1)
        self.offset_net = eqx.nn.MLP(1, 1, width, depth, jax.nn.silu, key=k2)
        self.log_jitter = jnp.log(jnp.asarray(jitter_init))
        self.learn_jitter = learn_jitter

    def params(self, band_feat):
        """(S_b, mu_b) for one token. band_feat: scalar log(lambda_b/lambda_ref)."""
        bf = jnp.atleast_1d(band_feat)
        # softplus keeps S > 0; the driver band (band_feat == 0) should learn S ~ 1.
        S = jax.nn.softplus(self.scale_net(bf))
        mu = self.offset_net(bf)[0]
        return S, mu

    def mean(self, x, band_feat):
        S, mu = self.params(band_feat)
        return mu + jnp.sum(S * x)

    def sigma(self, yerr):
        if not self.learn_jitter:
            return yerr
        return jnp.sqrt(yerr**2 + jnp.exp(self.log_jitter) ** 2)

    def log_prob(self, o, x, band_feat, yerr):
        """Gaussian log-likelihood of one observation. vmap over tokens."""
        m = self.mean(x, band_feat)
        s = self.sigma(yerr)
        return -0.5 * (((o - m) / s) ** 2 + 2.0 * jnp.log(s) + jnp.log(2.0 * jnp.pi))

    def sample(self, key, x, band_feat, yerr):
        m = self.mean(x, band_feat)
        s = self.sigma(yerr)
        return m + s * random.normal(key, ())


class MLPEmission(eqx.Module):
    """Unstructured control: mu = MLP([x, band_feat]). Ablation only."""

    net: eqx.nn.MLP
    log_jitter: jax.Array
    learn_jitter: bool = eqx.field(static=True)

    def __init__(self, d_latent: int, key, width: int = 32, depth: int = 2,
                 learn_jitter: bool = False, jitter_init: float = 1e-3):
        self.net = eqx.nn.MLP(d_latent + 1, 1, width, depth, jax.nn.silu, key=key)
        self.log_jitter = jnp.log(jnp.asarray(jitter_init))
        self.learn_jitter = learn_jitter

    def mean(self, x, band_feat):
        return self.net(jnp.concatenate([jnp.atleast_1d(x), jnp.atleast_1d(band_feat)]))[0]

    def sigma(self, yerr):
        if not self.learn_jitter:
            return yerr
        return jnp.sqrt(yerr**2 + jnp.exp(self.log_jitter) ** 2)

    def log_prob(self, o, x, band_feat, yerr):
        m = self.mean(x, band_feat)
        s = self.sigma(yerr)
        return -0.5 * (((o - m) / s) ** 2 + 2.0 * jnp.log(s) + jnp.log(2.0 * jnp.pi))

    def sample(self, key, x, band_feat, yerr):
        return self.mean(x, band_feat) + self.sigma(yerr) * random.normal(key, ())


def make_emission(cfg, d_latent: int, key):
    kind = AffineEmission if cfg.structured_emission else MLPEmission
    return kind(d_latent, key, learn_jitter=cfg.learn_jitter)
