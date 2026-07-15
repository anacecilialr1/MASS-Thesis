"""Training objective:  L = L_beta-NELBO + lam * L_flow + beta_skip * L_skip
(Kiyohara+25 Eq. 18).

Three terms, three jobs:

  L_beta-NELBO  (Eq. 16)  reconstruct observations through the emission + keep the
                          posterior near the transition prior. THIS is what
                          separates measurement noise from process stochasticity --
                          the thing the internship conflated by training directly
                          on noisy fluxes.

  L_flow        (Eq. 10)  Chapman-Kolmogorov consistency between one-step+bridge and
                          two-step transitions, over triplets t_i < t_j < t_k.
                          Ports over from the internship almost verbatim; the only
                          change is that x now comes from the encoder, not from y.

  L_skip        (Eq. 17)  KL between the posterior at t_j and a DIRECT transition
                          from a much earlier t_i. THE LONG-GAP LEVER. The
                          internship's headline failure -- medians flattening across
                          wide gaps -- is exactly what you get when the objective
                          only ever sees local triplets. Set beta_skip=0 to
                          reproduce that failure on purpose as a baseline figure.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import random


def gaussian_kl(m_q, s_q, m_p, s_p):
    """KL( N(m_q, s_q^2) || N(m_p, s_p^2) ), elementwise."""
    return (jnp.log(s_p / s_q)
            + (s_q**2 + (m_q - m_p) ** 2) / (2.0 * s_p**2)
            - 0.5)


def nelbo_curve(model, y, yerr, band_feat, t_obs, mask, key, beta=1.0,
                lag_oracle=None):
    """beta-NELBO for ONE curve. vmap over the batch.

    TODO: the reconstruction term must be masked -- padded/dropped tokens are NOT
    observations and must not contribute. Multiply by `mask` and normalise by
    mask.sum(), not by T_max, or curves with more padding get a smaller loss.
    """
    mean, std, t_drv = model.encode(y, yerr, band_feat, t_obs, mask, lag_oracle)

    # reparameterised latent samples at every observation time
    eps = random.normal(key, mean.shape)
    x = mean + std * eps

    # --- reconstruction  E_q[ log p_psi(o | x) ] ---------------------------
    logp = jax.vmap(model.emission.log_prob)(y, x, band_feat, yerr)
    recon = jnp.sum(logp * mask) / jnp.maximum(jnp.sum(mask), 1.0)

    # --- latent KL against the NSF transition prior ------------------------
    # TODO: q(x_ti | o_<=ti) vs p_theta(x_ti | x_{ti-1}, dt). The prior is a FLOW,
    # so its density is not Gaussian in general -> either
    #   (a) MC-estimate the KL with the samples you already drew (cheap, noisy), or
    #   (b) use the flow's closed-form log_prob (it has one -- that is the whole
    #       selling point) and pair it with log q, which IS Gaussian.
    # Prefer (b): kl ~= mean over i of [ log q(x_i) - log p_theta(x_i | x_{i-1}) ].
    kl = 0.0  # <-- implement

    return -(recon - beta * kl)


def flow_loss_triplet(model, x_i, t_i, t_j, t_k, key):
    """L_flow for one triplet (Eqs. 8-10). Bidirectional.

    Port directly from the internship's flow_1_to_2_loss / flow_2_to_1_loss -- the
    algebra is unchanged. The ONLY difference: x_i is now a latent state from the
    encoder, not y[ii] from the raw light curve.
    """
    k1, k2, k3, k4 = random.split(key, 4)

    # 1 -> 2 : one-step flow + bridge   vs   two-step flow
    p_full = model.transition(x_i, t_i, t_k)
    x_k, lp_full = p_full.sample_and_log_prob(k1)
    q_mid = model.bridge_dist(x_i, t_i, x_k, t_k, t_j)
    x_j, lq_mid = q_mid.sample_and_log_prob(k2)
    p_a = model.transition(x_i, t_i, t_j)
    p_b = model.transition(x_j, t_j, t_k)
    l_1to2 = lp_full + lq_mid - p_a.log_prob(x_j) - p_b.log_prob(x_k)

    # 2 -> 1 : two-step flow   vs   one-step flow + bridge
    p_a2 = model.transition(x_i, t_i, t_j)
    x_j2, lp_a = p_a2.sample_and_log_prob(k3)
    p_b2 = model.transition(x_j2, t_j, t_k)
    x_k2, lp_b = p_b2.sample_and_log_prob(k4)
    q_mid2 = model.bridge_dist(x_i, t_i, x_k2, t_k, t_j)
    p_full2 = model.transition(x_i, t_i, t_k)
    l_2to1 = lp_a + lp_b - q_mid2.log_prob(x_j2) - p_full2.log_prob(x_k2)

    return l_1to2 + l_2to1


def skip_kl(model, mean, std, x, t_drv, mask, key, horizon=10, n_samples=10):
    """L_skip (Eq. 17): posterior at t_j vs a DIRECT transition from a far t_i.

    KL( q(x_tj | o_<=tj) || p_theta(x_tj | x_ti) )  for j sampled well beyond i.

    Cheap here precisely because NSF does arbitrary dt in ONE step -- with a
    solver-based latent SDE this term would require a full rollout, which is why
    Kiyohara can afford it and latent-SDE baselines cannot.

    TODO: sample (i, j) pairs with j - i in [2, horizon] from VALID indices only
    (use tensorize.valid_indices), then average.
    """
    raise NotImplementedError


def total_loss(model, batch, key, cfg):
    """L = NELBO + lam * L_flow + beta_skip * L_skip, averaged over the batch.

    TODO: apply the warmup ramp on (beta, lam, beta_skip) over cfg.warmup_steps --
    Kiyohara ramp these linearly and it matters for stability.
    TODO: Algorithm 1 does K=cfg.inner_steps_bridge inner updates of the BRIDGE
    params (xi) per outer step, before the main update. The bridge is auxiliary; if
    it lags the transition, the flow loss is measuring the wrong thing.
    """
    raise NotImplementedError
