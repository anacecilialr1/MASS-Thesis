"""NSF (flow + bridge) trained on single-band observations.

Total loss, Kiyohara Eq. (11):      L = L_NLL + lambda * L_flow

The internship's triplet_loss used only y[ii] and not L_NLL 
"""
from __future__ import annotations
import jax, jax.numpy as jnp, numpy as np, equinox as eqx, optax
from jax import random
from jax_nsf.models.stochastic_flows import AffineCouplingStochasticFlow
from jax_nsf.models.bridge_models import AffineCouplingBridgeModel
from Tensorize import validIndices, sampleTriplets


def makeModels(key, d=1, width=64, depth=2, layers=4):
    k1, k2 = random.split(key)
    kw = dict(mvn_width_size=width, mvn_depth=depth, mvn_activation=jax.nn.tanh,
              conditioner_width_size=width, conditioner_depth=depth,
              conditioner_activation=jax.nn.tanh, num_flow_layers=layers,
              affine_coupling_scale_fn="tanh_exp", autonomous_sde=True)
    return (AffineCouplingStochasticFlow(input_dim=d, key=k1, **kw),
            AffineCouplingBridgeModel(state_dim=d, key=k2, **kw))


def nllTriplet(flow, y, t, ii, jj, kk):
    """-log p_theta over the three pairs in a triplet. The i->k pair is the LONG
    one: it is what teaches long-dt transitions directly from data."""
    xi, xj, xk = y[ii][None], y[jj][None], y[kk][None]
    lp = (flow(x_init=xi, t_init=t[ii], t_final=t[jj]).log_prob(xj)
          + flow(x_init=xj, t_init=t[jj], t_final=t[kk]).log_prob(xk)
          + flow(x_init=xi, t_init=t[ii], t_final=t[kk]).log_prob(xk))
    return -lp / 3.0


def flowLoss(flow, bridge, y, t, ii, jj, kk, key):
    """Bidirectional Chapman-Kolmogorov consistency, Kiyohara Eqs. (8)-(10)."""
    x_i = y[ii][None]
    ti, tj, tk = t[ii], t[jj], t[kk]
    k1, k2, k3, k4 = random.split(key, 4)

    p_full = flow(x_init=x_i, t_init=ti, t_final=tk)
    x_f, lp_full = p_full.sample_and_log_prob(k1)
    x_m, lq_mid = bridge(x_init=x_i, t_init=ti, x_final=x_f,
                         t_final=tk, t=tj).sample_and_log_prob(k2)
    l12 = (lp_full + lq_mid
           - flow(x_init=x_i, t_init=ti, t_final=tj).log_prob(x_m)
           - flow(x_init=x_m, t_init=tj, t_final=tk).log_prob(x_f))

    x_m2, lp_a = flow(x_init=x_i, t_init=ti, t_final=tj).sample_and_log_prob(k3)
    x_f2, lp_b = flow(x_init=x_m2, t_init=tj, t_final=tk).sample_and_log_prob(k4)
    l21 = (lp_a + lp_b
           - bridge(x_init=x_i, t_init=ti, x_final=x_f2, t_final=tk, t=tj).log_prob(x_m2)
           - flow(x_init=x_i, t_init=ti, t_final=tk).log_prob(x_f2))
    return l12 + l21


@eqx.filter_value_and_grad
def batchObjective(params, tB, yB, mB, key, lam=0.1, n_trip=32):
    flow, bridge = params
    B, T = tB.shape
    keys = random.split(key, B)

    def oneCurve(t, y, m, k):
        idx, K = validIndices(m, T)
        def ok():
            k1, k2 = random.split(k)
            ii, jj, kk = sampleTriplets(k1, idx, K, n=n_trip)
            tks = random.split(k2, n_trip)
            nll = jax.vmap(lambda a, b, c: nllTriplet(flow, y, t, a, b, c))(ii, jj, kk)
            fl = jax.vmap(lambda a, b, c, kk_: flowLoss(flow, bridge, y, t, a, b, c, kk_))(ii, jj, kk, tks)
            return jnp.mean(nll) + lam * jnp.mean(fl)
        return jax.lax.cond(K < 6, lambda: 0.0, ok)

    return jnp.mean(jax.vmap(oneCurve)(tB, yB, mB, keys))


def train(flow, bridge, D, key, steps=400, batch=16, lr=2e-3, lam=0.1, n_trip=32):
    params = (flow, bridge)
    opt = optax.adam(lr)
    st = opt.init(eqx.filter(params, eqx.is_inexact_array))
    N = D["t"].shape[0]

    @eqx.filter_jit
    def step(params, st, key):
        key, k1, k2 = random.split(key, 3)
        i = random.choice(k1, N, (batch,), replace=False)
        loss, g = batchObjective(params, D["t"][i], D["y"][i], D["mask"][i], k2,
                                 lam=lam, n_trip=n_trip)
        u, st = opt.update(g, st, eqx.filter(params, eqx.is_inexact_array))
        return eqx.apply_updates(params, u), st, key, loss

    hist = []
    for s in range(steps):
        params, st, key, loss = step(params, st, key)
        hist.append(float(loss))
        if (s + 1) % 100 == 0:
            print(f"  step {s+1:5d} | loss {np.mean(hist[-100:]):8.4f}")
    return params, np.array(hist)


def checkTransition(flow, tau_n, sigma_n, x_s, dts, key, n=4000):
    """STAGE 0 VALIDATION. The true DRW transition is known in closed form:
        X_t | X_s ~ N( e^{-dt/tau} X_s ,  sigma^2 (1 - e^{-2 dt/tau}) )
    so the learned p_theta can be checked against it directly. All quantities in NORMALISED units."""
    a = np.exp(-dts / tau_n)
    true_m, true_v = a * x_s, sigma_n**2 * (1.0 - a**2)
    lm, lv = [], []
    for dt, k in zip(dts, random.split(key, len(dts))):
        d = flow(x_init=jnp.array([x_s]), t_init=jnp.array(0.0), t_final=jnp.array(float(dt)))
        xs = np.asarray(d.sample_n(k, n)).ravel()
        lm.append(xs.mean()); lv.append(xs.var())
    return np.array(lm), np.array(lv), true_m, true_v
