# agn_nsf — gap filling in AGN light curves with Latent Neural Stochastic Flows

```
agn_nsf/
├── config.py              ★ the ONLY file that changes between Stage 1 and Stage 2
├── data/
│   ├── generator.py         MultibandLCGenerator — drop in what we built
│   ├── dataset.py           N curves -> list of sims, train/val split
│   └── tensorize.py       ★ merged token stream, padding, band features, band dropout
├── models/
│   ├── alignment.py       ★ lag: zero | oracle | learned
│   ├── emission.py        ★ p_psi(o | x, band)  — structured affine == EzTaoX Eq.5
│   └── latent_nsf.py        encoder + transition + bridge + emission assembly
├── training/
│   ├── triplets.py          (i,j,k) sampling from valid indices  [port internship]
│   ├── losses.py            beta-NELBO + flow loss + skip-ahead KL
│   └── train.py             loop (Kiyohara Algorithm 1, the ♠ path)
├── inference/
│   └── gapfill.py           encode -> bridge across gaps -> decode
└── eval/
    ├── metrics.py           coverage, CRPS, error vs gap/tau, GP baseline
    └── plots.py

★ = where the Stage-1 → Stage-2 hooks live. Everything else is band-agnostic.
```

## The staging is a config change

```python
# Stage 0 — sanity: can the flow learn a DRW transition at all?
DataConfig(bands="g", sigma_phot=1e-4, gap_frac=0.0)
TokenConfig(band_keep_prob=0.0)
ModelConfig(alignment_mode="zero")
# -> run models.latent_nsf.check_transition_against_drw BEFORE anything else.

# Stage 1 — single band, realistic noise + gaps. A complete result on its own.
DataConfig(bands="g", sigma_phot=0.01, gap_frac=0.3)
TokenConfig(band_keep_prob=0.0)

# Stage 2 — multiband, robust to bands being unavailable.
DataConfig(bands="ugriz", lag_mode="lambda43")
TokenConfig(band_keep_prob=0.5)     # band dropout
ModelConfig(alignment_mode="learned")
```

No module edits. The transition kernel `p_theta(x_t | x_s, dt)` is band-agnostic by
construction — bands enter only the encoder and the emission — so widening from one
band to five widens an input, it does not restructure a model.

## Three invariants worth not breaking

1. **The driver band is the reference.** `S=1, lag=0, mu=0` for band `g`, by
   definition. Single band cannot separate the driver's amplitude from the band's
   scaling (σ=0.112 through S=1 and σ=0.224 through S=0.5 are the same
   observations), so fixing the reference is what makes the problem identifiable.
   Only *relative* S and lag mean anything, and only with ≥2 bands.

2. **One driver τ.** Per-band τ from MacLeod is a label, never a kernel parameter.
   See the note in the thesis draft: a shared driver has one timescale by
   construction, and the λ^0.17 lever is ~10% across u→r anyway.

3. **`alignment_mode="oracle"` is a control, not a method.** It hands the model an
   alignment it would not have on real data. Report it as an upper bound; the gap
   between oracle and learned is itself a result.

## Order of work

1. `tensorize.py` + a shape test (streams merge, sort, pad, mask correctly;
   `band_keep_prob=0` really does leave only `g`).
2. `check_transition_against_drw` — the learned transition vs the analytic DRW
   Gaussian. Free validation figure; if this fails, stop.
3. `losses.py` KL term + `train.py`, Stage 1.
4. `gapfill.py` + `eval/metrics.py`, with the **DRW-GP baseline** early — it tells
   you how much room there is before you spend weeks tuning.
5. Stage 2: flip the config.
