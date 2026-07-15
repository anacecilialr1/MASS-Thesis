"""Configuration for AGN light-curve gap filling with Latent Neural Stochastic Flows.

The Stage 1 -> Stage 2 transition is a CONFIG CHANGE, not a refactor:

    Stage 0  (sanity)      bands="g",     band_keep_prob=0.0, sigma_phot=1e-4, gap_frac=0.0
    Stage 1  (single band) bands="g",     band_keep_prob=0.0
    Stage 2  (multiband)   bands="ugriz", band_keep_prob=0.5

Nothing downstream of this file needs editing to move between them. The transition
kernel p_theta(x_t | x_s, dt) is band-agnostic by construction; bands enter only
the encoder and the emission.
"""

from dataclasses import dataclass


@dataclass
class DataConfig:
    # --- which bands exist at all ------------------------------------------
    bands: str = "g"                  # Stage 1: "g"  |  Stage 2: "ugriz"
    driver_band: str = "g"            # reference band: S=1, lag=0, mu=0 BY DEFINITION.
                                      # (Single band cannot separate driver amplitude
                                      #  from band scaling -- fixing the reference is
                                      #  what makes the problem identifiable.)

    # --- driver DRW; None -> sample from the Yu+25 LogUniform ranges --------
    tau_driver: float | None = 100.0      # days  (Yu+25 fiducial g-band)
    sigma_driver: float | None = 0.112    # mag   (Yu+25 fiducial g-band)

    # --- baseline & cadence -------------------------------------------------
    t_max: float = 365.0 * 3          # 3-year baseline (Yu+25 Sets Two/Three)
    dt: float = 1.0                   # dense latent grid step [days]
    n_obs: int = 130                  # attempted epochs per band, before gaps
    gap_frac: float = 0.3
    sigma_phot: float = 0.01          # 10 mmag -- LSST system photometric accuracy.
                                      # NOTE: this is a calibration FLOOR, not the
                                      # brightness-dependent per-epoch error of
                                      # Ivezic+19 Eq.(5). Say so in the thesis.

    # --- interband lags (only bite once len(bands) > 1) ---------------------
    lag_mode: str = "zero"            # "zero" | "fixed" | "frac_tau" | "lambda43"
    lag_scale: float = 1.0            # fixed: days | frac_tau: fraction | lambda43: days at lambda_ref

    # --- dataset ------------------------------------------------------------
    n_curves: int = 500
    val_frac: float = 0.2
    seed: int = 0

    def diversity_required(self) -> bool:
        """Stage 2 needs lag/S diversity across curves, or the net memorises one
        configuration instead of learning a general band relation."""
        return len(self.bands) > 1


@dataclass
class TokenConfig:
    T_max: int = 200                  # padded stream length with ALL bands MERGED.
                                      # Stage 1 ~ n_obs ; Stage 2 ~ n_obs * n_bands.

    # ---- THE staging knob --------------------------------------------------
    band_keep_prob: float = 0.0       # 0.0 -> driver band only          (Stage 1)
                                      # 1.0 -> every band, always        (Stage 2)
                                      # 0.5 -> random subsets per curve  (Stage 2,
                                      #        robust to bands being unavailable
                                      #        at test time -- "band dropout")
    always_keep_driver: bool = True   # never drop the reference band


@dataclass
class ModelConfig:
    d_latent: int = 1                 # 1 == the driver itself. Keeps the learned
                                      # latent DIRECTLY comparable to the known
                                      # driver, which is a validation asset you do
                                      # not get on Lorenz/mocap. Raise only if the
                                      # model visibly needs slack.
    alignment_mode: str = "zero"      # "zero" | "oracle" | "learned"  (see alignment.py)
    structured_emission: bool = True  # affine in latent (== EzTaoX Eq. 5) vs free MLP
    learn_jitter: bool = False        # per-band jitter added in quadrature with yerr
                                      # (cf. EzTaoX on ZTF, Burke+21)

    # jax_nsf AffineCouplingStochasticFlow / AffineCouplingBridgeModel
    autonomous_sde: bool = True       # DRW is autonomous -> stationarity property holds
    num_flow_layers: int = 4
    mvn_width_size: int = 64
    mvn_depth: int = 2
    conditioner_width_size: int = 64
    conditioner_depth: int = 2

    # encoder q_phi
    gru_hidden: int = 64


@dataclass
class TrainConfig:
    steps: int = 2500
    batch_size: int = 16
    lr: float = 2e-3

    n_triplets: int = 128             # triplets sampled per curve per step
    min_gap: int = 2                  # min index separation within a triplet

    inner_steps_bridge: int = 3       # K in Kiyohara Algorithm 1
    beta: float = 1.0                 # latent KL weight (beta-NELBO, Eq. 16)
    lam: float = 1.0                  # flow-loss weight (Eq. 10)
    beta_skip: float = 0.3            # skip-ahead KL (Eq. 17) -- THE long-gap lever.
                                      # Set 0.0 to reproduce the internship failure
                                      # mode (median flattens across wide gaps).
    skip_horizon: int = 10
    n_skip_samples: int = 10
    warmup_steps: int = 200           # linearly ramp beta / lam / beta_skip
    seed: int = 0
