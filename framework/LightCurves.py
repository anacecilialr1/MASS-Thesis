import numpy as np
# the CARMA(1,0) kernel is equivalent to the damped random walk (DRW) kernel.
from eztao.carma import DRW_term 
# to simulate CARMA processes at fixed input time stamps.
from eztao.ts import gpSimByTime
import matplotlib.pyplot as plt

# A&A/ApJ conventions: ticks inward on all four sides, minors on, no legend
# frame, serif to match the LaTeX body text.

AA_STYLE = {
    "font.family": "serif", "mathtext.fontset": "dejavuserif",
    "axes.linewidth": 0.8, "lines.linewidth": 1.0,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "legend.frameon": False, "figure.dpi": 130,
}


class MultibandLCGenerator:
    """
    Multiband light curve generator using EzTao
    """
    # Limits for the parameter spaces from Yu et al. (2025)
    # tau: damping (characteristic) time-scale [days], it goes up to decades but
    # should lower the limits to around (30, 300)?
    low_tau = 0.01
    high_tau = 500

    # sigma: DRW stationary standard deviation (EzTao amplitude) [mag]
    low_sigma = 0.02
    high_sigma = 0.2

    # LSST bandpasses
    allowed_bands = "ugriz"
    wavelengths = {"u": 3520.0,
                   "g": 4800.0,
                   "r": 6250.0,
                   "i": 7690.0,
                   "z": 9110.0,
                   }

    # such a high SNR produces neglible noise in the lightcurves, the realistic noise
    # is added later with AddNoise()
    snrs = {"u": 1e6,
            "g": 1e6,
            "r": 1e6,
            "i": 1e6,
            "z": 1e6,
            }

    
    # make sure the observation baseline covers a significant fraction of the characteristic timescale
    def __init__(self, # initiation for any newly created instance of this class
                 bands: str = "ugriz", # don't use MultibandLCGenerator.variables since class is not created yet here
                 t_max: float = 365,*3 # days
                 dt: float = 1, # time step
                 driver_band: str | None = "g",
                 tau_driver: float | None = 100., # days, KEEP FIXED FOR NOW
                 sigma_driver: float | None = 0.112,# mag
                 snr_driver: float | None = 1e6, # these driver values correspond to the reference g-band taken from Yu et al. (2025)
                 rng: np.random.Generator | None = None,
                 lc_seed: int = 0,
                 log_flux: bool = True, # whether the flux/y vals are in astronomical magnitude
                 ): 

        
        if not all(c in MultibandLCGenerator.allowed_bands for c in bands):
            raise ValueError("Bands should be among LSST bands ('ugriz')")

        if driver_band not in bands:
            raise ValueError("Reference_band must be included in bands")
            
        self.rng = rng if rng is not None else np.random.default_rng(lc_seed)
        
        # time grid
        self.t_max = t_max
        self.dt = dt
        self.t_true = np.arange(0, t_max + dt, dt)
        self.bands = bands

        self.driver_band = driver_band

        if tau_driver is None:
            self.tau_driver = np.exp(self.rng.uniform(np.log(self.low_tau),
                                                      np.log(self.high_tau)))            
        else:
            self.tau_driver = tau_driver

        if sigma_driver is None:
            self.sigma_driver = np.exp(self.rng.uniform(np.log(self.low_sigma),
                                                        np.log(self.high_sigma)))
        else:
            self.sigma_driver = sigma_driver

        if snr_driver is None:
            self.snr_driver = self.snrs[self.driver_band]
        else:
            self.snr_driver = snr_driver

        self.seed = lc_seed
        self.log_flux = log_flux


    def describe(self):
        print(f"Simulated bands = {len(self.bands)}: {self.bands}")
        print(f"Driver band = {self.driver_band}")
        print(f"Driver sigma = {self.sigma_driver}")
        print(f"Driver tau = {self.tau_driver}")
        print(f"Driver snr = {self.snr_driver}")
        print(f"Baseline = {self.t_max} d = {self.t_max/self.tau_driver:.1f} tau")

    def checkRegime(self, n_obs: int = 130, gap_frac: float = 0.35, n_gaps: int = 3):
        """
        Report where this configuration sits in dt / tau, the only variable that
        controls how hard gap filling is, since the DRW forgets as exp(-dt/tau).
        Call it deliberately; it is not wired into makeSingleBand, which runs once
        per curve.

        Both thresholds below are quoted in the literature as tau-RECOVERY biases
        (Kozlowski 2017; Hu et al. 2024). They matter here for a different reason:
        together they bracket the range in which a gap carries any information.
        """
        # cadence is independent of gap_frac: gaps remove time and epochs in the
        # same proportion, so the two factors of (1 - gap_frac) cancel. This is the
        # MEAN; spacings are exponential, so the median comes out at ln2 = 0.69x.
        cadence = self.t_max / n_obs
        mean_gap = gap_frac * self.t_max / max(n_gaps, 1)

        print(f"mean cadence = {cadence:6.2f} d = {cadence / self.tau_driver:.3f} tau")
        print(f"mean gap     = {mean_gap:6.2f} d = {mean_gap / self.tau_driver:.2f} tau")
        print(f"baseline     = {self.t_max:6.0f} d = {self.t_max / self.tau_driver:.1f} tau")
        print(f"epochs kept  ~ {int(n_obs * (1 - gap_frac))} of {n_obs} attempted")

        if self.tau_driver < 5 * cadence:
            print("  ! tau < 5x cadence: consecutive epochs are essentially "
                  "uncorrelated, so there is nothing to interpolate between them "
                  "-- the model can only return the prior.")
        if self.t_max < 10 * self.tau_driver:
            print("  ! baseline < 10 tau: the driver barely decorrelates, so every "
                  "gap is easy and the training set is trivial.")
        
    def computeDRWparams(self):
        """
        Compute tau_D​RW and sigma_D​RW for each band using
        the wavelength scalings from MacLeod et al. (2010):

            tau_D​RW ∝ lambda^0.17
            sigma_D​RW ∝ lambda^-0.479

        The driver band is used as the reference band. Note that to keep the same underlying
        physical process, all bands have to share the same tau, so the one calculated here is
        never inserted into the kernels.
        """
        #taus = {}
        sigmas = {}

        #taus[self.driver_band] = self.tau_driver
        #sigmas[self.driver_band] = self.sigma_driver 

        lambda_ref = self.wavelengths[self.driver_band]

        for band in self.bands:
            lambda_band = self.wavelengths[band]

            #tau_band = self.tau_driver * (lambda_band / lambda_ref) ** 0.17
            sigma_band = self.sigma_driver * (lambda_band / lambda_ref) ** (-0.479)

            #taus[band] = tau_band
            sigmas[band] = sigma_band

        return sigmas #, taus

    def addGNoise(self, y, sigma_phot: float = 0.01, rng = None):
        """
        Add Gaussian observational noise to a clean light curve.
        Noise is drawn from N(0, sigma_phot^2), where sigma_phot
        is the expected LSST photometric error 0.01 mag = 10 mmag
        (obtained from: https://rubinobservatory.org/for-scientists/rubin-101/key-numbers). 
        """
        rng = rng if rng is not None else self.rng
        
        y = np,asarray(y, dtype = float)

        # also possible to give an array of sigmas instead of the LSST photometric error

        sigma = np.full(y.shape, sigma_phot, dtype = float)
        
        noise = rng.normal(loc = 0.0, scale = sigma, size = y.shape)
        y_noisy = y + noise

        return y_noisy, sigma
        
        
    def makeDriver(self,
                   t,  # so it can simulate any given time stamps
                   lc_seed: int | None = None,
                   ):
        """
        Primitive that creates one realization of the kernel at requested times. It is also
        the only caller where the kernel is used to it can be changed later (e.g., to CARMA(2,1))

        Also, the error here is neglible (~1e-7 mag) so it v
        """
        t = np.asarray(t, dtype=float)
            
        DRW_kernel = DRW_term(np.log(self.sigma_driver), np.log(self.tau_driver))

        # we do not store the EzTao error because we add it later AddNoise()
        t_out, y_out, _ = gpSimByTime(carmaTerm = DRW_kernel,
                                             SNR = self.snr_driver,
                                             t = self.t,
                                             log_flux = self.log_flux,
                                             lc_seed = lc_seed if lc_seed is not None else self.seed)

        # ravel(): nLC=1 returns 1-D, but nLC>1 would return (nLC, N)
        return np.asarray(t_out, dtype=float).ravel(), np.asarray(y_out, dtype=float).ravel()

    def makeSparse(self, n_obs: int = 130, gap_frac: float = 0.35, n_gaps: int = 3, rng = None):
        """
        Generates irregular, gappy observing cadence, only TIMES, no fluxes, The driver is
        simulated afterwards, directly at the union of the irregular timestamps and the dense
        truth, so that no interpolation is ever required.
        Noise is not applied here.

        n_obs    : attempted epochs BEFORE removing the gaps, so the final observed times will
                    be leq, real num of obs is approx n_obs * (1 - gap_frac)
        gap_frac : fraction of the baseline actually removed. Exact by construction:
                   windows are laid out disjointly and wholly inside [0, t_max], so
                   neither overlap nor edge-truncation can leak away removed time.
        n_gaps   : number of removed windows. Widths are random but sum to
                   gap_frac * t_max, so individual gaps span a range of sizes.

        Returns (t_obs, windows) with windows of shape (n_gaps, 2). The windows are
        kept only because they are RNG-drawn and cannot be recovered later; the gap
        analysis itself uses gapWidths(t_obs, ...), not these.
        """
        
        rng = rng if rng is not None else self.rng

        if not 0.0 <= gap_frac < 1.0:
            raise ValueError("gap_frac must be in [0, 1)")

        t_obs = np.sort(rng.uniform(0.0, self.t_max, size=n_obs))

        if gap_frac == 0.0 or n_gaps < 1:
            return t_obs

        # split the removed time G at random among n_gaps windows using np.dirichlet
        G = gap_frac * self.t_max
        
        widths = rng.dirichlet(np.ones(n_gaps)) * G
        
        # reserve a margin at each end before splitting the surviving time: a window
        # landing at the very edge has no epochs on one side, so it truncates the
        # baseline instead of making a fillable gap (the bridge needs TWO endpoints).
        # 3x the mean spacing -> ~0.4% edge cases, and it auto-scales with n_obs.
        margin = 3.0 * self.t_max / n_obs
        free = (self.t_max - G) - 2.0 * margin
        if free <= 0:
            raise ValueError("gap_frac too large for this n_obs: no room for edge margins")

        # and the surviving time among the n_gaps + 1 observable segments
        segments = rng.dirichlet(np.ones(n_gaps + 1)) * free
        segments[0] += margin
        segments[-1] += margin     # sum still equals t_max - G, so gap_frac stays exact

        # lay out alternately: seg, gap, seg, gap, ..., seg
        lo = np.cumsum(segments[:-1]) + np.concatenate([[0.0], np.cumsum(widths[:-1])])
        hi = lo + widths
        windows = np.column_stack([lo, hi])

        keep = ~((t_obs[:, None] >= lo) & (t_obs[:, None] <= hi)).any(axis=1)

            
        return t_obs[keep], windows 

        
    @staticmethod
    def gapWidths(t_obs, t_eval):
        """
        For each evaluation time, the width of the observational gap containing it.
        NaN where t_eval lies outside the observed span (extrapolation, not gap
        filling -- report those separately or drop them).

        Bin the error by gapWidths(...) / tau, not by the designed windows: this is
        the gap the model actually faces, and it also catches holes that arose by
        chance from sparse sampling rather than by design.
        """
        t_obs, t_eval = np.asarray(t_obs), np.asarray(t_eval)
        i = np.searchsorted(t_obs, t_eval)
        inside = (i > 0) & (i < t_obs.size)
        w = np.full(t_eval.shape, np.nan)
        w[inside] = t_obs[i[inside]] - t_obs[i[inside] - 1]
        return w

    def makeSingleBand(self, lc_seed: int | None = None,
                       n_obs: int = 130, gap_frac: float = 0.35, n_gaps: int = 3,
                       sparse: bool = True, add_noise: bool = True,
                       sigma_phot: float = 0.01):
        """
        One observable single-band light curve:
            cadence -> union grid -> ONE simulation -> slice -> noise.
        The whole curve is reproducible from lc_seed alone,
        so makeSingleBand(lc_seed=i) for i in range(N) builds a dataset.
        """
        seed = lc_seed if lc_seed is not None else self.seed
        #rng = np.random.default_rng(seed)
        rng = np.random.default_rng([seed, 1])   # independent of the __init__ stream

        # 1. cadence, so the driver can be simulated AT the observation times too
        if sparse:
            t_obs, windows = self.makeSparse(n_obs, gap_frac, n_gaps, rng=rng)
        else:
            t_obs, windows = self.t_true.copy(), np.empty((0, 2))

        # 2. ONE simulation on the union so truth and observations are the SAME path
        t_all = np.union1d(self.t_true, t_obs)
        t_all, y_all = self.makeDriver(t_all, lc_seed=seed)

        # 3. slice both out of that single realisation, no interpolation anywhere
        y_true = y_all[np.searchsorted(t_all, self.t_true)]
        y_obs_clean = y_all[np.searchsorted(t_all, t_obs)]

        # 4. noise and only on the observations
        if add_noise:
            y_obs, yerr = self.addGNoise(y_obs_clean, sigma_phot=sigma_phot, rng=rng)
        else:
            y_obs, yerr = y_obs_clean, np.full_like(y_obs_clean, sigma_phot)

        return {
            "t_true": self.t_true.astype(np.float32),   # dense truth: EVAL ONLY
            "y_true": y_true.astype(np.float32),
            "t_obs": t_obs.astype(np.float32),          # what the model sees
            "y_obs": y_obs.astype(np.float32),
            "yerr": yerr.astype(np.float32),
            "gap_windows": windows.astype(np.float32),  # RNG-drawn -> must be stored
            "tau": float(self.tau_driver),
            "sigma": float(self.sigma_driver),
            "band": self.driver_band,
            "seed": int(seed),
        }  
            

    def plotLC(lc, ax = None, recon = None, show_truth = True, magnitudes = True):
    """
    One light curve, Kozlowski-style.

    lc    : dict from makeSingleBand()
    recon : None, or a dict describing a reconstruction --
              {"t": (M,), "samples": (n, M), "label": str}      Monte Carlo (NSF)
              {"t": (M,), "med":, "lo":, "hi":, "label": str}   precomputed (GP)
    """
        plt.rcParams.update(AA_STYLE)
        if ax is None:
            _, ax = plt.subplots(figsize=(9, 3.4))

        if show_truth:
            ax.plot(lc["t_true"], lc["y_true"], color="0.75", lw=0.9, zorder=1,
                    label="latent driver (truth)")

        if recon is not None:
            t = np.asarray(recon["t"])
            if "samples" in recon:
                lo, med, hi = np.percentile(recon["samples"], [16, 50, 84], axis=0)
            else:
                lo, med, hi = recon["lo"], recon["med"], recon["hi"]
            lab = recon.get("label", "reconstruction")
            ax.fill_between(t, lo, hi, color="tab:blue", alpha=0.22, lw=0, zorder=2,
                            label=rf"{lab} $\pm1\sigma$")
            ax.plot(t, med, color="tab:red", lw=1.0, zorder=3, label=f"{lab} median")

        ax.errorbar(lc["t_obs"], lc["y_obs"], yerr=lc["yerr"], fmt="o", ms=2.2,
                    lw=0.7, color="k", capsize=0, zorder=4, label="observed")

        ax.set_xlabel("Time (day)")
        ax.set_ylabel("Magnitude" if magnitudes else "Flux")
        ax.set_xlim(lc["t_true"][0], lc["t_true"][-1])
        if magnitudes:
            ax.invert_yaxis()          # brighter is up
        ax.legend(ncol=4, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, 1.16))
        return ax

    def makeDataset(n_curves: int = 500, val_frac: float = 0.2, seed0: int = 0,
                gen_kwargs: dict | None = None, lc_kwargs: dict | None = None):
    """
    Stage 1 dataset: N independent single-band curves, split train/val.

    One generator per curve, so that tau_driver=None / sigma_driver=None in
    gen_kwargs draw a fresh value per curve from the Yu et al. (2025) LogUniform ranges.
    Every curve is re-creatable in isolation from seed0 + i.
    """
    gen_kwargs = {"bands": "g", "driver_band": "g"} | (gen_kwargs or {})
    lc_kwargs = lc_kwargs or {}

    curves = []
    for i in range(n_curves):
        gen = MultibandLCGenerator(lc_seed=seed0 + i, **gen_kwargs)
        curves.append(gen.makeSingleBand(lc_seed=seed0 + i, **lc_kwargs))

    n_train = int((1.0 - val_frac) * n_curves)
    
    return curves[:n_train], curves[n_train:]


##    def makeMultiband(self):
##        if len(self.bands) < 2:
##            raise ValueError("Bands should be at least 2")
##        
##        taus, sigmas = self.computeDRWparams()
##
##        lightcurves = {}
##
##        for i, band in enumerate(self.bands):
##

        
