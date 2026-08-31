# MASS-Thesis

Reconstructing photometric light curves of active galactic nuclei with neural
stochastic flows. Master's thesis, University of Belgrade, Faculty of Mathematics.

This repository contains the code, data, and LaTeX sources for the thesis. The
method reconstructs gaps in AGN light curves, modelled as a damped random walk,
using neural stochastic flows, and is validated against the analytic transition
probability and a Gaussian process baseline on simulated and real data from the
Zwicky Transient Facility.

## Repository structure

- `framework/` — code and notebooks for the low-noise simulation experiments
- `experiments/` — code and notebooks for the ZTF-matched and real-data experiments
- `data/` — simulated training and validation sets, and the digitized quasar mass
  distribution used to assign damping timescales
- `chapter/`, `assets/`, `main.tex`, `header.tex`, `references.bib` — LaTeX sources
  for the thesis document
- `build/` — compiled output

## Code

Both `framework/` and `experiments/` contain the same core modules:

- `LightCurves.py` — simulation of damped random walk light curves (uses EzTao)
- `NSF.py` — the neural stochastic flow and bridge, training loop
- `GapFill.py` — reconstruction with the trained bridge, and the DRW Gaussian
  process baseline
- `Tensorize.py` — padding, triplet sampling, dataset loading and normalization

## Dependencies

The reconstruction method builds on the `jax_nsf` codebase of Kiyohara et al. (2025).
Simulation uses EzTao; training uses JAX, Equinox, and Optax. Because EzTao's
dependencies conflict with those of `jax_nsf`, simulation and training are run in
separate environments, with datasets written to `data/` as the interface.

## Running the experiments

The notebooks are run in order. Simulation and sample assembly (Stage 0) write
their outputs to the repository, which the training and evaluation notebooks
(Stage 1) then read. These outputs — the simulated datasets in `data/` and the
fetched ZTF sample in `experiments/lightcurves/` — are committed to the
repository, so the Stage 1 notebooks can be run directly, without regenerating the
simulations or re-querying the ZTF archive.

**Low-noise experiments** (`framework/`):

1. `Stage0.ipynb` — simulates the damped random walk light curves at fixed and
   varying damping timescale and writes `train.npz` and `val.npz` to `data/`.
   Produces the example light curve and the sampling-regime figures.
2. `Stage1.ipynb` — loads those datasets, trains the neural stochastic flow and
   bridge, and reconstructs the validation curves. Produces the learned-transition
   comparison, the reconstruction figures, and the coverage, RMSE, and predictive
   log-likelihood against the Gaussian process baseline.
3. `Stage1Tests.ipynb` — repeats the training over a range of the loss weighting
   and reports the coverage and RMSE at each value.

**ZTF experiments** (`experiments/`):

4. `ZTF_Stage0.ipynb` — assembles the ZTF AGN sample, fetching the light curves
   from the ZTF archive or resuming from the cached `lightcurves/ztf_sample.pkl` if
   it is present, measures the sampling regime per band, and constructs the
   ZTF-matched simulations. Produces the survey-statistics and per-band regime
   figures.
5. `ZTF_Stage1.ipynb` — trains the model on the ZTF-matched simulations and
   evaluates its calibration, then applies the trained model to the real ZTF light
   curves through the masking experiment. Produces the multi-band coverage tables,
   the learned-transition figures, and the real-data reconstruction figures.

## Thesis

The compiled thesis is built from the LaTeX sources with `make` (see `Makefile`).
