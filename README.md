# IM_Gibbs

Bayesian component separation and power spectrum estimation for 21cm intensity
mapping data cubes.

Gibbs sampling with Gaussian Constrained Realisations, jointly inferring the
21cm signal `s`, the foreground amplitudes `f`, the signal covariance `S` (the
power spectrum), and the foreground covariance `F`. Each iteration draws

```
(s, f) | S, F, d     — one preconditioned LGMRES solve of A x = b
S      | s           — inverse-gamma, per radial k-bin
F      | f           — inverse-Wishart
```

Optionally a fourth component `g`, an instrumental systematic on a fixed
low-rank basis with a fixed prior, sampled in the same solve:
`(s, f, g) | S, F, d`. See [Systematics](#systematics).

Foregrounds are marginalised over rather than projected out. Measured against a
PCA clean on the same cube, k-bins, estimator and truth, the sampler retains
substantially more signal at intermediate k — 0.86 against 0.46 at
k = 0.37 Mpc⁻¹, 0.99 against 0.71 at k = 0.87 — so that loss does not have to
be corrected for after the fact. **At the lowest bin the two are equivalent**
(0.13 against 0.16): both lose ~85%, and marginalising does not rescue it. See
`docs/STATUS.md`.

This version runs on the **real MeerKLASS L2021 footprint** — a non-cubic
(70, 45, 250) grid with strongly anisotropic voxels (~8.6 x 8.6 x 1.0 Mpc) and
an irregular survey mask. Earlier work used a cubic 72³ slice.

> **Looking for the paper?** The example notebook and cubic-grid sampler behind
> Murphy et al. ([arXiv:2604.26890](https://arxiv.org/abs/2604.26890)) are
> tagged [`v1-paper`](../../tree/v1-paper). This branch has moved on
> considerably.

---

## Install

```bash
git clone https://github.com/GeoffMurphy/IM_Gibbs.git
cd IM_Gibbs
pip install -e .
```

### Installing FastBox

FastBox supplies the simulated H I cubes and the PCA filter.

```bash
pip install "git+https://github.com/GeoffMurphy/FastBox.git"
```

That is a fork of [philbull/FastBox](https://github.com/philbull/FastBox),
currently identical to upstream `main` (`883677f`). The fork exists as a pin,
not a patch: the simulated P(k) is the truth curve the sampler output is
compared against, so an upstream change should be something you adopt
deliberately rather than pick up on the next `pip install`. Sync it when you
want to.

Upstream added non-cubic grid support in 2026 — `nsamp` takes an `int` or a
3-tuple — along with `indexing='ij'` fixes in `filters.py` and a `meerklass.py`
module. Verified against this repository: the generation pipeline runs
end-to-end and `HITracer.signal_amplitude()` reproduces the recorded
`Tb_mK = 0.15048796349807` bit for bit.

FastBox is only needed to *generate* cubes and to run the PCA benchmark. The
sampler itself, and `imgibbs.grid`, work without it.

---

## Data

The **simulated and derived products ship in `data/`** and are enough to
inspect the priors and reproduce the k-binning:

| File | What it is |
|---|---|
| `Fastbox_cube_cropped.npy` | Simulated H I signal cube on this grid, in K |
| `S_starting_point_cropped.npy` | Diagonal signal covariance, 787,500 entries, K² |
| `S_starting_point_cropped_meta.json` | Geometry, k-bins and units of that run |
| `Fastbox_Pk_cropped.npy` | Directly measured P(k) — the truth curve |
| `Fastbox_PkSample_cropped.npy` | The inverse-gamma draw actually used for `S` |
| `Fastbox_kbins_cropped.npy` | Bin centres `sig_k` |
| `mock_ref_injection.npy` | Fixed injected mock, shared by both transfer-function arms |
| `L2021_cropped_cube_meta.json` | Provenance of the observational crop |

The **MeerKLASS L2021 observational cubes are not redistributed here.** They are
collaboration data. To run the notebooks on the real data, put
`L2021_polished_cube.npy` in `data/`, or point `IMGIBBS_DATA` at wherever you
keep it:

```bash
export IMGIBBS_DATA=/path/to/cubes
```

It is built from `Nscan961_Tsky_cube_p0.3d_sigma4.0_iter2.fits` (MeerKLASS
2021, 0.3 deg pixels), keeping FITS channels 550–1050 — a (133, 73, 500) cube
in K. `imgibbs.load_l2021_cube()` raises a message saying exactly this if it
cannot find it.

---

## Quickstart

```bash
# 1. Build the simulated cube and the signal-covariance starting point
jupyter lab notebooks/1_generate_signal_cube.ipynb

# 2. Sample — in the notebook, or as a batch job
jupyter lab notebooks/2_gibbs_sampling.ipynb
python scripts/run_gibbs.py 6 --n-samples 500 --seed 42
sbatch scripts/submit_gibbs.sh 6 --n-samples 500

# 3. The PCA benchmark the sampler is compared against
jupyter lab notebooks/3_pca_transfer_function.ipynb
```

`run_gibbs.py` checks the grid it is running on against the metadata of the `S`
it loaded, and refuses to start if they disagree:

```
S does not match this grid:
  shape: S built on (70, 45, 250), this run is (70, 45, 200)
  box_dims: S built on [601.1, 386.4, 254.4], this run is [610.1, 392.2, 204.5]
```

One sample takes ~4 s on this grid.

---

## Systematics

Three are implemented: **ground spill**, **1/f noise** and **polarisation
leakage**. All three reuse the same block, so adding another is a matter of
supplying templates rather than touching the sampler.

```bash
python scripts/systematics_injection.py --systematic onef --arm on
sbatch scripts/submit_systematics.sh leakage on --rm 1000     # on a cluster
```

A single lesson runs through all three: **whatever is smooth in frequency is
already inside the foreground block's span and cannot be separated from it —
and does not need to be.** The smooth part of ground spill, the 1/f common
mode, and polarisation leakage at ordinary Galactic Faraday depths are all
absorbed to machine precision. What is left is the identifiable part, and it
is usually a smaller and more structured thing than the systematic as a whole.

An instrumental systematic can be sampled as a fourth block,
`d = w * (Us s + Uf f + Ug g) + n`, where `Ug` is a fixed low-rank basis and
`g` a short vector of amplitudes. Ground spill is implemented:

```bash
python scripts/run_gibbs.py 6 --groundspill --gs-period 17.5
```

Two measured facts shape the whole thing, and both are worth knowing before
using it.

**The smooth part of ground spill is invisible, and harmlessly so.** The
foreground block has per-pixel free amplitudes on `n_modes` smooth frequency
modes, so a smooth spillover envelope lies inside its span to machine precision
at any `n_modes`. Inject 0.5 K of it — 600x the H I rms — and nothing moves.

**The standing-wave ripple is the part that matters.** It is a single
`k_parallel` mode at `k = 2*pi*B/(Lz*P)`, so all of its power lands in one
k-bin rather than spreading. On this grid a 10–20 MHz period puts it at
k = 0.064–0.129 Mpc⁻¹, against bin centres `[0.068, 0.160, ...]` — the lowest
signal bins. A 6-mode foreground clean absorbs only 18% of a 17.5 MHz ripple;
20 modes absorbs it all, but `docs/STATUS.md` records that 20 modes removes the
21cm signal too, so that is not a fix.

The model is deliberately tight — four parameters by default (constant +
scan-direction gradient, times cos + sin at one period) with a fixed prior `G`.
Per-pixel amplitudes would just be more foreground modes. `G` is a prior rather
than a sampled covariance because `g` is a single short vector, unlike `F`,
which has `Npix` amplitude vectors behind it.

On real data you cannot distinguish a systematic that was removed from one that
was never there, so the thing to run first is the injection test:

```bash
python scripts/systematics_injection.py --arm off --n-samples 120
python scripts/systematics_injection.py --arm on  --n-samples 120
python scripts/systematics_injection.py --summarise
```

It builds a synthetic cube — simulated H I, the real cube's Legendre
foreground, ground spill, noise — on the live grid and samples it twice with
identical seeds, differing only in whether the block is on. The block can only
recover the part of the ripple the foreground does not already take; both
scripts print that ceiling before they start.

`notebooks/4_systematics.ipynb` shows the structure of the block — the
templates, the separability, and what a 6-mode clean leaves behind — without
running the sampler.

`sys_basis=None` is the default throughout, so the three-block sampler is
unchanged.

### Disk

`x_sample` traces are ~13 MB each, so a 500-sample chain is ~6.5 GB. `S` is
piecewise-constant over the k-bins, so `Pk_trace` plus the bin metadata
reconstructs it exactly — writing the full cube each iteration costs another
~6 MB per sample for no extra information, and is now opt-in behind `--save-S`.
`outputs/` is gitignored.

---

## Layout

```
imgibbs/
  grid.py           CROP -> shape, frequencies, redshifts, box_dims
  kbins.py          radial k-binning and the P(k) estimator
  linear_system.py  A, b, and the block-diagonal preconditioner
  covariance.py     the inverse-gamma and inverse-Wishart draws
  systematics.py    instrumental systematics: the ground-spill block
  data.py           locating the input cubes
notebooks/
  1_generate_signal_cube.ipynb     simulated H I cube + S starting point
  2_gibbs_sampling.ipynb           the sampler, diagnostics, transfer function
  3_pca_transfer_function.ipynb    PCA clean benchmark
  4_systematics.ipynb              structure of the three systematics
scripts/
  run_gibbs.py              the sampling loop without the plots
  systematics_injection.py  systematics injection test, known answer
  systematics_report.py     tables and figures from a finished injection run
  pca_benchmark.py          PCA + transfer function, on the sampler's k-bins
  submit_gibbs.sh           SLURM wrapper
  submit_systematics.sh     SLURM wrapper, one job per arm
tests/              regression tests on the geometry, binning and systematics
docs/STATUS.md      what is settled, what is open, what is known to be wrong
```

### Why the geometry lives in one module

`S` is a per-voxel array indexed by k-bins derived from `box_dims`. If the
notebook that *builds* `S` and the notebook that *consumes* it disagree about
the grid, the signal prior is silently attached to the wrong wavenumbers and
the sampler produces plausible-looking nonsense with no error anywhere.

All three notebooks and the batch script therefore call the same
`survey_grid(CROP, shape)`, and everything follows `CROP` — change the crop and
the frequencies, redshifts and box dimensions move with it. `tests/` pins the
result to the values the shipped `S` was generated with.

---

## Status

`docs/STATUS.md` is the working record: the crop and its justification, the
k-binning rationale, and the open issues — the inpainting operating ~200x
outside its design case, the suspected `S` feedback loop, the missing primary
beam in the simulation, and the light-cone approximation in `box_dims`.

Read it before trusting a number out of this repository.

---

## Citation

If you use this code, please cite Murphy et al.,
[arXiv:2604.26890](https://arxiv.org/abs/2604.26890).

## Licence

MIT — see [LICENSE](LICENSE). FastBox is a separate project with its own terms.
