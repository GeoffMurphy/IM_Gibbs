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

Foregrounds are marginalised over rather than projected out, so the signal loss
that a PCA clean incurs at low k does not have to be corrected for after the
fact.

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

FastBox supplies the simulated H I cubes and the PCA filter. Upstream
[philbull/FastBox](https://github.com/philbull/FastBox) only supports **cubic**
grids — `nsamp` must be an `int` — so it fails on this grid with

```
TypeError: 'tuple' object cannot be interpreted as an integer
```

Install the fork that accepts a 3-tuple:

```bash
pip install "git+https://github.com/GeoffMurphy/FastBox.git"
```

The patch touches `box.py` (the bulk of it: `nsamp` as a tuple, `boxfactor`,
`kmax`, `set_fft_sample_spacing`, `realise_velocity`, `freq_array`,
`pixel_array`), `foregrounds.py` (`C_ell` normalisation and cube allocation),
`beams.py` (beam array allocation) and `halos.py` (voxel volume and catalogue
scaling). Cubic behaviour is unchanged: `box.N` is still a plain `int` on a
cubic grid and only becomes a tuple when the grid is not.

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
  data.py           locating the input cubes
notebooks/
  1_generate_signal_cube.ipynb     simulated H I cube + S starting point
  2_gibbs_sampling.ipynb           the sampler, diagnostics, transfer function
  3_pca_transfer_function.ipynb    PCA clean benchmark
scripts/
  run_gibbs.py      the sampling loop without the plots
  submit_gibbs.sh   SLURM wrapper
tests/              regression tests on the geometry and the binning
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
