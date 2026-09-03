# Status

The working record for this repository: what the current configuration is, why
it is that way, and what is still wrong with it. Read this before trusting a
number out of here.

Last substantive update: 2026-09-03.

---

## Current configuration

```python
CROP     = (slice(33, 103), slice(14, 59), slice(0, 250))   # -> (70, 45, 250)
box_dims = (601.10, 386.42, 254.39)                          # Mpc
n_k_bins = 5           # a CEILING; kbins_from_crop chose 5
n_modes  = 6           # foreground modes (Legendre)
T_sys    = 16 K,  del_nu = 0.208984 MHz,  del_t = 1000 s
```

- **Redshift** 0.3885 – 0.4629, midpoint 0.4257.
- **Voxels** 8.59 x 8.59 x 1.02 Mpc — strongly anisotropic, which is exactly
  why a cubic `box_dims` (the old `(232, 232, 232)`) was wrong: it mismaps every
  mode to the wrong |k| bin.
- **Modes per bin** `[898, 12474, 106814, 209276, 454834]`, `k` from 0.0447 to
  3.129 Mpc⁻¹.
- **Units** everything in K, so `S` is in K². `HITracer.signal_amplitude()`
  returns mK — the generation notebook divides by 1000. Getting this wrong
  misscales `S` by 10⁶.
- **`box_dims` describes the FFT grid** (21.0 x 13.5 deg), *not* the ~200 deg²
  scanned footprint. The footprint mask is handled by the `w` weights.

`tests/test_imgibbs.py` pins the geometry and the binning to the values the
shipped `S_starting_point_cropped.npy` was generated with, and
`scripts/run_gibbs.py` refuses to start if the two disagree.

> **Note on an earlier draft of this document.** It quoted the grid as
> (70, 45, 500) with `box_dims = (558.4, 359.0, 494.7)`. That was the
> 500-channel configuration; the working grid has been the 250-channel cut
> since 2026-08-24. The numbers above are read back from
> `data/S_starting_point_cropped_meta.json`.

---

## The crop

`L2021_polished_cube.npy[33:103, 14:59, :]` is the tight bounding box of the
drift-scan footprint. It loses **zero** valid voxels (930,938 either way) while
fill goes 19.2% → 59.1%, at 3.08x fewer voxels.

Source: `Nscan961_Tsky_cube_p0.3d_sigma4.0_iter2.fits`, channels 550–1050 (the
low band of Wang et al. 2021), 0.3 deg pixels.

The crop is a **fill-fraction decision, not a fastbox one**. Both the full cube
and the crop are non-cubic. Un-cropping recovers *no* additional sky — both
contain the same 1865 pixels with data — while tripling the voxel count and
dropping fill back to 19.2%.

---

## Why the k-binning changed

Chasing dips in the recovered P(k) on the 250-channel cut. Three separate
problems wearing one coat.

**The lowest k-bins contained no information.** `make_kbins` took `k_min` from
the *box*, but the cube is a bounding box round a diagonal band filling 59% of
it. Modes longer than the band are set by the zero padding, not by data. On the
250-channel cut this was stark: **bin 1 was 100% kz = 0** — modes exactly
constant in frequency, the smoothest possible spectrum and therefore degenerate
with the foreground. The sampler split that power by prior and the chain
wandered (tau_int 47 in bin 3, ESS 16).

**Why 500 channels hid it.** At 500 channels Lz = 495 Mpc so kz_min = 0.0127,
almost equal to kx_min = 0.0113 — a near-isotropic box, so every spherical
shell mixes radial and transverse modes. At 250, kz_min = 0.0247 = 2.4x kx_min,
and the lowest shells can only be reached transversally. Halving the band did
not create the problem, it *segregated* the kz = 0 modes into their own bins.

`kbins_from_crop` (in `imgibbs/kbins.py`) fixes four things:

1. `k_min` from the survey footprint, measured from the non-zero mask, so it
   follows the crop the way `box_dims` does.
2. Drops the kz = 0 plane — 0.4% of modes, and they are the
   foreground-degenerate ones.
3. Picks the bin count from occupancy rather than by hand.
4. Pins the lowest bin edge: `logspace(log10(x), ...)` does not round-trip, and
   the smallest modes were being silently binned with DC.

(1) and (2) did the work. (3) never bound — with (1) and (2) in place, 12 bins
is fine on every cut tried (250, 500, 125 channels). The current ceiling of 5
is a deliberately coarse choice for chain stability, not a constraint.

Both notebooks were verified to produce **identical** `sig_k` and `idxs`; that
is now enforced by construction, since they call the same function on the same
`survey_grid` output.

### One change here is a modelling decision, not a bug fix

`S[idxs == 0] = 1e30` became

```python
S[idxs == 0] = 1e-12 * np.median(PkSample)   # suppress the degenerate plane
S[kbin_meta['dc_index']] = 1e30              # keep the overall mean free
```

Bin 0 went from 1 voxel to ~3200, and a flat 1e30 prior across that plane
invites the signal to absorb foreground power in exactly the degenerate modes.

**Consequence:** the foreground model must now account for all
frequency-constant structure. Correct if `Uf` has the freedom; if not, it lands
in the noise term instead. Check the foreground residual after any change here.

**Never comment these lines out.** `1/S` appears in `construct_A`, `1/S` and
`1/np.sqrt(S)` in `construct_b`, and `1.0/S_rfft` in the preconditioner. `S`
starts as `np.zeros(...)`, so removing the assignment gives `1/0` and NaNs the
whole solve. If reverting to `make_kbins`, restore the original
`S[idxs == 0] = 1e30` — there bin 0 is only the DC mode and 1e30 is correct.

---

## `n_modes` is a trade with no unbiased setting

Measured on the 250-channel cut (foreground residual relative to the H I, and
the PCA transfer function T):

| n_modes | FG residual / HI, low k | T(k), low k | low-k tau_int |
|---|---|---|---|
| 6  | 1.3 – 2.5 (above the signal) | 0.10 – 0.26 | 1.2 – 1.7 |
| 10 | 0.24 – 0.86 | 0.056 – 0.17 | 1.8 – 17 |
| 20 | 0.06 – 0.13 | **0.006 – 0.06** | **54 – 105** |

Too few leaves foreground above the signal; too many removes the signal *and*
destroys convergence, because there is then nothing left for the sampler to
constrain. 10 was the working compromise; 6 converges best but is biased high.

---

## The comparison cannot work without injection

The sampler only ever sees `data_cube` — `construct_b(..., data_cube, ...)`.
The `inj` in the transfer-function cell is used *only* inside the PCA loop.

`true_pk` is the mean of 100 simulated mocks: an expectation of what H I would
look like at this redshift, not anything present in the L2021 map. So
`Gibbs / True` and `Corrected / True` are both "leftover divided by an
expectation", not recovery fractions. **Without injection you cannot
distinguish "the method destroyed the signal" from "there was no signal"** —
and for L-band single-dish auto-power the latter is the expected answer.

Note also that `corrected_PS = pca_pk / T_m` divides an *auto* power by T and is
inflated at both ends for different reasons: ~10x amplification of residual
foreground at low k, while at high k `pca_pk` is already 18x the mock power
because the data is noise-dominated (T ~ 0.95 there, so no amplification at
all). Do not treat that curve as a target to match.

The fix is `T_gibbs` vs `T_pca` on a shared injected mock, using
`data/mock_ref_injection.npy` so both arms and any later session use an
identical injected signal.

---

## Open issues

### The inpainting is ~200x outside the job it was designed for

It was meant for a handful of pixels lost to RFI flagging, but on this grid
**1285 of 3150 pixels (40.8%) have no valid data at any frequency** — they are
filled entirely from the prior. The other 1865 pixels have a median **499 of
500** good channels, i.e. ~0.2% flagged. Note 1865 × 0.998 / 3150 = 0.591, the
fill fraction to three decimals: essentially *all* the missing volume is empty
sky, and genuine per-channel flagging is a rounding error on top.

This matters because `Pk_trace` measures the power of the **whole** cube, so it
is a box average blending observed sky with prior draws. Restricting P(k) to the
footprint (mask, then normalise by `<w²>`) raises mid-k power by up to **15x** at
k ≈ 0.09, converging back to ~1 by k ≳ 0.6. The footprint cells in
`2_gibbs_sampling.ipynb` do this, with a mask-response control built in (mocks
with fluctuations switched off outside the footprint come back at 0.90–1.11, so
the effect is not mask leakage).

Cross-check on the 500-sample chain: over 0.02 < k < 0.2 the footprint estimate
agrees with the PCA + injection transfer function (Cunnington et al. 2023) to a
**median ratio of 1.03**, against **0.16** for the box average. Above k ≈ 0.24
the two legitimately diverge — PCA keeps the full thermal noise while the Gibbs
`s` field is Wiener-suppressed — so that is an estimator difference, not a
disagreement about the H I.

### Suspected `S` feedback loop (hypothesis, not yet tested)

`S` is estimated from the box-average power each iteration, the flagged voxels
are then filled at that `S`, and those fills are counted in the next box
average — so the inpainted volume pulls `S` down and the lowered `S` makes the
next fill smoother. This would explain why the gap peaks at k ≈ 0.09 rather
than sitting flat at the fill fraction.

The fix is to estimate `S` from the footprint-restricted power inside the
`S_samp` block instead of the box power. **Not a drop-in:** masking correlates
Fourier modes, so both the inverse-gamma draw's per-mode independence and its
`N_k/2 - 1` degrees of freedom need revisiting first.

Cropping cannot avoid this — the bounding box already loses zero valid voxels
at 59.1% fill, so the footprint is genuinely irregular.

### Smaller open items

- **The sampler hard-codes `data_cube`.** The `T_gibbs` comparison needs runs
  on `inj_ref` too; parameterising this (e.g. `sampler_input = data_cube` near
  the top) is the last piece before that comparison works.
- **`mask_flagged` must match across both arms** — it moves T's high-k plateau
  between ~0.55 and ~0.95.
- **High-k bins sit at tau_int ~ 100, ESS 7–13.** ~10x more samples would be
  needed, but the data is noise-dominated there anyway, so it buys little.
- **No beam in the simulation.** The MeerKAT primary beam is ~1 deg FWHM
  ≈ 26.6 Mpc at this redshift, about 3.3 voxels. The real data is smoothed on
  scales where the simulated cube still has full power, so the two P(k) curves
  will diverge at high k⊥ for reasons unrelated to the sampler. Decide whether
  to convolve before using `Fastbox_Pk_cropped.npy` as a truth curve.
- **`T_sys` is now 16 K**, from Wang et al. (2021) Table 1. Earlier runs used
  30 K; the switch lowered the noise by (30/16)² ≈ 3.5x, which materially
  changes the weighting between data and prior. Any comparison against a
  pre-2026-08 run is not like for like.
- **Light-cone approximation.** `Lx`/`Ly` use a single `D_M` at the midpoint
  redshift, but 21 deg subtends 465 Mpc at z = 0.32 and 647 Mpc at z = 0.46.
  Fine for a prior; revisit for precision P(k).
- **`3_pca_transfer_function.ipynb` uses a different grid** — 72 channels and
  the box-based `make_kbins` — from the other two notebooks. That is deliberate
  (the published PCA transfer function is calibrated against those bins) but it
  does mean its k axis is not identical to the sampler's.

### Known landmine

**`Us` breaks on an odd number of frequency channels.** `fft.irfftn` in
`imgibbs/linear_system.py` has no `s=` argument, so it infers the last-axis
length and always returns an even count. Dormant at 250 and 500 channels;
silently drops a channel if you re-channelise to an odd number. Pinned by
`test_Us_loses_a_channel_on_an_odd_count` so the behaviour cannot change
unnoticed.

---

## Future grid options (measured 2026-08-13, not acted on)

Cropping *tighter*, accepting a little loss, is the direction that helps.
Brute force over all 18,331,840 sub-rectangles of the (133, 73) pixel grid:

| keep ≥ | best fill | retained | crop | shape |
|---|---|---|---|---|
| 100% | 59.11% | 100.00% | `[33:103, 14:59, :]` | (70, 45, 500) ← current |
| 99% | 65.18% | 99.04% | `[33:102, 17:58, :]` | (69, 41, 500) |
| 95% | 73.61% | 95.08% | `[35:100, 20:57, :]` | (65, 37, 500) |
| 90% | 79.52% | 90.21% | `[35:99, 22:55, :]` | (64, 33, 500) |

**But the footprint is a diagonal band** (constant-elevation drift scan, so it
is tilted in RA/Dec by a fixed angle), which is why a rectangle fits it so
poorly. Shearing the grid — shifting y by one pixel every N rows, **pure integer
re-indexing, no interpolation** — fits it far better:

| shear dy/dx | fill @ 99% kept | fill @ 95% kept | shape @ 99% |
|---|---|---|---|
| 0.00 | 65.18% | 73.61% | (69, 41, 500) |
| 0.20 | 84.74% | 92.46% | (68, 32, 500) |
| **0.25** | **88.78%** | **95.43%** | **(67, 31, 500)** |
| 0.30 | 87.61% | 95.22% | (68, 31, 500) |
| 0.40 | 77.64% | 86.58% | (68, 35, 500) |

At dy/dx = 0.25 (one pixel every four rows) fill goes **59% → 89%** keeping 99%
of the data, and the grid shrinks to 1,038,500 voxels — **34% fewer than now**.
Inpainted volume would drop from 40.8% to ~11%, roughly the regime the
inpainting was actually designed for. It improves the inpainting problem and the
runtime at once, and is a much cheaper fix than reworking `S_samp`.

**Two things to get right before using it.**

1. **The k-binning must use the sheared metric.** Shearing the sampling lattice
   shears the reciprocal lattice, so grid modes no longer have
   |k| = √(kx² + ky² + kz²) — roughly kx → kx − a·ky. Both binners here assume
   a diagonal metric and take only `(Lx, Ly, Lz)`. Skip this and the binning is
   silently wrong in exactly the way the old cubic `box_dims` was.
2. **`round(0.25·i)` is a staircase, not a true shear**, so there is a residual
   ±0.5 pixel jitter between row groups. That is well under the ~1 deg beam
   (3.3 voxels), but it is a real distortion and should be checked rather than
   assumed harmless. A true shear needs interpolation, which correlates the
   noise.

---

## Changelog

### 2026-09-03 — repository restructure

Framework moved from the `Sampling Nb` working directory into this repository.
No numerics changed: `survey_grid` and `kbins_from_crop` reproduce
`S_starting_point_cropped_meta.json` bit for bit, which `tests/` now asserts.

- Geometry, k-binning, the linear system and the covariance samplers packaged as
  `imgibbs`. The grid derivation was triplicated across three notebooks,
  `make_kbins` appeared four times and `bin_it` three times; all are now single
  definitions.
- `scripts/run_gibbs.py` brought onto the current grid. It had been left on the
  old cubic setup — `box_dims = (232, 232, 232)`, `n_k_bins = 14`, `T_sys = 30`,
  `del_nu = 0.2 MHz`, the retired `S_starting_point.npy`, and
  `S[idxs == 0] = 1e30` — so it was silently sampling a different model from the
  notebook. It now shares every operator with `imgibbs` and refuses to start if
  the grid and the loaded `S` disagree.
- `S_trace` is now opt-in (`--save-S`). It is piecewise-constant over the
  k-bins, so `Pk_trace` plus the bin metadata reconstructs it exactly; writing
  it was 13 GB of the 26 GB a 500-sample chain produced.
- `binner` and `k_vecs` deleted. Both were unused but importable, and `k_vecs`
  hardcoded a cubic 2e3 Mpc box and returned integer-index |k| with the physical
  scaling commented out — it would have given silently wrong k on this grid.
- Dead imports removed (`corner`, `psutil`, `construct_Uf`, the unused scipy
  solvers). `#bin_fix` / `#EDIT` / `#xpk_fix` review markers stripped, the prose
  they carried kept.
- Two comments that contradicted their own code fixed: the `T_sys` note said
  30 K was retained when the code reads 16, and the `CROP` comment said
  `(70, 45, 500)` when the slice gives 250 channels.
- MeerKLASS observational cubes removed from the tree and gitignored;
  `imgibbs.data` locates them via `IMGIBBS_DATA`.

Notebook outputs are from the last run before this restructure and have not been
regenerated.

### 2026-08-24/25 — binning and comparison fixes

`kbins_auto.py` added (now `imgibbs/kbins.py`); the `S[idxs == 0]` modelling
change; the `T_gibbs` / `T_pca` injection comparison. See "Why the k-binning
changed" above.

### 2026-08 — non-cubic migration

FastBox patched for non-cubic grids; `Fastbox_Gen_Cropped.ipynb` written
against the real footprint, superseding the cubic `fastbox_gen.ipynb`; the
hardcoded `72`s removed from the sampling notebook.
