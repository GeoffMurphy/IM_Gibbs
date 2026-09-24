# Status

The working record for this repository: what the current configuration is, why
it is that way, and what is still wrong with it. Read this before trusting a
number out of here.

Last substantive update: 2026-09-23.

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

## Systematics: the ground-spill block

Added 2026-09-23. `imgibbs/systematics.py` plus an optional fourth block in
`construct_A` / `construct_b` / `construct_preconditioner`, so the model can be

    d = w * (Us s + Uf f + Ug g) + n

`sys_basis=None` is the default everywhere and leaves the published three-block
system numerically identical term for term — `tests/test_systematics.py` pins
that with an exact-equality check, not a tolerance.

### What is and is not identifiable, measured

The foreground block already has per-pixel free amplitudes on `n_modes` smooth
frequency modes, so **anything smooth in frequency is inside its span whatever
its spatial structure**. Fraction of a template's power the Legendre foreground
basis absorbs, on this band:

| template | n=6 | n=10 | n=20 |
|---|---|---|---|
| smooth spill, `nu^beta` | 1.0000 | 1.0000 | 1.0000 |
| ripple, 40 MHz period | 0.9945 | 1.0000 | 1.0000 |
| ripple, 20 MHz period | 0.2872 | 0.9886 | 1.0000 |
| ripple, 17.5 MHz period | 0.1837 | 0.9378 | 1.0000 |
| ripple, 10 MHz period | 0.0577 | 0.2155 | 0.9987 |
| ripple, 5 MHz period | 0.0149 | 0.0526 | 0.1954 |

Two consequences, and they are the entire design:

**The smooth part of ground spill is unidentifiable, and does not need to be
identified.** Absorbed to machine precision at every `n_modes`. Injecting 0.5 K
of it — 600x the H I rms — changes nothing. Putting a smooth template in `Ug`
would only add a direction the data cannot constrain, so
`spectral_templates(include_smooth=False)` is the default.

**The ripple is identifiable and lands in the signal bins.** A ripple of period
`P` is a *single* `k_parallel` mode at `k = 2*pi*B/(Lz*P)`, so it does not
spread — it dumps all of its power into one bin. With B = 52.04 MHz and
Lz = 254.39 Mpc: 20 MHz -> k = 0.064, 17.5 -> 0.073, 10 -> 0.129, against bin
centres `[0.068, 0.160, 0.374, 0.875, 2.046]`. Anything in the 10–20 MHz range
sits on the lowest one or two bins.

**Raising `n_modes` is not the fix.** The table says 20 modes absorbs the
ripple completely — but "`n_modes` is a trade with no unbiased setting" above
records that 20 modes removes the 21cm signal too (T(k) 0.006–0.06) and pushes
tau_int to 54–105. That is the argument for a tight-prior systematic block
rather than more free polynomials.

### Modelling decisions worth the reasoning

- **Cosine/sine quadratures, not amplitude and phase.** Amplitude and phase are
  a nonlinear pair; the two quadratures are linear parameters. That is what
  keeps the block inside the constrained realisation with no Metropolis step.
  The *period* is genuinely nonlinear and is **fixed**, not sampled.
- **Low-order spatial structure, not per-pixel.** Per-pixel amplitudes on a
  frequency template is exactly what `Uf` already is, so a systematic with that
  much spatial freedom is unidentifiable however distinctive its spectrum. The
  physics agrees: ground pickup is fixed in the telescope frame and varies only
  with pointing. Default is constant + scan-direction gradient, giving **four
  parameters** total.
- **`G` is a prior, not a sampled covariance.** `F` can be drawn from an
  inverse-Wishart because there are `Npix` independent amplitude vectors to
  estimate it from. `g` is a single vector of four numbers — its covariance is
  not identified by the data. It has to come from instrument characterisation,
  and `--gs-prior` is deliberately generous.
- **The Gram matrix factorises exactly.** Templates are separable, so
  `Ug^T Ug = kron(spatial Gram, spectral Gram)` — an identity, not an
  approximation, which is why the preconditioner block costs nothing.

### `construct_A` is not symmetric, and that predates this

Found while testing the new block. `Us(..., True)` is `rfftn`, which is the
*pseudo-inverse* of `irfftn`, not its transpose: `irfftn` sums each interior
mode with its conjugate, so the true adjoint is `2 x rfftn` there and
`1 x rfftn` on the `kz = 0` and `kz = Nz/2` planes. Measured on a small grid,
the blocks coupling `s` to real space differ from their transpose by exactly
that factor of 2; the `s-s`, `f-f` and `f-g` blocks are symmetric to 1e-17.

This is **pre-existing** — the signal-foreground block has always had it — and
**not** changed here. It does not affect the MAP solution: `b0` is built with
the same `Us(..., True)` convention, so the `s` rows are consistently scaled
and the linear system is equivalent. It is a question for the *sampling* step,
where the omega terms' covariance then does not exactly match the operator, on
the Hermitian-redundant modes only (2 of `Nz+2` packed planes, ~0.8% at
Nz = 250, and the `kz = 0` plane is already suppressed deliberately). Worth a
look; not touched, because changing the operator silently would re-baseline the
published sampler.

`test_groundspill_block_matches_the_existing_hermitian_convention` pins that
the new block follows the same convention, so there is one convention in the
operator and not two.

### Running it

    python scripts/run_gibbs.py 6 --groundspill --gs-period 17.5

writes a `g_trace` alongside the others. On real data you cannot tell a
systematic that was removed from one that was never there — the same lesson as
"The comparison cannot work without injection" — so the thing to run first is

    python scripts/systematics_injection.py --arm off --n-samples 120
    python scripts/systematics_injection.py --arm on  --n-samples 120
    python scripts/systematics_injection.py --summarise

which builds a synthetic cube (simulated H I + the real cube's Legendre
foreground + ground spill + noise) on the live grid and samples it twice with
identical seeds, differing only in whether the block is on.

**The recoverable fraction is capped by the table above.** The table is a
*power* fraction, so the ceiling on a recovered *amplitude* is
`sqrt(1 - absorbed)` — 90.3% at 17.5 MHz with 6 modes. Both scripts print the
ceiling before they start, and warn if the period chosen is one the foreground
already takes.

`notebooks/4_systematics.ipynb` shows the structure of the block — the
templates, the separability, and what a 6-mode clean leaves behind — without
running the sampler. It needs only data shipped with the repository.

### Measured, 2026-09-23

80 samples per arm, burn 20, identical seeds, on the live grid. Injected:
1e-3 K ripple at 17.5 MHz (1.16x the simulated H I rms) plus 0.5 K of smooth
spill. Footprint-restricted P(k), as a ratio to the true injected H I power:

| bin | k | s + f | s + f + g |
|---|---|---|---|
| 0 | 0.0684 | **88.49** | **15.04** | <- the ripple |
| 1 | 0.1599 | 1.03 | 0.78 |
| 2 | 0.3740 | 0.90 | 0.88 |
| 3 | 0.8748 | 1.00 | 0.99 |
| 4 | 2.0462 | 1.07 | 1.07 |

- The 0.5 K smooth component is **600x the H I rms and does nothing**, in
  either arm. That is the degeneracy table working as advertised.
- The contamination is **confined to one bin**, as a single `k_parallel` mode
  must be. Away from it, both arms recover the injected P(k) to within ~10%.
- The block cuts the contaminated bin from 88x to 15x — a large improvement,
  and **not a fix**. Bin 0 is still an order of magnitude high.

> **Estimator note, and it is not a detail.** These are per-sample footprint
> P(k), averaged over samples. An earlier draft of this table computed P(k)
> from the *posterior-mean* signal cube instead, which gave 0.85 / 0.64 / 0.36
> / 0.07 in bins 1-4 and looked like a severe high-k bias. It is not: the
> posterior mean is Wiener-suppressed, so its power is biased low by
> construction. The same distinction is already recorded above, where the Gibbs
> and PCA curves "legitimately diverge" above k = 0.24 for exactly this reason.
> Average the per-sample spectra, never the cubes.

### Two more systematics, 2026-09-23

Both reuse `SystematicBasis`, so the linear system, preconditioner and
samplers are untouched. `scripts/systematics_injection.py` (renamed from
`groundspill_injection.py`) takes `--systematic {groundspill,onef,leakage}`;
use a separate `--out` per systematic.

**Polarisation leakage.** Faraday rotation makes the leaked signal oscillate
as `cos(2 chi_0 + 2 RM lambda^2)`, carried as quadratures so `chi_0` is linear
and `RM` is the one fixed nonlinear parameter. The band spans
`lambda^2 = 0.0859-0.0953 m^2`, so the cycle count is `RM * 0.0095 / pi`, and
the 6-mode foreground basis absorbs:

| RM (rad m^-2) | 10 | 100 | 300 | 500 | 1000 | 2000 |
|---|---|---|---|---|---|---|
| absorbed | 1.000 | 1.000 | 1.000 | 0.976 | 0.186 | 0.047 |

**Ordinary Galactic Faraday depths are tens of rad m^-2, so leakage from them
is invisible here** — smooth across 52 MHz and absorbed whole, exactly like
smooth ground spill and harmless for the same reason. Only `RM >~ 500` is
identifiable. Widening the band is what buys sensitivity to lower RM; this is
an argument for the 500-channel cut if leakage ever matters.

Being periodic in `lambda^2` it is a chirp: the local frequency period goes as
`nu^3` and drifts 17% across the band. **On this band that does not spread it.**
The `k_parallel` spacing is `2*pi/Lz = 0.0247`, i.e. 33% at `k ~ 0.074`, so a
17% drift falls inside one mode — 98.5% of the power sits within 10% of the
peak, against 99.8% for a fixed-period ripple, and both land wholly in bin 0.
At 52 MHz leakage and ground spill are indistinguishable from the binned P(k)
alone; separating them is what the model is for. The chirp resolves on a wider
band, which is one more argument for the 500-channel cut.

(An earlier draft of this section claimed the power "spreads over a range of k
and cannot be dealt with by excising one bin". That is what the chirp would do
given enough bandwidth; it is not what it does here. Measured in
`notebooks/4_systematics.ipynb` §9.)

Spatially it uses a 2D polynomial, not the scan-direction one: leakage follows
the beam's polarisation response.

**1/f.** A stochastic process, so it has no natural low-rank basis — but it
has a covariance, and the leading Karhunen-Loeve modes of that covariance are
a basis in the form this module already uses. **That gives `G` a derivation
for once**: the eigenvalues *are* the prior variances, rather than coming from
instrument characterisation as they must for ground spill.

The common mode — a gain fluctuation moving every channel together — is
constant in frequency and so inside the foreground span to machine precision.
Passing `fg_basis` deflates it, and everything else the foreground can absorb,
out of the frequency covariance *before* the modes are taken. The retained
modes are then orthogonal to the foreground by construction, which is the
ground-spill lesson applied before the fact rather than after: the run log
reports `absorbs 0.0% of the onef templates`.

`best_fit_amplitudes()` scores recovery against the best the block could do
rather than the injected process, since a realisation is not exactly
representable in a truncated basis and charging the sampler for the truncation
would be wrong.

### Running on ilifu

`~/imgibbs-sys` is a clone of this branch; `scripts/submit_systematics.sh`
submits one arm per job. `data/L2021_polished_cube.npy` is symlinked to the
copy under `Sampling Nb/` — same file, md5 `204516c8a3a7484e6bbf5c5fc41a401a`.
The interpreter is `~/ska/.venv` (Python 3.12.13), which already carries
numpy, scipy, pyccl and tqdm; imgibbs needs nothing else, and fastbox is
optional since `imgibbs.grid` imports it only to check the cosmology.

**Validated against the laptop 2026-09-23**, which is worth doing rather than
assuming — pyccl 3.3.0/numpy 2.4.2 here against 3.3.6/2.5.3 there:

- `box_dims` bit-identical to all 15 decimals;
- the per-voxel bin assignment `idxs` bit-identical, md5
  `63b43f2c09533c439f090da11c5a768b`;
- a 5-sample chain at the same seed agrees to ~12 significant figures in
  every bin, the residual being BLAS/FFT ordering.

Two tests failed there before this check and neither was a real difference;
both are now version-robust. See the changelog. The second one is worth
knowing about: on a **cubic** toy grid many modes share exactly the same `|k|`
and a whole shell sits on a bin edge, where numpy 2.4 and 2.5 tie-break
differently. The production grid is non-cubic and has no such ties — so the
geometry that caused this project so much trouble is exactly what makes its
binning reproducible.

### 2500-sample run, 2026-09-23 — `outputs/groundspill_run1`

Four arms, 2500 samples each, identical chain seed and identical H I /
foreground / noise realisation. Traces flat from ~sample 250; ESS 100–2250 in
bins 0–3. Bin 4 has tau_int ~180 (ESS 12) in every arm — it is noise-dominated
and always has been, see "Smaller open items".

Footprint-restricted P(k), per sample then averaged, as a ratio to the true
injected H I:

| bin | k | control (no systematic) | s + f | s + f + g | s + f + g, S held |
|---|---|---|---|---|---|
| 0 | 0.0684 | **0.13** | 88.57 | 15.29 | 3.15 |
| 1 | 0.1599 | **0.59** | 1.02 | 0.77 | 0.78 |
| 2 | 0.3740 | 0.86 | 0.89 | 0.87 | 0.91 |
| 3 | 0.8748 | 0.99 | 1.00 | 1.00 | 0.99 |
| 4 | 2.0462 | 1.17 | 1.17 | 1.15 | 1.06 |

Ground-spill amplitude, and map correlation against the truth:

| arm | g[0,0] / true | map r | slope |
|---|---|---|---|
| control | — | 0.725 | 0.50 |
| s + f | — | 0.416 | 0.50 |
| s + f + g | 76.4% | 0.626 | 0.51 |
| s + f + g, S held | **90.1%** | 0.705 | 0.52 |

**Posterior width, (84th-16th)/2 as a fraction of the median:**

| arm | estimator | bin 0 | bin 1 | bin 2 | bin 3 | bin 4 |
|---|---|---|---|---|---|---|
| control | footprint, per sample | 10.3% | 1.6% | 0.5% | 1.2% | 4.5% |
| control | `S` draw | 14.9% | 2.2% | 0.8% | 1.3% | 4.6% |
| s+f+g | footprint, per sample | 1.2% | 1.4% | 0.5% | 1.3% | 3.2% |
| s+f+g | `S` draw | 5.2% | 2.2% | 0.8% | 1.3% | 3.2% |

**Do not read the contaminated bin's error bar as an accuracy.** In bin 0 the
s+f+g posterior is 1.2% wide and the answer is 15x high — *tighter* than the
control's 10.3% because the contamination pins the chain. Same behaviour as the
`g` pulls of -22: the width measures the conditional draw, not the model error.

**Bin 1 in the s+f+g arm looks better than the control, and is not.** It reads
0.77 against the control's 0.59, which invites the reading that the block
improves recovery there. It does not — that is leftover ripple power partly
filling the foreground-induced deficit. The ripple residual contributes 0.17x
the H I power to bin 1 (measured in `notebooks/4_systematics.ipynb`), and
0.59 + 0.17 = 0.76 against the measured 0.77. Bin 1 is contaminated too; it
just happens to be contaminated in the direction that hides the loss.

- The block roughly halves the map damage: r goes 0.416 -> 0.626, and with S
  held 0.705, against the control's 0.725. `figures/hi_map.png` shows why —
  without the block the systematic's **scan-direction gradient is imprinted
  straight onto the recovered H I map**, a red-to-blue ramp across RA.
- The slope is ~0.50 in every arm including the control. That is the
  posterior-mean Wiener suppression, not a bias introduced by the systematic.

### The control arm loses low-k power, and it is the foreground basis

**This is the result that matters most, and it has nothing to do with
systematics.** With *nothing* injected, the sampler recovers **13% of the true
P(k) in bin 0 and 59% in bin 1**. The trace is flat from sample 250, so it is
converged, not burn-in.

It is not signal-to-noise. Thermal noise P(k) is flat at 1.225e-6 K^2, so:

| bin | H I / noise | recovered |
|---|---|---|
| 0 | 5.84 | 0.13 |
| 1 | 4.25 | 0.59 |
| 2 | 2.76 | 0.86 |
| 3 | 0.53 | 0.99 |
| 4 | 0.05 | 1.17 |

The *most* signal-dominated bin recovers worst and the noise-dominated bins
recover perfectly — the opposite of a noise explanation.

It is the foreground basis. `kbins_from_crop` drops `kz = 0`, so the lowest
surviving modes are 1, 2, 3 cycles across the 52 MHz band — periods of 52, 26
and 17 MHz. Project each `kz` mode onto the 6-mode Legendre basis, average
within each k-bin, and the predicted surviving fraction is:

| bin | mean absorbed | predicted survival | measured |
|---|---|---|---|
| 0 | 0.693 | 0.307 | 0.13 |
| 1 | 0.319 | 0.681 | 0.59 |
| 2 | 0.097 | 0.903 | 0.86 |
| 3 | 0.002 | 0.998 | 0.99 |
| 4 | 0.000 | 1.000 | 1.17 |

Bins 1–4 are predicted from first principles with no free parameters. Bin 0
measured lower than predicted (0.13 vs 0.31), so a fifth arm — control with `S`
held at the prior — was run to test whether the `S` step supplies the rest. It
does:

| bin | predicted from the FG basis | control, S held | control, S sampled |
|---|---|---|---|
| 0 | 0.307 | **0.35** | **0.13** |
| 1 | 0.681 | 0.72 | 0.59 |
| 2 | 0.903 | 0.89 | 0.86 |
| 3 | 0.998 | 0.99 | 0.99 |
| 4 | 1.000 | 1.06 | 1.17 |

With `S` held, every bin matches the first-principles prediction. So the low-k
loss splits cleanly in two: **the foreground basis takes most of it (1.00 ->
0.31 in bin 0), and the `S` sampling step takes a further 2.7x on top (0.35 ->
0.13)**, and 1.2x in bin 1.

### One mechanism, both directions

The `S` step and the ripple result are the same pathology seen twice:

- **Contaminated bin:** the ripple adds power, `S` inflates, the signal block
  can afford to absorb more of the contaminant, `g` gives it up. 90.1% -> 76.4%.
- **Foreground-degenerate bin:** the foreground removes power, `S` deflates,
  the signal block is penalised for holding power there, and loses more. 0.35
  -> 0.13.

In both cases `S` is estimated from a signal field whose power has already been
distorted by another component, and the next draw amplifies the distortion.
`S` has no anchor other than the data it is trying to help separate. This is
also the shape of the suspected inpainting feedback below — three symptoms, one
cause, and worth treating as one problem.

**This bears directly on the `S_samp` task left to the student** (see the MSc
scope note): estimating `S` from footprint-restricted power addresses the
inpainting arm of it. It does not by itself address the degeneracy arm.

**Why this matters beyond this experiment.** The README says foregrounds are
marginalised over rather than projected out, "so the signal loss that a PCA
clean incurs at low k does not have to be corrected for after the fact". The
measured low-k loss here is comparable to the PCA transfer function recorded
above for the same `n_modes = 6` (T(k) 0.10–0.26 at low k). These are not
identical quantities — `T_pca` is the fraction of an *injected* mock that
survives a clean, this is the ratio of recovered to true P(k) for the signal
actually present — so a like-for-like comparison is still owed. But the claim
should not be repeated until that comparison is done.

> **Done, 2026-09-24** — see "The PCA comparison, done like-for-like" below.
> The claim holds at every bin except the lowest, where PCA (0.16) and the
> sampler (0.13) lose the same ~85%. The README needs narrowing to match.

### The `S` step steals from `g` — open, and the most interesting result

`g` is recovered at **76%** of the injected amplitude, against a ceiling of
90.3%. The traces say why, and they are unambiguous:

```
g[0,0]     9.14e-4 -> 8.40e-4 -> 7.53e-4 -> ... -> 7.33e-4   (true 9.58e-4)
P(k) bin 0 8.81e-6 -> 4.45e-5 -> 8.79e-5 -> ... -> 9.90e-5   (true 7.15e-6)
```

At iteration 0, before `S` has adapted, `g` is recovered almost perfectly and
bin 0 is close to the true H I power. As the chain runs, `S` in bin 0 inflates
on the contaminated data, which makes it progressively cheaper for the *signal*
block to absorb ripple power — and `g` gives it up. The two traces are
anti-correlated throughout.

This is the same class of problem as the suspected inpainting feedback below:
`S` is estimated from data that contains the thing `S` is then used to
separate. It is worth treating as one problem rather than two.

Consequences for anyone using this:

- **Do not read the `g` posterior width as an error bar.** It is ~1e-5 while
  the bias is ~2.2e-4, so the pulls are -22 and -7. The scatter measures the
  conditional draw, not the model competition.
- A tighter `--gs-prior` will not help; it shrinks `g` further, the wrong way.
- Directions not yet tried: hold `S` fixed at the prior in the contaminated
  bin, sample the systematic before `S` rather than after, or down-weight the
  affected modes in the `S` draw.

### Open

- **The period is fixed, not sampled.** Getting it wrong is the interesting
  failure mode and is untested. A search over it needs a nonlinear step.
- **The preconditioner's `g` block uses the unmasked Gram**, consistent with
  the existing `N_inv_scalar` choice. For `f` the mask barely matters; for the
  spatial templates here the fill fraction is 59%, so the approximation is
  cruder. Convergence only, not correctness.
- **Only ground spill so far.** A multiplicative systematic (gain) does not fit
  this block at all — it would need linearisation or a separate Metropolis
  step.

## Picking this up again — state at 2026-09-24

Stopping point agreed after the deflation test, before the beam. Nothing is
half-finished on disk; the items below are decisions and follow-ups, in the
order they are worth doing.

1. **Fix `build_truth` and re-run the leakage deflated arm.** The only real
   loose end. Everything except ground spill is dispatched through
   `build_basis`, which applies `--deflate`, so the deflated leakage and 1/f
   runs *injected* a foreground-orthogonal signal rather than modelling the
   identifiable part of a realistic one. Build the truth from the undeflated
   basis and let `best_fit_amplitudes` project onto the deflated model basis,
   exactly as ground spill already does. ~2 line change, then a 2 h job.
   Until then **ground spill is the only clean evidence for deflation** — it
   is strong, but 1/f and leakage must not be cited as independent support.

2. **Make deflation the default for new bases.** The ground-spill result is
   unambiguous: recovery 76.4% -> 89.6% against a 90.4% ceiling, the
   contaminated bin 15.29 -> 0.12 against a control of 0.13, posterior width
   5x smaller and runtime 3x shorter. Any new systematic basis should be built
   with `fg_basis=evecs` unless there is a reason not to.

3. **Narrow the README's claim about low-k signal loss.** The like-for-like
   PCA comparison supports it at every bin except the lowest, where PCA (0.16)
   and the sampler (0.13) lose the same ~85%. As written the README overstates
   it.

4. **The beam.** The agreed next step, deliberately not started — see "No beam
   in the MODEL" below. Note that a chromatic beam turns spatial structure
   into spectral structure, which is the assumption every absorbed-fraction
   number in `imgibbs/systematics.py` rests on, so those tables need
   re-measuring once `B` exists.

### Where things live

- Branch `systematics`, unmerged. `main` contains none of this.
- ilifu: `~/imgibbs-sys`, run with `~/ska/.venv`, always through Slurm — the
  login node is throttled and blocks `rsync`. The transfer node
  (`ilifu-transfer`) needs its own OTP, so bulk cube transfer needs a human at
  a terminal. `systematics_report.py --export-arms/--import-arms` exists to
  avoid needing one: it ships ~5 MB per arm instead of 0.8 GB of cubes.
- `outputs/groundspill_run1` (local, gitignored, ~3 GB) holds five arms plus
  the imported deflated one. **Do not delete it** — the ilifu copy has only
  the deflated arm, and the baselines are a 7 h rerun.
- `pyccl` and `fastbox` are installed *nowhere* reachable, including ilifu.
  `systematics_report.py` and `pca_benchmark.py` are written to work without
  them; `notebooks/3_pca_transfer_function.ipynb` still needs both.

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

### 1/f and leakage, 2500 samples each, 2026-09-24 — and what the difference means

Run on ilifu, `~/imgibbs-sys/outputs/{onef,leakage}_run`. Same seed, same H I /
foreground / noise realisation, same 1e-3 K injected amplitude as the ground
spill run. Footprint-restricted P(k), ratio to the true injected H I:

| bin | k | control | **1/f** off / on | **leakage** off / on | ground spill off / on |
|---|---|---|---|---|---|
| 0 | 0.0684 | 0.13 | 106.44 / **0.11** | 79.65 / **11.84** | 88.57 / 15.29 |
| 1 | 0.1599 | 0.59 | 1.29 / 0.60 | 1.29 / 0.91 | 1.02 / 0.77 |
| 2 | 0.3740 | 0.86 | 0.90 / 0.86 | 0.92 / 0.89 | 0.89 / 0.87 |
| 3 | 0.8748 | 0.99 | 0.99 / 0.98 | 0.99 / 0.99 | 1.00 / 1.00 |
| 4 | 2.0462 | 1.17 | 1.17 / 1.16 | 1.22 / 1.16 | 1.17 / 1.15 |

Amplitude recovery:

| systematic | recovered | FG absorbs the templates |
|---|---|---|
| **1/f** | **100.3 - 103.5%** | **0.0%** (deflated by construction) |
| leakage | 71 - 90% | 18.6% |
| ground spill | 76% | 18.4% |

> **Read the 1/f row with care.** Its basis is deflated in *both* the truth and
> the model, so its 100% is partly by construction and is not independent
> evidence that deflation helps. See "But the leakage arm does not test this"
> below. The ground-spill `--deflate` arm is the clean test.

**1/f is removed completely.** Bin 0 goes 106 -> 0.11 against a control of
0.13, and every other bin lands on the control to two decimals. In
`figures/power_spectrum.png` the `s+f+g` curve lies on top of the control
everywhere. The amplitudes come back at 100%, not at a ceiling.

### The likely reason, and it is actionable

**The 1/f basis is the only one deflated against the foreground.**
`onef_basis` projects the foreground span out of the frequency covariance
*before* taking the KL modes, so its templates are orthogonal to `Uf` by
construction — the run log reports `absorbs 0.0%`. Ground spill and leakage use
physical templates with 18% of their power inside the foreground span, and
both recover badly and leave an order-of-magnitude residual in bin 0.

The hypothesis is that the residual is not the foreground absorbing its share
harmlessly, but a genuine `f`-`g` degeneracy: in the overlapping direction the
split is set by the priors, and with `G` much tighter than `F` neither block
claims it cleanly, so it lands in `s` and inflates `S` — the feedback recorded
above.

**Concrete test, and it is cheap:** deflate the ground-spill templates against
`evecs` the same way `onef_basis` does, and re-run the `on` arm. The prediction
is that recovery goes from 76% toward the 90.3% ceiling and bin 0 drops from
15x toward the control. If it does, deflation should become the default for
every basis, and the module's advice changes from "omit the smooth template"
to "project out the whole foreground span".

**Submitted 2026-09-24** as `--deflate` (jobs 13875233 ground spill, 13875234
leakage, 2500 samples, seed 1, same realisation as every other arm).

Reading the result when it lands:

- **Leakage is self-contained.** `outputs/leakage_run` on ilifu already holds
  `clean`, `off` and `on`, so the deflated arm joins them and
  `scripts/systematics_report.py --out outputs/leakage_run` gives the four-way
  comparison directly.
- **Ground spill is not.** Its baseline ran on the laptop
  (`outputs/groundspill_run1`), and only the deflated arm is on ilifu. Either
  compare against the recorded numbers above — 88.57 in bin 0, 76% recovery —
  which is valid because the seed and the realisation are identical and the
  data cube's md5 matches, or copy the deflated arm's `samples/` down into
  `outputs/groundspill_run1` and run the report on all six arms at once.

What counts as confirmation: `g` recovering ~100% **of the deflated target**,
which is itself 90.4% of the undeflated injected amplitude — the deflation
rescales `g_true` to the `sqrt(1 - 0.184)` ceiling, so 100% of it means the
block found everything it was given. Plus bin 0 dropping from 15x toward the
control's 0.13.

What would falsify it: recovery staying near 76% of the new target, or bin 0
staying high. That would mean the `f`-`g` degeneracy was not the mechanism and
the residual is something else — most likely the `S` feedback acting on its
own, in which case the `--fix-S` arm is the one to look at next.

### The PCA comparison, done like-for-like, 2026-09-24

The comparison owed above is now in `scripts/pca_benchmark.py`, overlaid on the
power spectrum figure with `systematics_report.py --pca`. Same simulated cube
the `clean` arm saw, same footprint estimator, same five `kbins_from_crop`
bins, same truth. Both methods then report the same quantity: **the fraction
of the true H I power that survives.**

| bin | k | PCA, 8 modes | Gibbs control | Gibbs `ondeflated` |
|---|---|---|---|---|
| 0 | 0.0684 | 0.162 | 0.13 | 0.12 |
| 1 | 0.1599 | 0.355 | **0.59** | 0.59 |
| 2 | 0.3740 | 0.460 | **0.86** | 0.86 |
| 3 | 0.8748 | 0.706 | **0.99** | 0.99 |
| 4 | 2.0462 | 1.000 | 1.17 | 1.15 |

**The sampler retains more signal than PCA everywhere except the lowest bin**,
and the gap is large in the middle: 0.86 against 0.46 at k = 0.37, 0.99 against
0.71 at k = 0.87. At bin 0 the two are equivalent (0.13 vs 0.16) — both lose
~85% — so the README's claim that marginalising avoids the low-k signal loss a
PCA clean incurs is **not** supported at the lowest bin, and is supported at
every other. That is a narrower claim than the README currently makes and it
should be rewritten to match.

Two things were needed to make the numbers comparable, and both change the
answer materially:

- **Noise debias.** PCA leaves the full thermal noise in the map; the Gibbs
  `s` field is Wiener-filtered. Raw, PCA "recovers" 21.8x the truth in bin 4,
  all of it noise. The script pushes independent noise-only draws through the
  same projector and subtracts. The debiased `(PCA - noise)/true` reproduces
  the exact `T(k)` to a few per cent in every bin, which validates both.
- **The transfer function is computed exactly**, as `P(F s_true)/P(s_true)`
  with the known signal and the same projector, rather than by injection. In
  a simulation the signal is known, so the injection estimator is a needless
  layer in front of a quantity already in hand.

**The injection estimator is badly biased with Gaussian mocks**, which is worth
recording because it is the on-sky procedure. With mocks built by randomising
the phases of the H I cube — reproducing its power spectrum mode for mode — the
injection TF came back 1.3-1.8x too high:

| bin | T exact | T injected (Gaussian mocks) | ratio |
|---|---|---|---|
| 0 | 0.162 | 0.210 | 1.30 |
| 1 | 0.355 | 0.581 | 1.64 |
| 2 | 0.460 | 0.813 | 1.77 |
| 3 | 0.706 | 0.984 | 1.39 |
| 4 | 1.000 | 0.995 | 1.00 |

`F = I - A A^T` acts along frequency, so in Fourier space it is a *coherent*
sum over `k_par` at fixed `k_perp`; the surviving power depends on the relative
phases across `k_par`, which a Gaussian field with the right P(k) does not
carry. Matching the power spectrum is not enough — the mocks have to match the
phase structure, i.e. be lognormal with RSD, as Cunnington et al. use.
Reproduce with `--injection-tf N`.

**Structural caveat on all of the above.** The simulated foreground is exactly
rank 6 by construction, so any clean with >= 6 modes removes it completely.
This flatters PCA relative to real data, where the foreground is only
approximately low rank. What the table measures is **signal loss, not
foreground residual** — the right quantity for this comparison, but not a claim
that PCA cleans the real sky this well. On real data PCA would also carry a
foreground residual that the Gibbs run, by construction, does not.

### Result, 2026-09-24 — confirmed for ground spill, confounded for leakage

Both jobs `COMPLETED`, exit 0 (13875233 ground spill 1:59:17, 13875234 leakage
2:02:35). Report: job 13878080.

**Ground spill is the clean test and it confirms the hypothesis.**
`groundspill_cube` builds the injected cube itself and never sees `fg_basis`,
so the deflated arm was given *the same contaminated data* as `on` and differs
only in the model basis.

| | injected | recovered | of injected |
|---|---|---|---|
| `on` (undeflated) | 9.578e-4 | 7.321e-4 +- 9.9e-6 | 76.4% |
| `ondeflated` | 9.578e-4 | 8.583e-4 +- 2.0e-6 | **89.6%** |

The ceiling is `sqrt(1 - 0.184)` = 90.37%, and `g_true` was rescaled to
8.656e-4 = 0.9037 x 9.578e-4, confirming the rescaling is correct. Recovery is
99.2% of what the block could reach. Footprint P(k), ratio to true H I:

| bin | k | control | ground spill `on` | ground spill `ondeflated` |
|---|---|---|---|---|
| 0 | 0.0684 | 0.13 | 15.29 | **0.12** |
| 1 | 0.1599 | 0.59 | 0.77 | 0.59 |
| 2 | 0.3740 | 0.86 | 0.87 | 0.86 |
| 3 | 0.8748 | 0.99 | 1.00 | 0.99 |
| 4 | 2.0462 | 1.17 | 1.15 | 1.15 |

**The deflated arm is indistinguishable from the uncontaminated control in
every bin.** Bin 0 goes 15.29 -> 0.12 against a control of 0.13. The bin-1
excess (0.77 vs 0.59) also disappears — that excess was leftover ripple
filling the foreground-degenerate deficit, not improved recovery.

Two effects that were not predicted:

- **Error bars shrink 5x** (9.9e-6 -> 2.0e-6). Deflation tightens the
  posterior far more than it moves the mean, because the removed direction was
  the one carrying almost all the `f`-`g` covariance.
- **Runtime drops 3x**: 9.14 -> 2.86 s/sample, 6.3 h -> 2.0 h for 2500
  samples. The near-degenerate direction was what LGMRES was struggling on.
  Deflation is a conditioning fix as much as a bias fix.

**So deflation should become the default**, as the test proposed. The module's
advice changes from "omit the smooth template" to "project out the whole
foreground span".

### But the leakage arm does not test this, and neither did 1/f

`build_truth` dispatches everything except ground spill through `build_basis`,
which applies `--deflate`. So the deflated leakage run **injected a different
signal** — drawn from the already-orthogonalised basis — rather than injecting
the same leakage and modelling only its identifiable part. The tell is in the
logs: `g_true` moved 7.420e-4 -> 7.408e-4, a ratio of 0.998, where ground
spill's moved by the expected 0.904.

Its numbers (bin 0 = 0.10, amplitudes 95.5 - 101.9%) are therefore close to
tautological: model basis = injection basis, nothing degenerate to lose. They
show a clean systematic is recoverable, not that deflating a realistic one
helps.

**The same wiring affects 1/f**, since `onef_basis` always deflates, in both
truth and model. The table above already labels it "deflated by construction",
but its 100% recovery was then used as evidence *for* deflation, which
double-counts. **Ground spill is the only clean evidence.** It does hold, and
it holds strongly, but the 1/f and leakage rows should not be cited as
independent confirmation.

**Fix before this goes in a paper:** build the truth from the *undeflated*
basis and let `best_fit_amplitudes` project onto the deflated model basis —
exactly what ground spill already does. Then re-run the leakage `ondeflated`
arm. Roughly a two-line change in `build_truth` plus a 2 h job.

### Reporting bugs found while checking the above

Cosmetic, but they mislead anyone reading a run log:

- **`absorbed` is a power fraction**, so `scripts/systematics_injection.py`
  printing `ceiling on recoverable amplitude: 1 - absorbed` is wrong: for
  ground spill it printed 81.6% where the amplitude ceiling is
  `sqrt(0.816)` = 90.3%. `g_true` is computed by `best_fit_amplitudes` and is
  unaffected, so only the printed line is wrong.
- **For non-groundspill systematics `absorbed` is measured after deflation**,
  because the probe comes from `build_basis` which applies `--deflate`. That
  is why the deflated leakage log claims `absorbs 0.0%` where the undeflated
  run correctly reported 18.6%.
- **`scripts/systematics_report.py` prints nonsense percentages** for
  templates whose true amplitude is ~1e-12 (`-3670896.7%`). Suppress the
  column when `|true|` is below the error bar.

### Smaller things from the same run

- **Both 1/f and leakage contaminate bin 1**, unlike ground spill: `off` reads
  1.29 in bin 1 against the control's 0.59. For 1/f that is expected — it is
  broadband. For leakage it is the chirp's wings, which
  `notebooks/4_systematics.ipynb` §9 predicted at ~10x the ground-spill level.
  The block removes most of it (1/f 0.60, leakage 0.91).
- **`--fix-S` chains mix far better.** With `S` held, `tau_int` is 1.0 in every
  bin (ESS 2250) against 6-28 when it is sampled. Sampling `S` is most of the
  autocorrelation in this sampler.
- **Bin 4 remains poorly converged in every arm**, `tau_int` 144-202, ESS ~12 at
  2500 samples. Unchanged from the ground spill run and from the pre-systematics
  numbers; it is noise-dominated and always has been.
- Runtimes on ilifu, 8 CPUs: 1.4-9.1 s/sample depending on arm.

### No beam in the MODEL — the next thing to add

Distinct from "No beam in the simulation" below, which is about the truth
curve. This is about the forward model itself:

    d = w * (B Us s + B Uf f + Ug g) + n

**Which components B acts on is the structurally important part**, and it is
not "all of them":

| component | beam? | why |
|---|---|---|
| signal `s` | **yes** | it is sky |
| foreground `f` | **yes** | also sky |
| ground spill, 1/f (`g`) | **no** | far-sidelobe and receiver effects; they enter *after* the main beam |
| polarisation leakage | **no** (own operator) | leakage *is* a beam effect — the leakage beam, not the total-intensity one |
| noise `n` | **no** | |

So the systematics block sits **outside** B. That is a good reason it belongs
where it is, and it means adding a beam does not disturb it.

**The chromaticity is the point, not the smoothing.** For a 13.5 m dish at
1.22 lambda/D the FWHM runs 1.60 deg at 970.9 MHz to 1.52 deg at 1023.0 MHz —
a 5.4% change across the band. A frequency-dependent beam turns *spatial*
structure into *spectral* structure, which is the standard mode-mixing
problem, and it directly undermines the assumption everything in the
systematics work rests on: that the foreground is smooth in frequency in each
pixel. It is smooth per pixel only for an achromatic beam. With a chromatic
one the foreground acquires spectral structure proportional to its own spatial
gradients, and **the absorbed-fraction table above would have to be
re-measured.** Nothing in that table survives a chromatic beam unexamined.

At 1.55 deg the beam is 44.6 Mpc, 5.2 voxels, and cuts off around
`k_perp ~ 0.141 Mpc^-1` — i.e. between bin 0 (0.068) and bin 1 (0.160). It
suppresses transverse power across the whole range the measurement lives in.

> **The "~1 deg FWHM, 26.6 Mpc, 3.3 voxels" in the simulation item below does
> not match this band.** 1 deg is about right near the top of L band; at
> 970-1023 MHz the same formula gives ~1.55 deg. Neither number should be
> trusted for real work — take the FWHM from the MeerKLASS beam model rather
> than from `1.22 lambda/D`, which assumes uniform illumination and is known
> to be narrow for a tapered feed.

**Implementation.** B is diagonal in `(k_perp, nu)`: transform the two
transverse axes, multiply per channel, transform back. Cheap as an operator.
It is **not** diagonal in the 3D Fourier basis, because the frequency
dependence mixes `k_parallel` — so `construct_preconditioner`'s
`M00_inv = 1/(1/S + N_inv)`, which assumes `Us` is unitary, stops being right.
This is the same class of problem as `N_inv_scalar` not being allowed to carry
the mask (see the note on that above): the exact operator is dense in the
basis the preconditioner needs diagonal. The usual approximation is a
band-averaged `|B(k_perp)|^2` on the diagonal.

**Related but separate work.** Geoff is investigating beam effects on
*observational* data in a separate run out of `Sampling Nb/` — that is where
the numbers for what the beam actually does should come from, rather than from
a formula here. There is also a distinct MeerKLASS ripple-beam-correction
project in `museek` (UHF band, `~/beam_correction/NOTES.md`) aimed at a
separate paper; the physics overlaps, the code does not. Check the boundary
before building anything here.

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

### 2026-09-24 — deflation confirmed, and the PCA comparison settled

- `--deflate` results read. Ground spill is the clean test and confirms the
  hypothesis: 76.4% -> 89.6% recovery against a 90.4% ceiling, contaminated
  bin 15.29 -> 0.12 against a control of 0.13, posterior width 5x smaller,
  runtime 3x shorter (9.14 -> 2.86 s/sample).
- Found that `build_truth` applies `--deflate` to the *injected* signal for
  everything except ground spill, so the leakage and 1/f deflated arms do not
  test the hypothesis. Recorded rather than silently fixed, since the fix
  implies a re-run.
- `scripts/pca_benchmark.py` — the like-for-like PCA comparison STATUS had
  listed as owed, on the sampler's own k-bins, cube, estimator and truth. The
  transfer function is computed exactly rather than by injection; the
  injection estimator with Gaussian mocks is 1.3-1.8x biased, because the PCA
  projector's action depends on phase structure a Gaussian mock does not
  carry.
- `systematics_report.py`: `--pca` overlays that benchmark; `--export-arms` /
  `--import-arms` move one arm between machines as ~5 MB instead of 0.8 GB;
  `box_dims` falls back to the value each run records, so the report no longer
  needs `pyccl` (which is installed nowhere reachable, including ilifu).
- Reporting fixes: the absorbed fraction is a *power* fraction, so the printed
  amplitude ceiling is now its square root (81.6% -> 90.3%); the absorbed
  probe is measured on undeflated templates, where under `--deflate` it had
  been reporting 0.0% by construction; recovery percentages are suppressed
  where the true amplitude is below the posterior width, instead of printing
  -3670896.7%.


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
- FastBox is a dependency rather than vendored, pinned to a fork.

### FastBox: the non-cubic patch is obsolete

The local FastBox checkout at `/home/geoff/FastBox` sits on the `joss-paper`
branch with the non-cubic patch applied on top, and its `origin/main` ref had
not been fetched since **2022-05-05**. Upstream has moved a long way since:

- **Upstream `main` already supports non-cubic grids.** `CosmoBox` takes
  `nsamp` as an `int` or a 3-tuple and sets `Nx`/`Ny`/`Nz`, with the same
  `boxfactor` correction the local patch made.
- It also fixes non-cubic bugs the patch never touched — `filters.py` was
  building `ky` from `field.shape[0]` and meshgridding without `indexing='ij'`,
  so the *vendored* copy is the more broken one on this grid.
- It adds `meerklass.py`, `power.py`, and a MeerKLASS geometry example.
- `voids` imports cleanly again, so commenting it out of `__init__.py` is no
  longer needed.

Verified against this repository on upstream `883677f`: `CosmoBox` with a
3-tuple `nsamp`, `realise_density`, `lognormal`, `realise_velocity`,
`redshift_space_density`, `HITracer` and `pca_filter` all run end-to-end, and
`signal_amplitude()` returns `Tb_mK = 0.15048796349807` — bit-identical to the
value recorded in `S_starting_point_cropped_meta.json`.

So no patch branch was published. `GeoffMurphy/FastBox` is a plain fork of
upstream `main`, and exists as a **pin** rather than a patch: the simulated
P(k) is the truth curve the sampler is judged against, so an upstream change
should be adopted deliberately. Sync the fork when you want to.

Note also that FastBox's `setup.py` declares `'license': 'MIT'`, even though
the repository has no LICENSE file — so redistribution would have been
permitted after all. The dependency is still the cleaner arrangement.

`scripts/make_paper_figures.py` from that checkout -- untracked, on no upstream
branch, and the thing that generates the two JOSS paper figures -- is preserved
at `GeoffMurphy/FastBox` on `joss-paper` (`50d3b2e`). It was pushed to the fork
rather than to `philbull/FastBox`, which is a shared branch, though push access
to upstream does exist.

**Open:** `/home/geoff/FastBox` still holds the superseded non-cubic patch as
uncommitted changes on `joss-paper`. Nothing depends on it now; discard it or
rebase what is still wanted onto current `main`.

Notebook outputs stripped — the notebooks ship bare. They had not been
regenerated against the refactor, so keeping them would have shown output that
no longer corresponded to the code above it. It also took the three notebooks
from 2.0 MB to 119 KB.

### 2026-08-24/25 — binning and comparison fixes

`kbins_auto.py` added (now `imgibbs/kbins.py`); the `S[idxs == 0]` modelling
change; the `T_gibbs` / `T_pca` injection comparison. See "Why the k-binning
changed" above.

### 2026-08 — non-cubic migration

FastBox patched for non-cubic grids; `Fastbox_Gen_Cropped.ipynb` written
against the real footprint, superseding the cubic `fastbox_gen.ipynb`; the
hardcoded `72`s removed from the sampling notebook.
