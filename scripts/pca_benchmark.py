"""PCA + transfer function benchmark, on the sampler's own k-bins.

``notebooks/3_pca_transfer_function.ipynb`` measures the PCA benchmark on the
box-based ``make_kbins`` grid with its own band and mode count, so its ``T(k)``
cannot be laid on top of a systematics run: different bins, different
estimator, different field. ``docs/STATUS.md`` records that like-for-like
comparison as owed. This script is it.

Everything is held identical to the Gibbs run being compared against:

* the **same simulated cube** the ``clean`` arm saw -- H I + the 6-mode
  Legendre projection of the real L2021 foreground + noise, on the real
  footprint, from the same ``TRUTH_SEED``;
* the **same footprint-restricted estimator** (mask, mean over valid voxels,
  divide by ``<w^2>``) on the **same five ``kbins_from_crop`` bins**;
* the **same truth**, ``Fastbox_cube_cropped.npy``.

Both methods then produce the same quantity -- the fraction of the true H I
power that survives -- so they can go on one axis.

The transfer function here is **exact, not estimated**
--------------------------------------------------------
``T(k) = P(F s_true) / P(s_true)``, where ``F = I - A A^T`` is the PCA
projector. The signal is known, so the loss can be measured directly instead
of being inferred by injecting mocks. The injection procedure exists because
on sky the signal is unknown; in a simulation it is an unnecessary estimator
in front of a quantity already in hand.

That matters more than it sounds, because the injection estimator is **badly
biased unless the mocks match the signal's phase structure**. Measured here
with mocks built by randomising the phases of the H I cube -- which reproduces
its power spectrum mode for mode -- ``T`` came out 0.585 in bin 1 against a
true 0.355, a factor of 1.6. ``F`` acts along frequency, so in Fourier space it
is a coherent sum over ``k_par`` at fixed ``k_perp``; the result depends on the
relative phases across ``k_par``, which a Gaussian mock with the right P(k)
does not reproduce. ``--injection-tf`` recomputes that comparison.

**One structural caveat.** The simulated foreground is *exactly* rank 6 by
construction, so any clean with >= 6 modes removes it completely. This flatters
PCA relative to real data, where the foreground is only approximately low rank.
What is measured here is therefore **signal loss, not foreground residual** --
which is what a transfer function is for, but it is not a claim that PCA cleans
the real sky this well.

Usage
-----
    python scripts/pca_benchmark.py --out outputs/groundspill_run1
    python scripts/pca_benchmark.py --pca-modes 6 --injection-tf 50

Writes ``<out>/pca_benchmark.npz``, which
``scripts/systematics_report.py --pca <path>`` overlays on the power spectrum
figure.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imgibbs import kbins_from_crop, load, load_l2021_cube  # noqa: E402

CROP = (slice(33, 103), slice(14, 59), slice(0, 250))
N_K_BINS = 5
T_SYS, DEL_T = 16.0, 1000.0
DEL_NU = (1712.0 - 856.0) / 4096 * 1e6
TRUTH_SEED = 20260923


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--out', default='outputs/groundspill_run1',
                   help='run directory to read box_dims from and write into')
    p.add_argument('--pca-modes', type=int, default=8,
                   help='PCA modes removed (notebook 3 uses 8; the sampler '
                        'marginalises over 6 Legendre modes)')
    p.add_argument('--n-modes', type=int, default=6,
                   help='Legendre modes in the simulated foreground, matching '
                        'the run being compared against')
    p.add_argument('--injection-tf', type=int, default=0, metavar='N',
                   help='also run the on-sky injection estimator with N '
                        'Gaussian mocks, to show how far it lands from the '
                        'exact T(k) above')
    p.add_argument('--n-noise', type=int, default=20,
                   help='noise-only draws averaged for the noise debias')
    p.add_argument('--seed', type=int, default=7,
                   help='seed for the mock phases (independent of TRUTH_SEED)')
    return p.parse_args()


def pca_projector(cube, n_modes, valid):
    """The leading ``n_modes`` frequency eigenvectors, as (n_freq, n_modes)."""
    d = cube[valid]
    cov = (d.T @ d) / d.shape[0]
    vals, vecs = np.linalg.eigh(cov)
    return vecs[:, np.argsort(vals)[::-1][:n_modes]]


def apply_projector(cube, A):
    flat = cube.reshape(-1, cube.shape[2])
    return (flat - (flat @ A) @ A.T).reshape(cube.shape)


def pca_clean(cube, n_modes, valid):
    """Project out the leading ``n_modes`` frequency eigenmodes.

    Matches ``PCA_clean`` in notebook 3: the covariance is built from pixels
    that are non-zero in **every** channel, and no mean spectrum is removed
    before the eigendecomposition.
    """
    A = pca_projector(cube, n_modes, valid)
    return apply_projector(cube, A)


def phase_randomised(cube, rng):
    """A Gaussian field with this cube's power spectrum, mode for mode.

    Keep ``|F|`` and take the phases from a white-noise field. Both factors
    carry the Hermitian symmetry of a real field, so the product does too and
    the inverse transform is real to rounding.
    """
    F = np.fft.fftn(cube)
    G = np.fft.fftn(rng.normal(size=cube.shape))
    G /= np.abs(G) + 1e-300
    return np.fft.ifftn(np.abs(F) * G).real


def main():
    args = parse_args()

    metas = sorted(glob.glob(os.path.join(args.out, 'run_*_meta.json')))
    if not metas:
        raise SystemExit(f'no run_*_meta.json under {args.out}')
    meta = json.loads(Path(metas[0]).read_text())
    box_dims = np.array(meta['box_dims'], float)
    sig_k = np.array(meta['sig_k'], float)

    real = load_l2021_cube()[CROP]
    shape = real.shape
    n_freq = shape[2]
    flag = (real != 0).astype(float)
    hi = load('Fastbox_cube_cropped.npy')[:, :, :n_freq]

    # The clean arm's cube, rebuilt exactly: same Legendre foreground, same
    # seed, and -- because the control arm injects no systematic -- the noise
    # is the first draw off the generator, as it was in the run.
    leg = np.linalg.qr(np.polynomial.legendre.legvander(
        np.linspace(-1, 1, n_freq), args.n_modes - 1))[0].T
    foreground = ((real.reshape(-1, n_freq) @ leg.T) @ leg).reshape(shape)
    rng = np.random.default_rng(TRUTH_SEED)
    noise_rms = np.sqrt(T_SYS ** 2 / (DEL_NU * DEL_T))
    noise = rng.normal(scale=noise_rms, size=shape)
    cube = (hi + foreground + noise) * flag

    _, idxs, _ = kbins_from_crop((hi + 1) * flag, box_dims, max_bins=N_K_BINS)
    occupied = np.unique(idxs)
    occupied = occupied[occupied > 0]
    assert len(occupied) == len(sig_k), (len(occupied), len(sig_k))

    mask = flag == 1
    w2 = (flag ** 2).mean()

    def pk(a, b=None):
        """The report's footprint estimator, as an auto- or cross-spectrum."""
        A = np.fft.fftn((a - a[mask].mean()) * flag, norm='ortho').flatten()
        if b is None:
            B = A
        else:
            B = np.fft.fftn((b - b[mask].mean()) * flag,
                            norm='ortho').flatten()
        p = (A * np.conj(B)).real
        return np.array([p[idxs == i].mean() for i in occupied]) / w2

    true_pk = pk(hi)
    valid = np.all(real != 0, axis=2)

    A = pca_projector(cube, args.pca_modes, valid)
    cleaned = apply_projector(cube, A)

    # PCA leaves the thermal noise in the map; the Gibbs `s` field is
    # Wiener-filtered, so its noise is suppressed. Left in, this reads as PCA
    # "recovering" 22x the truth in the top bin, which is entirely noise.
    # Estimate the noise power surviving the SAME projector from independent
    # noise-only draws and subtract it, so both curves estimate the signal.
    nrng = np.random.default_rng(args.seed + 1000)
    noise_pk = np.mean([pk(apply_projector(
        nrng.normal(scale=noise_rms, size=shape) * flag, A))
        for _ in range(args.n_noise)], axis=0)
    pk_pca_raw = pk(cleaned)
    pk_pca = pk_pca_raw - noise_pk

    # The exact transfer function: push the KNOWN signal through the SAME
    # projector. No mocks, no estimator, no realisation scatter.
    T = pk(apply_projector(hi * flag, A)) / true_pk

    print(f'grid        : {shape}, {len(sig_k)} bins')
    print(f'PCA modes   : {args.pca_modes}   '
          f'(foreground is exactly rank {args.n_modes} by construction)')

    print('\n  bin       k      T(k) exact   noise/true   (PCA-noise)/true')
    for i, k in enumerate(sig_k):
        print(f'  {i:^3d} {k:8.4f}  {T[i]:10.4f}   {noise_pk[i] / true_pk[i]:10.2f}'
              f'   {pk_pca[i] / true_pk[i]:16.3f}')
    print('\n  The last two columns are a consistency check: with the '
          'foreground\n  exactly removed, (PCA - noise)/true should reproduce '
          'T(k).')

    T_inj = T_inj_std = None
    if args.injection_tf:
        # The estimator you are forced to use on sky, measured against the
        # exact answer above. Mocks are phase-randomised copies of the H I
        # cube: identical power spectrum, independent phases.
        mrng = np.random.default_rng(args.seed)
        acc = np.zeros((args.injection_tf, len(sig_k)))
        for i in range(args.injection_tf):
            mock = phase_randomised(hi, mrng) * flag
            inj = pca_clean(cube + mock, args.pca_modes, valid)
            acc[i] = pk(inj - cleaned, mock) / pk(mock)
        T_inj, T_inj_std = acc.mean(axis=0), acc.std(axis=0)
        print(f'\n  injection TF, {args.injection_tf} Gaussian mocks vs exact:')
        print('  bin       k      T exact   T injected      ratio')
        for i, k in enumerate(sig_k):
            print(f'  {i:^3d} {k:8.4f}  {T[i]:9.4f}  {T_inj[i]:6.4f}'
                  f'+-{T_inj_std[i]:.4f}  {T_inj[i] / T[i]:9.2f}')
        print('  A ratio far from 1 means Gaussian mocks do not stand in for '
              'this\n  signal; the loss depends on phase structure they do not '
              'carry.')

    dest = os.path.join(args.out, 'pca_benchmark.npz')
    out = dict(sig_k=sig_k, true_pk=true_pk, pk_pca=pk_pca, T=T,
               pk_pca_raw=pk_pca_raw, noise_pk=noise_pk,
               pca_modes=args.pca_modes, n_modes=args.n_modes)
    if T_inj is not None:
        out.update(T_inj=T_inj, T_inj_std=T_inj_std)
    np.savez_compressed(dest, **out)
    print(f'\nwrote {dest}')


if __name__ == '__main__':
    main()
