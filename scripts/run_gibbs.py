#!/usr/bin/env python
"""Batch Gibbs sampler -- the notebook's sampling loop, without the plots.

Mirrors ``notebooks/2_gibbs_sampling.ipynb`` up to and including the sampling
loop. Anything that changes the posterior lives in one place (``imgibbs``) and
is shared with the notebook, so the two cannot drift apart numerically.

Usage
-----
    python scripts/run_gibbs.py 6 --n-samples 500

The one required argument is the number of foreground (Legendre) modes.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import LinearOperator, lgmres
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imgibbs import (                                          # noqa: E402
    Us, construct_A, construct_b, construct_preconditioner,
    bin_it, kbins_from_crop, survey_grid, data_path, load, load_l2021_cube,
    signal_covariance_sampler as SCS,
    foreground_covariance_sampler as FCS,
)

# The crop and bin ceiling MUST match the ones the loaded S was built with in
# 1_generate_signal_cube.ipynb. The run asserts this against the metadata file
# rather than trusting it.
CROP = (slice(33, 103), slice(14, 59), slice(0, 250))
N_K_BINS = 5                       # ceiling; kbins_from_crop may return fewer

T_SYS = 16.0                       # K, Wang et al. (2021) Table 1, L band
DEL_T = 1000.0                     # s
# True channel width: 856-1712 MHz over 4096 channels = 0.208984 MHz, not the
# 0.2 MHz round number the paper quotes.
DEL_NU = (1712.0 - 856.0) / 4096 * 1e6      # Hz

S_FILE = 'S_starting_point_cropped.npy'
S_META = 'S_starting_point_cropped_meta.json'


def parse_args():
    p = argparse.ArgumentParser(
        description='Gibbs sampling on the MeerKLASS L2021 cube.')
    p.add_argument('n_modes', type=int,
                   help='number of foreground (Legendre) modes')
    p.add_argument('--n-samples', type=int, default=100,
                   help='number of Gibbs samples to draw (default: 100)')
    p.add_argument('--out', default='outputs',
                   help='output directory (default: outputs)')
    p.add_argument('--suffix', default=None,
                   help='trace filename suffix (default: _LP<n_modes>_)')
    p.add_argument('--tol', type=float, default=1e-6,
                   help='LGMRES relative tolerance (default: 1e-6)')
    p.add_argument('--seed', type=int, default=None,
                   help='numpy random seed, for reproducible chains')
    p.add_argument('--no-flagging', action='store_true',
                   help='do not downweight flagged (zero) voxels')
    p.add_argument('--save-S', action='store_true',
                   help='also write the full S cube each iteration. S is '
                        'piecewise-constant over the k-bins, so Pk_trace plus '
                        'the bin metadata reconstructs it exactly -- writing it '
                        'costs ~6 MB per sample for no extra information.')
    return p.parse_args()


def check_grid_matches_S(grid, kbin_meta):
    """Fail loudly if S was built on a different grid than this run uses.

    S is a per-voxel array indexed by k-bins derived from box_dims. If the two
    disagree the signal prior is silently attached to the wrong wavenumbers,
    which is the single easiest way to get plausible-looking nonsense out of
    this sampler.
    """
    meta_path = data_path(S_META)
    if not meta_path.exists():
        print(f'WARNING: {S_META} not found; skipping the grid consistency check')
        return
    meta = json.loads(meta_path.read_text())

    problems = []
    if tuple(meta['shape']) != tuple(grid.shape):
        problems.append(f"shape: S built on {tuple(meta['shape'])}, "
                        f"this run is {tuple(grid.shape)}")
    if not np.allclose(meta['box_dims'], grid.box_dims, rtol=1e-9):
        problems.append(f"box_dims: S built on {meta['box_dims']}, "
                        f"this run is {list(grid.box_dims)}")
    if meta['n_k_bins'] != kbin_meta['n_k_bins']:
        problems.append(f"n_k_bins: S built with {meta['n_k_bins']}, "
                        f"this run chose {kbin_meta['n_k_bins']}")
    if meta.get('kbin_modes_per_bin') != kbin_meta['modes_per_bin']:
        problems.append('modes per bin differ between S and this run')

    if problems:
        raise SystemExit('S does not match this grid:\n  '
                         + '\n  '.join(problems)
                         + f'\n\nRe-run 1_generate_signal_cube.ipynb with '
                           f'CROP = {CROP}, or change CROP here to match.')
    print('grid check   : S matches this run')


def main():
    args = parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)

    n_modes = args.n_modes
    flagging = not args.no_flagging
    suffix = args.suffix if args.suffix is not None else f'_LP{n_modes}_'
    sample_dir = os.path.join(args.out, 'samples')
    os.makedirs(sample_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Data and geometry
    # ------------------------------------------------------------------
    data_cube = load_l2021_cube()[CROP]
    shape = data_cube.shape
    grid = survey_grid(CROP, shape)
    box_dims = grid.box_dims
    print(grid.summary())

    if flagging:
        flag = np.where(data_cube == 0, 0, data_cube)
        flag = np.where(flag != 0, 1, flag)
        data_cube = data_cube * flag
        w = flag.flatten()
    else:
        w = np.ones(np.prod(shape))

    sig_k, idxs, kbin_meta = kbins_from_crop(data_cube, box_dims,
                                            max_bins=N_K_BINS)
    occupied_bins = np.unique(idxs)
    occupied_bins = occupied_bins[occupied_bins > 0]
    S = load(S_FILE)
    check_grid_matches_S(grid, kbin_meta)
    if S.size != np.prod(shape):
        raise SystemExit(f'{S_FILE} has {S.size} entries, this grid has '
                         f'{np.prod(shape)} voxels')

    # ------------------------------------------------------------------
    # Foreground model: orthonormalised Legendre polynomials in frequency
    # ------------------------------------------------------------------
    n_freq = shape[2]
    poly_basis = np.polynomial.legendre.legvander(
        np.linspace(-1, 1, n_freq), n_modes - 1).T          # (n_modes, n_freq)
    evecs = np.linalg.qr(poly_basis.T)[0].T                 # orthonormal rows

    print(f'evecs        : {evecs.shape}  orthonormality '
          f'{np.abs(evecs @ evecs.T - np.eye(n_modes)).max():.2e}')

    d_2d = data_cube.reshape(-1, n_freq)
    f_true = (d_2d @ evecs.T).reshape(shape[0], shape[1], n_modes)
    s_true = Us(data_cube, True)

    # ------------------------------------------------------------------
    # Noise
    # ------------------------------------------------------------------
    N = (T_SYS**2) / (DEL_NU * DEL_T)
    N_inv = 1 / N
    N_inv_scalar = 1.0 / N
    Nw_inv = (1 / N) * w
    print(f'noise        : T_sys {T_SYS:g} K, channel {DEL_NU/1e6:.6f} MHz, '
          f'N = {N:.4e} K^2')

    # Shapes for packing/unpacking x = [Re(s), Im(s), f]
    rfft_len = s_true.size
    rfft_shape = s_true.shape
    f_len = f_true.size
    f_shape = f_true.shape
    total_len = 2 * rfft_len + f_len

    # ------------------------------------------------------------------
    # Starting points
    # ------------------------------------------------------------------
    s_mean = np.zeros(rfft_shape)                # zero-mean signal prior
    f_mean = (d_2d @ evecs.T).reshape(shape[0], shape[1], n_modes)

    # Perturb before the inverse-Wishart draw: at f == f_mean exactly the
    # scatter matrix is singular, so FCS needs the spread.
    f_init = f_mean * np.random.normal(1.0, 0.05, f_mean.shape)
    F = np.diag(FCS(f_init.reshape(-1, n_modes)))

    x = np.concatenate([s_mean.real.flatten(), s_mean.imag.flatten(),
                        f_mean.flatten()])

    def A_flat(vec):
        return construct_A(vec, S, Nw_inv, F, w, evecs, rfft_len, rfft_shape,
                           f_len, f_shape, shape).flatten()

    L = LinearOperator(matvec=A_flat, rmatvec=A_flat, shape=(len(x), len(x)))

    precond_apply = construct_preconditioner(S, N_inv_scalar, F, evecs,
                                             rfft_len, rfft_shape, f_len,
                                             f_shape, shape)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    with open(os.path.join(args.out, f'run{suffix}config.json'), 'w') as fh:
        json.dump({'crop': [[sl.start, sl.stop] for sl in CROP],
                   'shape': list(shape), 'box_dims': list(box_dims),
                   'n_modes': n_modes, 'n_samples': args.n_samples,
                   'T_sys': T_SYS, 'del_nu_Hz': DEL_NU, 'del_t_s': DEL_T,
                   'tol': args.tol, 'seed': args.seed, 'flagging': flagging,
                   'sig_k': sig_k.tolist(), **kbin_meta}, fh, indent=2,
                  default=str)

    print(f'\nsampling     : {args.n_samples} samples, {n_modes} FG modes, '
          f'traces -> {sample_dir}/*{suffix}.npy')
    start = time.time()

    for rr in tqdm(range(args.n_samples)):
        x0 = x      # warm-start the solver from the previous sample
        np.save(os.path.join(sample_dir, f'x_sample_{rr}{suffix}.npy'), x)

        # Omega draws: these turn the solve from a MAP estimate into a
        # posterior sample. Zero them for the MAP/Wiener-filter solution.
        ws = np.fft.rfftn(np.random.normal(size=shape), norm='ortho').flatten()
        wf = np.random.normal(size=f_shape)
        wd = np.random.normal(size=shape).flatten()

        b = construct_b(S, N_inv, F, w, s_mean, f_mean, evecs, data_cube,
                        ws, wf, wd, shape)

        M_inv = LinearOperator((total_len, total_len), matvec=precond_apply)
        x, exit_code = lgmres(L, b.flatten(), x0=x0, rtol=args.tol, atol=0,
                              M=M_inv)
        if exit_code != 0:
            print(f'  [{rr}] lgmres did not converge (exit {exit_code})')

        # --- P(k) step: inverse-gamma draw per k-bin ---
        s = (x[0:rfft_len].reshape(rfft_shape)
             + x[rfft_len:2 * rfft_len].reshape(rfft_shape) * 1j)
        s = Us(s, False)
        s = np.fft.fftn(s - np.mean(s), norm='ortho')

        binned_s, k_bins = bin_it(s, sig_k, idxs)
        _, PkSample = SCS(np.concatenate(binned_s), np.concatenate(k_bins))

        S = np.zeros(len(idxs))
        for gg, bin_idx in enumerate(occupied_bins):
            S[idxs == bin_idx] = PkSample[gg]
        # idxs == 0 is DC plus the kz=0 plane plus sub-footprint modes. A flat
        # 1e30 prior there would let the signal absorb foreground power in
        # exactly the degenerate modes, so suppress them and leave only the
        # true DC mode free to carry the overall mean.
        S[idxs == 0] = 1e-12 * np.median(PkSample)
        S[kbin_meta['dc_index']] = 1e30

        np.save(os.path.join(sample_dir, f'Pk_trace{rr}{suffix}.npy'), PkSample)
        if args.save_S:
            np.save(os.path.join(sample_dir, f'S_trace{rr}{suffix}.npy'), S)

        # --- F step: inverse-Wishart draw ---
        f_ms = x[2 * rfft_len:2 * rfft_len + f_len].reshape(f_shape).real
        f_centered = (f_ms - f_mean).reshape((shape[0] * shape[1], n_modes))
        F = np.diag(FCS(f_centered))
        np.save(os.path.join(sample_dir, f'F_trace{rr}{suffix}.npy'), F)

        precond_apply = construct_preconditioner(S, N_inv_scalar, F, evecs,
                                                 rfft_len, rfft_shape, f_len,
                                                 f_shape, shape)
        gc.collect()

    elapsed = time.time() - start
    print(f'\n{args.n_samples} samples with {n_modes} FG modes in '
          f'{elapsed:.1f} s ({elapsed / args.n_samples:.2f} s/sample)')


if __name__ == '__main__':
    main()
