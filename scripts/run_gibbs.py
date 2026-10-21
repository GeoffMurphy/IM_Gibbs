#!/usr/bin/env python
"""Batch Gibbs sampler -- the notebook's sampling loop, without the plots.

Mirrors ``notebooks/2_gibbs_sampling.ipynb`` up to and including the sampling
loop. Anything that changes the posterior lives in one place (``imgibbs``) and
is shared with the notebook, so the two cannot drift apart numerically.

Usage
-----
    python scripts/run_gibbs.py 6 --n-samples 500

The one required argument is the number of foreground (Legendre) modes.

Add ``--groundspill`` to sample a standing-wave systematic jointly with the
signal and the foreground::

    python scripts/run_gibbs.py 6 --n-samples 500 --groundspill --gs-period 17.5

That writes a ``g_trace`` alongside the others. To see what it buys on data
with a known answer, run ``scripts/systematics_injection.py`` instead.

Add ``--beam-fwhm`` to put a chromatic Gaussian primary beam on the sky
(signal and foreground, not the systematic)::

    python scripts/run_gibbs.py 6 --beam-fwhm 1.5 --beam-nu-ref 1000

or any tabulated ``B(|k_perp|, nu)`` (format: ``imgibbs.load_beam_table``)::

    python scripts/run_gibbs.py 6 --beam-table my_beam_table.npz

``S`` is then the power spectrum of the *unconvolved* sky. With a beam the
``S`` step is collapsed: ``S`` is drawn from the Fourier modes the beam
leaves visible (``--beam-threshold``) and the rest are marginalised, rather
than letting ``S`` re-estimate itself from modes the data never saw.
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
    groundspill_basis, ripple_wavenumber, gaussian_beam, beam_fwhm,
    load_beam_table, MaskedForegroundBlock,
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
    p.add_argument('--data', default=None,
                   help='sample this .npy cube instead of the L2021 data. It '
                        'must already be cropped to CROP, in K, with flagged '
                        'voxels at exactly 0 -- e.g. a mock from '
                        'scripts/beam_injection.py.')
    p.add_argument('--save-x-every', type=int, default=1,
                   help='write the full x sample every N iterations '
                        '(default 1). Each is ~13 MB; Pk/F/g traces are '
                        'always written every iteration.')
    p.add_argument('--S-start-scale', type=float, default=1.0,
                   help='multiply the starting S by this (default 1). The '
                        'shipped S_starting_point is the power of the same '
                        'simulated H I the injection mocks use, so a chain '
                        'started from it starts at the answer; start from '
                        '0.1 and 10 to see whether the data, not the start, '
                        'set the result.')
    p.add_argument('--fix-S', action='store_true',
                   help='DIAGNOSTIC, not an analysis. Hold S at its starting '
                        'value (the true simulated H I power, times '
                        '--S-start-scale) instead of sampling it, to separate '
                        'what the S step does from what the foreground basis '
                        'and the beam do. Pk_trace then records the binned '
                        'power of the sampled signal field, the quantity the '
                        'S draw would have used. Same as '
                        'systematics_injection.py --fix-S.')
    p.add_argument('--f-prior-mean', choices=('data', 'zero'),
                   default='data',
                   help="centre of the foreground prior (default 'data': "
                        "the data's own projection onto the foreground "
                        "basis). F is sampled from the scatter of f about "
                        "this centre; with 'data' that scatter is tiny, F "
                        "collapses and the prior pins f to the data -- a "
                        "projection rather than a marginalisation. See "
                        "docs/STATUS.md, finding 4 of 'The beam in the "
                        "MODEL'.")
    p.add_argument('--no-flagging', action='store_true',
                   help='do not downweight flagged (zero) voxels')
    g = p.add_argument_group(
        'ground spill',
        'Add a systematic block for a standing-wave ripple. The SMOOTH part '
        'of ground spill is not modelled and does not need to be -- the '
        'foreground block absorbs it completely (see imgibbs/systematics.py).')
    g.add_argument('--groundspill', action='store_true',
                   help='sample a ground-spill systematic jointly with s and f')
    g.add_argument('--gs-period', type=float, default=17.5,
                   help='standing-wave period in MHz (default: 17.5). This is '
                        'the one parameter that is NOT sampled -- it enters '
                        'nonlinearly. Check the reported k_par lands where you '
                        'expect before trusting a run.')
    g.add_argument('--gs-harmonics', type=int, default=1,
                   help='number of ripple harmonics (default: 1)')
    g.add_argument('--gs-order', type=int, default=1,
                   help='polynomial order across the scan direction '
                        '(default: 1, i.e. constant + gradient)')
    g.add_argument('--gs-scan-axis', type=int, default=0, choices=(0, 1),
                   help='map axis the telescope scanned along (default: 0, RA)')
    g.add_argument('--gs-prior', type=float, default=1e-2,
                   help='prior std on each ground-spill amplitude, K '
                        '(default: 1e-2). This is a PRIOR, not a sampled '
                        'covariance: g is a handful of numbers, so unlike F it '
                        'cannot be estimated from a single draw. Set it from '
                        'instrument characterisation, generously -- it is a '
                        'nuisance parameter being marginalised over.')
    bm = p.add_argument_group(
        'beam',
        'Convolve the sky (signal and foreground) with a chromatic Gaussian '
        'primary beam, FWHM(nu) = fwhm * (nu_ref/nu)^index. Off unless '
        '--beam-fwhm is given. The systematic block stays outside the beam.')
    bm.add_argument('--beam-table', default=None,
                    help='tabulated beam, imgibbs.load_beam_table format '
                         '(k, freq_MHz, transfer). Takes precedence over '
                         '--beam-fwhm.')
    bm.add_argument('--beam-threshold', type=float, default=1e-3,
                    help='band-averaged |B|^2 below which a Fourier mode is '
                         'treated as invisible to the data and left out of '
                         'the S draw (default 1e-3). The run prints how much '
                         'information the excluded modes could still carry; '
                         'keep it well below 1.')
    bm.add_argument('--fg-dense-modes', type=int, default=3,
                    help='foreground modes preconditioned exactly, with the '
                         'mask, under the beam (default 3; 0 disables). '
                         'Memory ~2 x (Npix * n)^2 doubles: 1.4 GB at 3.')
    bm.add_argument('--fg-beam-groups', type=int, default=5,
                    help='channel groups for the beam in that block '
                         '(default 5)')
    bm.add_argument('--fg-refresh', type=int, default=20,
                    help='rebuild that block with the current F at least '
                         'every N samples (default 20), and sooner whenever a '
                         'solve takes over twice the matvecs of the first '
                         'solve after the last rebuild -- F moves fast in '
                         'burn-in, slowly after')
    bm.add_argument('--no-collapse', action='store_true',
                    help='draw S from every mode, as without a beam. For '
                         'comparison only -- see docs/STATUS.md.')
    bm.add_argument('--beam-fwhm', type=float, default=None,
                    help='beam FWHM in degrees at --beam-nu-ref. Take it from '
                         'a measured beam, not 1.22 lambda/D.')
    bm.add_argument('--beam-nu-ref', type=float, default=None,
                    help='reference frequency for --beam-fwhm, MHz '
                         '(default: band centre)')
    bm.add_argument('--beam-index', type=float, default=1.0,
                    help='chromatic index (default: 1, diffraction scaling). '
                         '0 gives an achromatic beam -- smoothing without '
                         'mode mixing, useful as a control.')
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
    if args.data is None:
        data_cube = load_l2021_cube()[CROP]
    else:
        data_cube = np.load(args.data)
        expected = load_l2021_cube()[CROP].shape
        if data_cube.shape != expected:
            raise SystemExit(f'{args.data} is {data_cube.shape}, CROP gives '
                             f'{expected}')
        print(f'data         : {args.data}')
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
    S = load(S_FILE) * args.S_start_scale
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
    # Systematics: ground spill
    # ------------------------------------------------------------------
    if args.groundspill:
        sys_basis = groundspill_basis(
            grid.freqs, shape, period=args.gs_period,
            n_harmonics=args.gs_harmonics, order=args.gs_order,
            scan_axis=args.gs_scan_axis)
        G = np.full(sys_basis.n_params, args.gs_prior ** 2)
        g_mean = np.zeros(sys_basis.g_shape)
        bandwidth = grid.freqs[-1] - grid.freqs[0]
        # A ripple is a single k_parallel mode: all of its power goes into one
        # bin rather than spreading. Print where, so a run that is about to
        # contaminate the lowest signal bin says so before it starts.
        k_ripple = [ripple_wavenumber(args.gs_period / m, bandwidth,
                                      box_dims[2])
                    for m in range(1, args.gs_harmonics + 1)]
        print(f'groundspill  : {sys_basis.n_params} params '
              f'({sys_basis.n_s} spatial x {sys_basis.n_t} spectral), '
              f'prior std {args.gs_prior:g} K')
        print(f'               period {args.gs_period:g} MHz -> k_par '
              + ', '.join(f'{k:.4f}' for k in k_ripple) + ' Mpc^-1')
        for k in k_ripple:
            bin_i = int(np.argmin(np.abs(sig_k - k)))
            print(f'               k = {k:.4f} is nearest bin {bin_i} '
                  f'(k = {sig_k[bin_i]:.4f})')
    else:
        sys_basis = G = g_mean = None

    # ------------------------------------------------------------------
    # Beam
    # ------------------------------------------------------------------
    nu_ref = None
    if args.beam_table is not None:
        beam = load_beam_table(args.beam_table, grid.freqs, shape)
        with np.load(args.beam_table) as tab:
            fw = np.interp(grid.freqs[[0, -1]], tab['freq_MHz'],
                           tab['fwhm_deg'])
            src = str(tab['source'])
        print(f'beam         : tabulated ({src}), FWHM {fw[0]:.3f} deg at '
              f'{grid.freqs[0]:.1f} MHz -> {fw[1]:.3f} deg at '
              f'{grid.freqs[-1]:.1f} MHz')
    elif args.beam_fwhm is not None:
        nu_ref = (args.beam_nu_ref if args.beam_nu_ref is not None
                  else float(grid.freqs.mean()))
        beam = gaussian_beam(grid.freqs, shape, args.beam_fwhm, nu_ref,
                             index=args.beam_index)
        fw = beam_fwhm(grid.freqs[[0, -1]], args.beam_fwhm, nu_ref,
                       args.beam_index)
        print(f'beam         : Gaussian, FWHM {fw[0]:.3f} deg at '
              f'{grid.freqs[0]:.1f} MHz -> {fw[1]:.3f} deg at '
              f'{grid.freqs[-1]:.1f} MHz (index {args.beam_index:g})')
    else:
        beam = None

    # Collapsed S step. Modes the beam has removed carry (almost) no
    # information about S; drawing S from them as well lets S re-estimate
    # itself from its own prior draws -- a near random walk. Instead draw S
    # from the visible modes only (marginalising the rest), then apply it to
    # every mode in the bin. Exact as the excluded modes' information -> 0.
    if beam is not None and not args.no_collapse:
        visible = beam.observable_modes(args.beam_threshold)
        idxs_S = np.where(visible, idxs, 0)
        kept = [int(np.sum(idxs_S == b)) for b in occupied_bins]
        total = [int(np.sum(idxs == b)) for b in occupied_bins]
        if min(kept) < 3:
            raise SystemExit(f'--beam-threshold {args.beam_threshold:g} '
                             f'leaves bins with {kept} visible modes')
        print('S step       : drawn from beam-visible modes only, '
              + ', '.join(f'{k}/{t}' for k, t in zip(kept, total)))
    else:
        idxs_S = idxs
        kept = None

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
    g_len = 0 if sys_basis is None else sys_basis.n_params
    total_len = 2 * rfft_len + f_len + g_len

    # ------------------------------------------------------------------
    # Starting points
    # ------------------------------------------------------------------
    s_mean = np.zeros(rfft_shape)                # zero-mean signal prior
    # The data's own projection onto the foreground basis: the starting
    # point for f, and (--f-prior-mean data) the centre of its prior.
    f_proj = (d_2d @ evecs.T).reshape(shape[0], shape[1], n_modes)
    f_mean = f_proj if args.f_prior_mean == 'data' else np.zeros_like(f_proj)

    # Perturb before the inverse-Wishart draw: at f == f_proj exactly the
    # scatter matrix is singular, so FCS needs the spread.
    f_init = f_proj * np.random.normal(1.0, 0.05, f_proj.shape)
    F = np.diag(FCS(f_init.reshape(-1, n_modes)))

    x = np.concatenate([s_mean.real.flatten(), s_mean.imag.flatten(),
                        f_proj.flatten()]
                       + ([g_mean.flatten()] if sys_basis is not None else []))

    A_calls = [0]
    matvecs = []

    def A_flat(vec):
        A_calls[0] += 1
        return construct_A(vec, S, Nw_inv, F, w, evecs, rfft_len, rfft_shape,
                           f_len, f_shape, shape,
                           sys_basis=sys_basis, G=G, beam=beam).flatten()

    L = LinearOperator(matvec=A_flat, rmatvec=A_flat, shape=(len(x), len(x)))

    # With a beam and a mask the default foreground preconditioner does not
    # converge (docs/STATUS.md); treat the stiff low-order modes exactly.
    fg_block = None
    if beam is not None and args.fg_dense_modes > 0:
        t0 = time.time()
        fg_block = MaskedForegroundBlock(beam, w, evecs, N_inv_scalar,
                                         n_dense=args.fg_dense_modes,
                                         n_groups=args.fg_beam_groups)
        fg_block.factor(F)
        print(f'fg precond   : exact-mask block, {fg_block.n_dense} modes, '
              f'{args.fg_beam_groups} beam groups, built in '
              f'{time.time() - t0:.0f} s, refreshed every {args.fg_refresh}')

    precond_apply = construct_preconditioner(S, N_inv_scalar, F, evecs,
                                             rfft_len, rfft_shape, f_len,
                                             f_shape, shape,
                                             sys_basis=sys_basis, G=G,
                                             beam=beam, fg_block=fg_block)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    with open(os.path.join(args.out, f'run{suffix}config.json'), 'w') as fh:
        json.dump({'crop': [[sl.start, sl.stop] for sl in CROP],
                   'shape': list(shape), 'box_dims': list(box_dims),
                   'n_modes': n_modes, 'n_samples': args.n_samples,
                   'T_sys': T_SYS, 'del_nu_Hz': DEL_NU, 'del_t_s': DEL_T,
                   'tol': args.tol, 'seed': args.seed, 'flagging': flagging,
                   'S_start_scale': args.S_start_scale,
                   'fix_S': bool(args.fix_S),
                   'f_prior_mean': args.f_prior_mean,
                   'groundspill': bool(args.groundspill),
                   'gs_period_MHz': args.gs_period if args.groundspill else None,
                   'gs_harmonics': args.gs_harmonics if args.groundspill else None,
                   'gs_order': args.gs_order if args.groundspill else None,
                   'gs_scan_axis': args.gs_scan_axis if args.groundspill else None,
                   'gs_prior_K': args.gs_prior if args.groundspill else None,
                   'data': args.data,
                   'beam_table': args.beam_table,
                   'beam_collapse': (beam is not None
                                     and not args.no_collapse),
                   'beam_threshold': (args.beam_threshold if beam is not None
                                      else None),
                   'beam_visible_modes_per_bin': kept,
                   'fg_dense_modes': (fg_block.n_dense if fg_block is not None
                                      else 0),
                   'fg_beam_groups': args.fg_beam_groups,
                   'fg_refresh': args.fg_refresh,
                   'beam_fwhm_deg': args.beam_fwhm,
                   'beam_nu_ref_MHz': nu_ref,
                   'beam_index': (args.beam_index if beam is not None
                                  else None),
                   'sig_k': sig_k.tolist(), **kbin_meta}, fh, indent=2,
                  default=str)

    print(f'\nsampling     : {args.n_samples} samples, {n_modes} FG modes, '
          f'traces -> {sample_dir}/*{suffix}.npy')
    start = time.time()
    warned_q = False
    fg_baseline, since_refresh, fg_refreshes = None, 0, []

    for rr in tqdm(range(args.n_samples)):
        x0 = x      # warm-start the solver from the previous sample
        if rr % args.save_x_every == 0:
            np.save(os.path.join(sample_dir, f'x_sample_{rr}{suffix}.npy'), x)

        # Omega draws: these turn the solve from a MAP estimate into a
        # posterior sample. Zero them for the MAP/Wiener-filter solution.
        ws = np.fft.rfftn(np.random.normal(size=shape), norm='ortho').flatten()
        wf = np.random.normal(size=f_shape)
        wd = np.random.normal(size=shape).flatten()
        wg = (None if sys_basis is None
              else np.random.normal(size=sys_basis.g_shape))

        b = construct_b(S, N_inv, F, w, s_mean, f_mean, evecs, data_cube,
                        ws, wf, wd, shape, sys_basis=sys_basis, G=G,
                        wg=wg, g_mean=g_mean, beam=beam)

        M_inv = LinearOperator((total_len, total_len), matvec=precond_apply)
        n_mv = A_calls[0]
        x, exit_code = lgmres(L, b.flatten(), x0=x0, rtol=args.tol, atol=0,
                              M=M_inv)
        matvecs.append(A_calls[0] - n_mv)
        if exit_code != 0:
            print(f'  [{rr}] lgmres did not converge (exit {exit_code})')

        # --- P(k) step: inverse-gamma draw per k-bin ---
        s = (x[0:rfft_len].reshape(rfft_shape)
             + x[rfft_len:2 * rfft_len].reshape(rfft_shape) * 1j)
        s = Us(s, False)
        s = np.fft.fftn(s - np.mean(s), norm='ortho')

        binned_s, k_bins = bin_it(s, sig_k, idxs_S)
        if args.fix_S:
            # S is NOT updated; record the power the S draw would have used.
            PkSample = np.array([np.mean(np.abs(b) ** 2) for b in binned_s])
        else:
            _, PkSample = SCS(np.concatenate(binned_s),
                              np.concatenate(k_bins))
            S = np.zeros(len(idxs))
            for gg, bin_idx in enumerate(occupied_bins):
                S[idxs == bin_idx] = PkSample[gg]
            # idxs == 0 is DC plus the kz=0 plane plus sub-footprint modes. A
            # flat 1e30 prior there would let the signal absorb foreground
            # power in exactly the degenerate modes, so suppress them and
            # leave only the true DC mode free to carry the overall mean.
            S[idxs == 0] = 1e-12 * np.median(PkSample)
            S[kbin_meta['dc_index']] = 1e30

        if beam is not None and not args.no_collapse and not args.fix_S:
            # The excluded modes have |B|^2 < threshold, so each carries at
            # most this data/prior precision ratio. If S has grown enough to
            # make that non-negligible, the collapse is no longer exact.
            q_excl = (w.mean() * N_inv * args.beam_threshold
                      * float(np.max(PkSample)))
            if q_excl > 0.05 and not warned_q:
                print(f'  [{rr}] WARNING: excluded modes may now carry up to '
                      f'{q_excl:.2f} of their prior precision; lower '
                      f'--beam-threshold')
                warned_q = True

        np.save(os.path.join(sample_dir, f'Pk_trace{rr}{suffix}.npy'), PkSample)
        if args.save_S:
            np.save(os.path.join(sample_dir, f'S_trace{rr}{suffix}.npy'), S)

        # --- F step: inverse-Wishart draw ---
        # --- g: no covariance step. G is a prior, not a sampled quantity --
        # a single vector of n_params numbers does not identify its own
        # covariance the way Npix foreground amplitude vectors identify F.
        if sys_basis is not None:
            g_off = 2 * rfft_len + f_len
            np.save(os.path.join(sample_dir, f'g_trace{rr}{suffix}.npy'),
                    x[g_off:g_off + g_len].reshape(sys_basis.g_shape))

        f_ms = x[2 * rfft_len:2 * rfft_len + f_len].reshape(f_shape).real
        f_centered = (f_ms - f_mean).reshape((shape[0] * shape[1], n_modes))
        F = np.diag(FCS(f_centered))
        np.save(os.path.join(sample_dir, f'F_trace{rr}{suffix}.npy'), F)

        if fg_block is not None:
            if fg_baseline is None:
                fg_baseline = matvecs[-1]
            since_refresh += 1
            if (since_refresh >= args.fg_refresh
                    or matvecs[-1] > 2 * fg_baseline):
                fg_block.factor(F)
                fg_refreshes.append(rr)
                fg_baseline, since_refresh = None, 0
        precond_apply = construct_preconditioner(S, N_inv_scalar, F, evecs,
                                                 rfft_len, rfft_shape, f_len,
                                                 f_shape, shape,
                                                 sys_basis=sys_basis, G=G,
                                                 beam=beam, fg_block=fg_block)
        gc.collect()

    np.save(os.path.join(args.out, f'matvecs{suffix}.npy'), np.array(matvecs))
    if fg_block is not None:
        np.save(os.path.join(args.out, f'fg_refreshes{suffix}.npy'),
                np.array(fg_refreshes))
    elapsed = time.time() - start
    print(f'\n{args.n_samples} samples with {n_modes} FG modes in '
          f'{elapsed:.1f} s ({elapsed / args.n_samples:.2f} s/sample, '
          f'median {np.median(matvecs):.0f} matvecs/solve)')


if __name__ == '__main__':
    main()
