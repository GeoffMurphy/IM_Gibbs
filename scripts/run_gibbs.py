#!/usr/bin/env python
"""Batch Gibbs sampler -- the end-to-end notebook's chain, without the plots.

Builds a mock observation (``imgibbs.mock``) or reads a cube, then runs
:func:`imgibbs.sampler.run_chain`, the same function
``notebooks/1_end_to_end.ipynb`` calls, so the two cannot drift apart
numerically.

Usage
-----
    python scripts/run_gibbs.py 6 --n-samples 500

The one required argument is the number of foreground (Legendre) modes. By
default the data are a mock with HI, FastBox foregrounds and noise on a
drift-scan footprint; it is written to ``<out>/mock.npz`` so the run can be
scored against the truth afterwards. ``--mock-*`` options add a beam or a
systematic to it.

The run in the end-to-end notebook -- a beam and all three systematics in
the mock, all modelled::

    python scripts/run_gibbs.py 6 --n-samples 2000 --out outputs/end_to_end \\
        --mock-beam-fwhm 1.5 --mock-systematic all \\
        --beam-fwhm 1.5 --groundspill --onef --leakage

``--groundspill``, ``--onef`` and ``--leakage`` each add a block of
systematic templates, sampled jointly with the signal and the foreground as
one ``g``; that writes a ``g_trace`` alongside the others.
``--beam-fwhm`` puts a chromatic Gaussian primary beam on the sky (signal and
foreground, not the systematic); ``--beam-table`` takes any tabulated
``B(|k_perp|, nu)`` (format: ``imgibbs.load_beam_table``).

``S`` is then the power spectrum of the *unconvolved* sky. With a beam the
``S`` step is collapsed: ``S`` is drawn from the Fourier modes the beam
leaves visible (``--beam-threshold``) and the rest are marginalised, rather
than letting ``S`` re-estimate itself from modes the data never saw.

``--data cube.npy`` samples your own cube instead: K, flagged voxels exactly
zero, frequency on the last axis starting at L-band channel 550 (see
``imgibbs.grid``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imgibbs import (                                          # noqa: E402
    CompositeBasis, kbins_from_crop, groundspill_basis, leakage_basis,
    onef_basis, ripple_wavenumber, gaussian_beam, beam_fwhm, load_beam_table,
    beam_project,
)
from imgibbs.mock import (                                     # noqa: E402
    DEL_NU, DEL_T, MOCK_SHAPE, SYSTEMATICS, T_SYS, _systematic_names,
    binned_power, expected_hi_power, load_mock, make_mock, mock_grid,
)
from imgibbs.sampler import (                                  # noqa: E402
    legendre_basis, resolve_gs_solve, run_chain, signal_prior,
)

N_K_BINS = 5                       # ceiling; kbins_from_crop may return fewer


def parse_args():
    p = argparse.ArgumentParser(
        description='Gibbs sampling on a mock observation (or your own cube).')
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
                   help='numpy random seed for the chain, for reproducible '
                        'chains (the mock has its own, --mock-seed)')
    p.add_argument('--data', default=None,
                   help='sample this cube instead of building a mock: a '
                        '.npy in K with flagged voxels at exactly 0, or a '
                        '.npz written by imgibbs.mock.Mock.save')
    p.add_argument('--burn', type=int, default=None,
                   help='samples left out of the posterior-mean maps '
                        '(signal_mean, fg_mean; default: a quarter of '
                        '--n-samples). The traces are always complete.')
    p.add_argument('--save-x-every', type=int, default=1,
                   help='write the full x sample every N iterations '
                        '(default 1; 0 never). Each is ~6 MB; Pk/F/g traces '
                        'are always written every iteration.')
    p.add_argument('--S-start', choices=('expected', 'truth'),
                   default='expected',
                   help="starting S. 'expected' (default): the mean HI power "
                        "of --S-start-real independent realisations, i.e. "
                        "the fiducial cosmology, not the realisation in the "
                        "data. 'truth': the power of the mock's own HI -- the "
                        "chain starts at the answer, which is how every run "
                        "before 2026-11 started (docs/STATUS.md). Diagnostic "
                        "only, and only with a mock.")
    p.add_argument('--S-start-real', type=int, default=8,
                   help="realisations averaged for --S-start expected "
                        "(default 8)")
    p.add_argument('--S-start-scale', type=float, default=1.0,
                   help='multiply the starting S by this (default 1). Start '
                        'from 0.1 and 10 to see whether the data, not the '
                        'start, set the result.')
    p.add_argument('--fix-S', action='store_true',
                   help='DIAGNOSTIC, not an analysis. Hold S at its starting '
                        'value instead of sampling it, to separate what the '
                        'S step does from what the foreground basis and the '
                        'beam do. Pk_trace then records the binned power of '
                        'the sampled signal field, the quantity the S draw '
                        'would have used. Same as systematics_injection.py '
                        '--fix-S.')
    p.add_argument('--f-prior-mean', choices=('auto', 'data', 'zero'),
                   default='auto',
                   help="centre of the foreground prior. 'data' is the "
                        "data's own projection onto the foreground basis; "
                        "'zero' is zero. 'auto' (default) is 'data' without "
                        "a beam, the published behaviour, and 'zero' with "
                        "one, because under a beam the data's projection is "
                        "already beamed and B Uf f_mean would beam it twice. "
                        "Without a beam neither choice is right at the "
                        "lowest k: F is sampled about this centre, so 'data' "
                        "collapses F and projects the H I out, and 'zero' "
                        "makes F as small as the H I in the high-order modes "
                        "and leaks foreground into s. See docs/STATUS.md, "
                        "finding 4 of 'The beam in the MODEL'.")
    p.add_argument('--no-flagging', action='store_true',
                   help='do not downweight flagged (zero) voxels')
    mk = p.add_argument_group(
        'mock',
        'The mock observation sampled when --data is not given: FastBox HI '
        'and foregrounds, radiometer noise, a drift-scan footprint '
        '(imgibbs.mock). The --mock-* options describe the DATA; the beam '
        'and ground-spill options below describe the MODEL. Give both to '
        'model what was injected.')
    mk.add_argument('--mock-seed', type=int, default=0,
                    help='master seed for the mock (default 0)')
    mk.add_argument('--mock-shape', type=int, nargs=3, default=MOCK_SHAPE,
                    metavar=('NX', 'NY', 'NZ'),
                    help=f'mock grid (default {MOCK_SHAPE[0]} {MOCK_SHAPE[1]} '
                         f'{MOCK_SHAPE[2]})')
    mk.add_argument('--mock-beam-fwhm', type=float, default=None,
                    help='convolve the mock sky with a Gaussian beam of this '
                         'FWHM in degrees at the band centre (default: none)')
    mk.add_argument('--mock-systematic', default=None,
                    type=_mock_systematic,
                    help='inject this systematic: ' + ', '.join(SYSTEMATICS)
                         + ', a comma-separated list, or all (default: none)')
    mk.add_argument('--mock-sys-rms', type=float, default=None,
                    help='rms of each injected systematic, K (default '
                         '1e-3, about the HI rms)')
    g = p.add_argument_group(
        'ground spill',
        'Add a systematic block for a standing-wave ripple. Without a beam '
        'the SMOOTH part of ground spill does not need modelling -- the '
        'foreground block absorbs it completely (see imgibbs/systematics.py). '
        'With a beam it does: spill is not beamed but the foreground model '
        'is, so B Uf f cannot absorb it (see --gs-smooth and --gs-solve).')
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
    g.add_argument('--gs-deflate', action='store_true',
                   help='project the foreground span out of the ripple '
                        'templates. The part of a template inside it is '
                        'degenerate with f and only confuses the split; '
                        'deflating took the beamless injection test from 15x '
                        'to the control in the lowest bin (docs/STATUS.md). '
                        'With a beam it made no difference (finding 9 of '
                        "'The beam in the MODEL'), and it cannot be combined "
                        'with --gs-smooth.')
    g.add_argument('--gs-smooth', choices=('auto', 'on', 'off'),
                   default='auto',
                   help='add an unbeamed smooth template (the ripple envelope) '
                        'per spatial term, for the smooth spill. auto (the '
                        'default): on when a beam is modelled. The spill '
                        'enters after the beam, so B Uf f cannot fit it; '
                        'leaving it out put 22x the H I power into the lowest '
                        'bin of the beamed injection test (docs/STATUS.md, '
                        "finding 9 of 'The beam in the MODEL'). Without a "
                        'beam it is degenerate with the foreground block.')
    g.add_argument('--gs-solve', choices=('auto', 'joint', 'marginal'),
                   default='auto',
                   help='joint: sample g in the same solve as s and f. '
                        'marginal: integrate g out of that solve (it becomes '
                        'correlated noise, N + Ug G Ug^T) and draw it '
                        'afterwards from g | s, f, d -- the same posterior. '
                        'auto (the default): marginal when a beam is '
                        'modelled, where the joint solve barely moves g '
                        "(docs/STATUS.md, finding 9 of 'The beam in the "
                        "MODEL').")
    g.add_argument('--gs-smooth-prior', type=float, default=10.0,
                   help='prior std on each smooth-spill amplitude, K '
                        '(default: 10). Spill is ~0.5 K, far above the '
                        'ripple, so it needs its own, much wider prior.')
    g.add_argument('--gs-prior', type=float, default=1e-2,
                   help='prior std on each ground-spill amplitude, K '
                        '(default: 1e-2). This is a PRIOR, not a sampled '
                        'covariance: g is a handful of numbers, so unlike F it '
                        'cannot be estimated from a single draw. Set it from '
                        'instrument characterisation, generously -- it is a '
                        'nuisance parameter being marginalised over.')
    o = p.add_argument_group(
        '1/f and polarisation leakage',
        'More template blocks, sampled with the ground spill as one g. Each '
        'amplitude multiplies a template of unit rms, so a prior std is the '
        'rms it allows in the cube.')
    o.add_argument('--onef', action='store_true',
                   help='sample a 1/f block: the leading KL modes of a 1/f '
                        'covariance, with the foreground span deflated out '
                        '(imgibbs.onef_basis) -- the part f cannot absorb')
    o.add_argument('--onef-prior', type=float, default=1e-2,
                   help='prior std on each 1/f amplitude, K (default 1e-2)')
    o.add_argument('--leakage', action='store_true',
                   help='sample a polarisation-leakage block: {1, x, y} x '
                        '{cos, sin} of the Faraday angle (imgibbs.leakage_basis)')
    o.add_argument('--leak-rm', type=float, default=1000.0,
                   help='rotation measure, rad m^-2 (default 1000). Like the '
                        'ripple period, not sampled.')
    o.add_argument('--leak-prior', type=float, default=1e-2,
                   help='prior std on each leakage amplitude, K (default 1e-2)')
    bm = p.add_argument_group(
        'beam',
        'Convolve the sky (signal and foreground) with a chromatic Gaussian '
        'primary beam, FWHM(nu) = fwhm * (nu_ref/nu)^index. Off unless '
        '--beam-fwhm is given. The systematic block stays outside the beam.')
    bm.add_argument('--beam-table', default=None,
                    help='a measured beam: a radial table from '
                         'build_beam_table.py, a transfer already on this '
                         "map's grid (.npz with 'beam' (Nx, Ny, n_freq)), or "
                         'a pickle of azimuthal power profiles -- detected '
                         'automatically (imgibbs.load_beam_table). Takes '
                         'precedence over --beam-fwhm.')
    bm.add_argument('--beam-width-scale', type=float, default=1.0,
                    help='stretch the model beam by this factor in angle '
                         '(default 1; 1.02 is 2%% wider), for --beam-table '
                         'or --beam-fwhm. For testing how accurately the '
                         'beam must be known.')
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


def _mock_systematic(text):
    try:
        return _systematic_names(text)
    except ValueError as err:
        raise argparse.ArgumentTypeError(str(err))


def load_data(args):
    """The cube to sample, and the Mock behind it if there is one."""
    if args.data is None:
        names = args.mock_systematic or []
        sys_kwargs = ({} if args.mock_sys_rms is None else
                      {n: {'rms': args.mock_sys_rms} for n in names})
        mock = make_mock(shape=tuple(args.mock_shape), seed=args.mock_seed,
                         beam_fwhm=args.mock_beam_fwhm,
                         systematic=args.mock_systematic,
                         sys_kwargs=sys_kwargs)
        os.makedirs(args.out, exist_ok=True)
        mock.save(os.path.join(args.out, 'mock.npz'))
        print(mock.summary())
        print(f'mock         : written to {args.out}/mock.npz')
        return mock.cube, mock
    if args.data.endswith('.npz'):
        mock = load_mock(args.data)
        print(f'data         : {args.data} (mock)')
        return mock.cube, mock
    print(f'data         : {args.data}')
    return np.load(args.data), None


def main():
    args = parse_args()

    n_modes = args.n_modes
    flagging = not args.no_flagging
    suffix = args.suffix if args.suffix is not None else f'_LP{n_modes}_'
    sample_dir = os.path.join(args.out, 'samples')
    os.makedirs(sample_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Data and geometry
    # ------------------------------------------------------------------
    data_cube, mock = load_data(args)
    shape = data_cube.shape
    grid = mock_grid(shape)
    box_dims = grid.box_dims
    print(grid.summary())

    if flagging:
        flag = (data_cube != 0).astype(float)
        data_cube = data_cube * flag
        w = flag.flatten()
    else:
        w = np.ones(np.prod(shape))

    sig_k, idxs, kbin_meta = kbins_from_crop(data_cube, box_dims,
                                            max_bins=N_K_BINS)

    # Starting S: the expected power under the fiducial cosmology, from
    # realisations independent of the one in the data.
    if args.S_start == 'truth':
        if mock is None:
            raise SystemExit('--S-start truth needs a mock')
        pk0 = binned_power(mock.hi, sig_k, idxs)
    else:
        pk0, pk0_std = expected_hi_power(grid, sig_k, idxs,
                                         n_real=args.S_start_real)
    pk0 = pk0 * args.S_start_scale
    S = signal_prior(pk0, idxs, kbin_meta['dc_index'])
    print(f'S start      : {args.S_start} x {args.S_start_scale:g}, '
          + ', '.join(f'{p:.3e}' for p in pk0))

    # ------------------------------------------------------------------
    # Foreground model: orthonormalised Legendre polynomials in frequency
    # ------------------------------------------------------------------
    n_freq = shape[2]
    evecs = legendre_basis(n_freq, n_modes)
    print(f'evecs        : {evecs.shape}  orthonormality '
          f'{np.abs(evecs @ evecs.T - np.eye(n_modes)).max():.2e}')

    # ------------------------------------------------------------------
    # Beam
    # ------------------------------------------------------------------
    nu_ref = None
    if args.beam_table is not None:
        beam = load_beam_table(args.beam_table, grid.freqs, shape,
                               width_scale=args.beam_width_scale)
        print(f'beam         : {beam.source} '
              f'({grid.freqs[0]:.1f}-{grid.freqs[-1]:.1f} MHz)')
    elif args.beam_fwhm is not None:
        nu_ref = (args.beam_nu_ref if args.beam_nu_ref is not None
                  else float(grid.freqs.mean()))
        fwhm = args.beam_fwhm * args.beam_width_scale
        beam = gaussian_beam(grid.freqs, shape, fwhm, nu_ref,
                             index=args.beam_index)
        fw = beam_fwhm(grid.freqs[[0, -1]], fwhm, nu_ref, args.beam_index)
        print(f'beam         : Gaussian, FWHM {fw[0]:.3f} deg at '
              f'{grid.freqs[0]:.1f} MHz -> {fw[1]:.3f} deg at '
              f'{grid.freqs[-1]:.1f} MHz (index {args.beam_index:g})')
    else:
        beam = None
    if mock is not None and (mock.beam is None) != (beam is None):
        print('WARNING: the mock was ' + ('not ' if mock.beam is None else '')
              + 'beamed but the model is ' + ('' if beam is not None else
                                              'not ') + 'beamed')

    # ------------------------------------------------------------------
    # Systematics: ground spill
    # ------------------------------------------------------------------
    if args.groundspill:
        gs_smooth = (args.gs_smooth == 'on'
                     or (args.gs_smooth == 'auto' and beam is not None))
        if gs_smooth and args.gs_deflate:
            raise SystemExit('--gs-deflate would project the smooth template '
                             'out entirely; use one or the other '
                             '(--gs-smooth off to deflate)')
        sys_basis = groundspill_basis(
            grid.freqs, shape, period=args.gs_period,
            n_harmonics=args.gs_harmonics, order=args.gs_order,
            scan_axis=args.gs_scan_axis, include_smooth=gs_smooth,
            fg_basis=evecs if args.gs_deflate else None)
        G = np.full(sys_basis.g_shape, args.gs_prior ** 2)
        if gs_smooth:
            G[:, 0] = args.gs_smooth_prior ** 2     # spectral row 0 is smooth
        G = G.ravel()
        bandwidth = grid.freqs[-1] - grid.freqs[0]
        # A ripple is a single k_parallel mode: all of its power goes into one
        # bin rather than spreading. Print where, so a run that is about to
        # contaminate the lowest signal bin says so before it starts.
        k_ripple = [ripple_wavenumber(args.gs_period / m, bandwidth,
                                      box_dims[2])
                    for m in range(1, args.gs_harmonics + 1)]
        print(f'groundspill  : {sys_basis.n_params} params '
              f'({sys_basis.n_s} spatial x {sys_basis.n_t} spectral), '
              f'prior std {args.gs_prior:g} K'
              + (f', smooth {args.gs_smooth_prior:g} K' if gs_smooth else ''))
        print(f'               period {args.gs_period:g} MHz -> k_par '
              + ', '.join(f'{k:.4f}' for k in k_ripple) + ' Mpc^-1')
        for k in k_ripple:
            bin_i = int(np.argmin(np.abs(sig_k - k)))
            print(f'               k = {k:.4f} is nearest bin {bin_i} '
                  f'(k = {sig_k[bin_i]:.4f})')
    else:
        sys_basis = G = None
        gs_smooth = False

    # In a joint solve the smooth spill template is almost reproducible by
    # B Uf f, and the solver barely moves its amplitude (docs/STATUS.md,
    # finding 9 of 'The beam in the MODEL'). Projecting that part out leaves
    # what the model can fit unchanged and removes that near-null direction,
    # though not the one against B Us s. Integrating g out removes both.
    gs_solve = (None if not (args.groundspill or args.onef or args.leakage)
                else resolve_gs_solve(args.gs_solve, beam))
    if gs_smooth and beam is not None and gs_solve == 'joint':
        sys_basis = beam_project(sys_basis, beam.full_transfer(), evecs,
                                 rows=[0])
        print('               smooth templates: B Uf projected out')

    # ------------------------------------------------------------------
    # Systematics: 1/f and leakage, appended to the same g
    # ------------------------------------------------------------------
    blocks, priors, names = [], [], []
    if sys_basis is not None:
        blocks, priors, names = [sys_basis], [G], ['groundspill']
    if args.onef:
        b, _ = onef_basis(grid.freqs, shape, fg_basis=evecs)
        blocks.append(b)
        priors.append(np.full(b.n_params, args.onef_prior ** 2))
        names.append('onef')
        print(f'1/f          : {b.n_params} params ({b.n_s} scan x {b.n_t} '
              f'spectral KL modes, foreground span deflated), prior std '
              f'{args.onef_prior:g} K')
    if args.leakage:
        b = leakage_basis(grid.freqs, shape, rm=args.leak_rm)
        blocks.append(b)
        priors.append(np.full(b.n_params, args.leak_prior ** 2))
        names.append('leakage')
        print(f'leakage      : {b.n_params} params ({b.n_s} spatial x '
              f'{b.n_t} spectral), RM {args.leak_rm:g} rad m^-2, prior std '
              f'{args.leak_prior:g} K')
    if len(blocks) > 1:
        sys_basis = CompositeBasis(blocks, names)
        G = np.concatenate(priors)
        ov = sys_basis.overlap(w)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                note = ('  <- nearly degenerate: the data fix the sum, the '
                        'prior the split' if ov[i, j] > 0.9 else '')
                print(f'overlap      : {names[i]} / {names[j]} '
                      f'{ov[i, j]:.3f}{note}')
    elif blocks:
        sys_basis, G = blocks[0], priors[0]
    if gs_solve is not None:
        print(f'g solve      : {gs_solve}, {sys_basis.n_params} amplitudes')

    # ------------------------------------------------------------------
    # Noise
    # ------------------------------------------------------------------
    N = (T_SYS**2) / (DEL_NU * DEL_T)
    print(f'noise        : T_sys {T_SYS:g} K, channel {DEL_NU/1e6:.6f} MHz, '
          f'N = {N:.4e} K^2')

    burn = args.burn if args.burn is not None else args.n_samples // 4
    config = {'shape': list(shape), 'box_dims': list(box_dims),
              'n_modes': n_modes, 'n_samples': args.n_samples,
              'T_sys': T_SYS, 'del_nu_Hz': DEL_NU, 'del_t_s': DEL_T,
              'tol': args.tol, 'seed': args.seed, 'flagging': flagging,
              'burn': burn,
              'S_start': args.S_start, 'S_start_scale': args.S_start_scale,
              'S_start_pk': pk0.tolist(),
              'true_pk': (None if mock is None else
                          binned_power(mock.hi, sig_k, idxs).tolist()),
              'fix_S': bool(args.fix_S),
              'groundspill': bool(args.groundspill),
              'gs_period_MHz': args.gs_period if args.groundspill else None,
              'gs_harmonics': args.gs_harmonics if args.groundspill else None,
              'gs_order': args.gs_order if args.groundspill else None,
              'gs_scan_axis': args.gs_scan_axis if args.groundspill else None,
              'gs_prior_K': args.gs_prior if args.groundspill else None,
              'gs_deflate': bool(args.groundspill and args.gs_deflate),
              'gs_smooth': bool(args.groundspill and gs_smooth),
              'gs_smooth_projected': bool(args.groundspill and gs_smooth
                                          and beam is not None
                                          and gs_solve == 'joint'),
              'gs_solve': gs_solve,
              'sys_blocks': names,
              'onef': bool(args.onef),
              'onef_prior_K': args.onef_prior if args.onef else None,
              'leakage': bool(args.leakage),
              'leak_rm': args.leak_rm if args.leakage else None,
              'leak_prior_K': args.leak_prior if args.leakage else None,
              'gs_smooth_prior_K': (args.gs_smooth_prior
                                    if args.groundspill and gs_smooth
                                    else None),
              'data': args.data,
              'mock': None if mock is None else mock.config,
              'beam_table': args.beam_table,
              'beam_collapse': beam is not None and not args.no_collapse,
              'beam_threshold': (args.beam_threshold if beam is not None
                                 else None),
              'fg_beam_groups': args.fg_beam_groups,
              'fg_refresh': args.fg_refresh,
              'beam_fwhm_deg': args.beam_fwhm,
              'beam_nu_ref_MHz': nu_ref,
              'beam_width_scale': args.beam_width_scale,
              'beam_index': args.beam_index if beam is not None else None,
              'sig_k': sig_k.tolist(), **kbin_meta}

    print(f'\nsampling     : {args.n_samples} samples, {n_modes} FG modes, '
          f'traces -> {sample_dir}/*{suffix}.npy')
    chain = run_chain(
        data_cube, w, evecs, S, sig_k, idxs, kbin_meta['dc_index'], N,
        args.n_samples, tol=args.tol, seed=args.seed,
        sys_basis=sys_basis, G=G, gs_solve=args.gs_solve, beam=beam,
        collapse=not args.no_collapse, beam_threshold=args.beam_threshold,
        fg_dense_modes=args.fg_dense_modes, fg_beam_groups=args.fg_beam_groups,
        fg_refresh=args.fg_refresh, f_prior_mean=args.f_prior_mean,
        fix_S=args.fix_S, mean_from=burn,
        sample_dir=sample_dir, suffix=suffix,
        save_x_every=args.save_x_every, save_S=args.save_S)

    config.update(f_prior_mean=chain.f_prior_mean,
                  beam_visible_modes_per_bin=chain.visible_modes,
                  fg_dense_modes=(min(args.fg_dense_modes, n_modes)
                                  if beam is not None else 0))
    with open(os.path.join(args.out, f'run{suffix}config.json'), 'w') as fh:
        json.dump(config, fh, indent=2, default=str)
    np.save(os.path.join(args.out, f'matvecs{suffix}.npy'), chain.matvecs)
    if beam is not None and args.fg_dense_modes > 0:
        np.save(os.path.join(args.out, f'fg_refreshes{suffix}.npy'),
                np.array(chain.fg_refreshes))
    np.save(os.path.join(args.out, f'signal_mean{suffix}.npy'),
            chain.signal_mean)
    np.save(os.path.join(args.out, f'fg_mean{suffix}.npy'), chain.fg_mean)
    print(f'\n{args.n_samples} samples with {n_modes} FG modes in '
          f'{chain.elapsed:.1f} s ({chain.elapsed / args.n_samples:.2f} '
          f's/sample, median {np.median(chain.matvecs):.0f} matvecs/solve)')


if __name__ == '__main__':
    main()
