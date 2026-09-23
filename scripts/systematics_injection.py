#!/usr/bin/env python
"""Does modelling ground spill actually buy anything? Injection test.

``docs/STATUS.md`` records the structural limitation of the real-data
comparison: the sampler only ever sees ``data_cube``, so "the method destroyed
the signal" and "there was no signal" are indistinguishable. That applies
doubly to a systematic -- without injecting one you cannot tell a systematic
that was removed from one that was never there.

So this runs entirely on a **synthetic cube with a known answer**::

    d = HI (simulated) + foreground + ground spill + noise

built on the live (70, 45, 250) grid with the real footprint mask, and samples
it twice with identical seeds:

  * ``--arm off``  : the published three-block model, s + f
  * ``--arm on``   : with the ground-spill block, s + f + g

Everything except the systematic block is identical between the two arms, so
the difference in the recovered P(k) is attributable to the block and nothing
else.

What to look at in the summary
------------------------------
1. **The smooth spill should not matter.** It is injected at ``--spill-level``
   (0.5 K by default, ~600x the H I rms) and the foreground block absorbs it
   completely. If arm "off" is not already fine in the bins away from the
   ripple, something other than the systematic is wrong.
2. **The ripple should matter, in one bin.** It is a single ``k_parallel``
   mode, so it contaminates whichever bin contains it and leaves the rest
   alone. The script prints which bin before it starts.
3. **``g`` should be recovered.** ``g_true`` is known exactly because the
   ripple is built from the same basis the sampler uses.

Usage
-----
    python scripts/groundspill_injection.py --arm off --n-samples 120
    python scripts/groundspill_injection.py --arm on  --n-samples 120
    python scripts/groundspill_injection.py --summarise

Use ``--quick`` to cut the frequency axis to 64 channels for a smoke test;
the k-binning then no longer matches the shipped ``S``, so a quick run proves
the plumbing and nothing about the science.
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
    Us, best_fit_amplitudes, bin_it, construct_A, construct_b,
    construct_preconditioner, groundspill_basis, groundspill_cube,
    kbins_from_crop, leakage_basis, load, load_l2021_cube, onef_basis,
    period_scan, realise, ripple_wavenumber, survey_grid,
    foreground_covariance_sampler as FCS,
    signal_covariance_sampler as SCS,
)

CROP = (slice(33, 103), slice(14, 59), slice(0, 250))
N_K_BINS = 5
T_SYS, DEL_T = 16.0, 1000.0
DEL_NU = (1712.0 - 856.0) / 4096 * 1e6      # Hz

#: Seed for the synthetic cube. Fixed and separate from the chain seed, so
#: both arms see the identical realisation of signal, foreground and spill.
TRUTH_SEED = 20260923


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--arm', choices=('on', 'off', 'clean'),
                   help="'on' injects ground spill and samples the block; "
                        "'off' injects it and does not; 'clean' injects "
                        "NOTHING and is the control -- what this sampler "
                        "recovers when there is no systematic to recover from. "
                        "Without it you cannot tell residual bias in the other "
                        "two arms from the sampler's own behaviour.")
    p.add_argument('--summarise', action='store_true',
                   help='compare two finished arms instead of running one')
    p.add_argument('--n-modes', type=int, default=6,
                   help='foreground (Legendre) modes (default: 6)')
    p.add_argument('--n-samples', type=int, default=120)
    p.add_argument('--burn', type=int, default=20,
                   help='samples to discard. S starts flat, so iteration 0 is '
                        'degenerate -- check the trace, do not trust this.')
    p.add_argument('--out', default='outputs/groundspill')
    p.add_argument('--tol', type=float, default=1e-6)
    p.add_argument('--seed', type=int, default=1,
                   help='chain seed. Both arms MUST use the same one.')
    p.add_argument('--quick', action='store_true',
                   help='64 channels, for a plumbing smoke test only')
    p.add_argument('--fix-S', action='store_true',
                   help='DIAGNOSTIC, not an analysis. Hold S at the shipped '
                        'prior (the true simulated H I power) instead of '
                        'sampling it. On a real survey you do not know P(k) in '
                        'advance, so this is not a usable configuration -- it '
                        'exists to isolate the S feedback: S inflates in the '
                        'contaminated bin, which makes it cheap for the SIGNAL '
                        'block to absorb ripple power, and g gives it up. If '
                        'bin 0 recovers here and not with --arm on, that is '
                        'the mechanism confirmed. See docs/STATUS.md.')
    p.add_argument('--thin', type=int, default=10,
                   help='save the signal cube every Nth sample (default: 10). '
                        'Pk and g traces are written every sample regardless; '
                        'they are a few floats. The cubes are 3.2 MB each, so '
                        'a 3000-sample chain is 9.5 GB unthinned. Thinning '
                        'does not bias the posterior mean.')

    p.add_argument('--systematic', choices=('groundspill', 'onef', 'leakage'),
                   default='groundspill',
                   help="which systematic to inject and model. Use a SEPARATE "
                        "--out directory per systematic; the report script "
                        "discovers arms by name within one directory.")

    g = p.add_argument_group('what to inject')
    g.add_argument('--ripple-rms', type=float, default=1e-3,
                   help='RMS of the injected ripple, K (default: 1e-3, which '
                        'is ~1.2x the simulated H I rms of 8.6e-4 K)')
    g.add_argument('--spill-level', type=float, default=0.5,
                   help='RMS of the injected SMOOTH spill, K (default: 0.5). '
                        'Large on purpose -- it should make no difference.')
    g.add_argument('--period', type=float, default=17.5,
                   help='standing-wave period, MHz (default: 17.5)')
    g.add_argument('--gradient', type=float, default=0.3,
                   help='scan-direction gradient relative to the constant term')
    g.add_argument('--sys-prior', type=float, default=1e-2,
                   help='prior std on the systematic amplitudes, K. For '
                        '1/f this scales the KL eigenvalues, which supply the '
                        'RELATIVE variances; for the other two it is flat.')
    g.add_argument('--rm', type=float, default=1000.0,
                   help='Faraday depth for --systematic leakage, rad/m^2 '
                        '(default: 1000). Anything below ~500 is absorbed '
                        'whole by the foreground on a 52 MHz band -- see '
                        'imgibbs/systematics.py.')
    g.add_argument('--alpha', type=float, default=1.0,
                   help='1/f index along the scan direction')
    g.add_argument('--beta', type=float, default=1.0,
                   help='1/f correlation index across frequency')
    g.add_argument('--knee-cycles', type=float, default=1.0,
                   help='1/f knee, in cycles across the scan')
    g.add_argument('--n-scan', type=int, default=2,
                   help='1/f: KL modes kept along the scan')
    g.add_argument('--n-spec', type=int, default=2,
                   help='1/f: KL modes kept across frequency')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Which systematic
# ---------------------------------------------------------------------------

def build_basis(args, freqs, shape, fg_basis):
    """The basis the sampler models with, and the RELATIVE prior variances.

    Returns ``(basis, prior_var)`` where ``prior_var`` sums to 1. ``G`` is then
    ``prior_var * sys_prior**2``.

    Only 1/f has a derivation for the relative variances -- they are the KL
    eigenvalues of its own covariance. Ground spill and leakage get a flat
    prior, because nothing in their models says one template should carry more
    amplitude than another.
    """
    if args.systematic == 'groundspill':
        basis = groundspill_basis(freqs, shape, period=args.period)
    elif args.systematic == 'leakage':
        basis = leakage_basis(freqs, shape, rm=args.rm, order=1)
    elif args.systematic == 'onef':
        basis, prior_var = onef_basis(
            freqs, shape, alpha=args.alpha, beta=args.beta,
            knee_cycles=args.knee_cycles, n_scan=args.n_scan,
            n_spec=args.n_spec, fg_basis=fg_basis)
        return basis, prior_var
    else:
        raise SystemExit(f'unknown systematic {args.systematic}')
    return basis, np.full(basis.g_shape, 1.0 / basis.n_params)


# ---------------------------------------------------------------------------
# The synthetic cube
# ---------------------------------------------------------------------------

def build_truth(args):
    """Simulated H I + foreground + ground spill + noise, on the real mask.

    The foreground is the least-squares Legendre model of the *real* L2021
    cube, so its amplitude and spectral shape are realistic rather than
    invented -- it is ~140x the H I rms, which is the regime that makes this
    problem hard. Only the H I, the spill and the noise are synthetic.
    """
    crop = CROP if not args.quick else (CROP[0], CROP[1], slice(0, 64))
    rng = np.random.default_rng(TRUTH_SEED)
    # The control arm sees the same H I, foreground and noise realisation --
    # same TRUTH_SEED -- with the systematic switched off, so any difference
    # from it is attributable to the ground spill and nothing else.
    ripple_rms = 0.0 if args.arm == 'clean' else args.ripple_rms
    spill_level = 0.0 if args.arm == 'clean' else args.spill_level

    real = load_l2021_cube()[crop]
    shape = real.shape
    grid = survey_grid(crop, shape)

    flag = (real != 0).astype(float)

    # Foreground: project the real cube onto the same Legendre basis the
    # sampler uses. Keeps the foreground realistic without needing a model.
    n_freq = shape[2]
    basis = np.linalg.qr(np.polynomial.legendre.legvander(
        np.linspace(-1, 1, n_freq), args.n_modes - 1))[0].T
    foreground = ((real.reshape(-1, n_freq) @ basis.T) @ basis).reshape(shape)

    hi = load('Fastbox_cube_cropped.npy')[:, :, :n_freq]

    if ripple_rms == 0.0 and spill_level == 0.0:
        # Control arm: build the basis anyway so g_true has a shape, but
        # inject nothing.
        sys_basis, prior_var = build_basis(args, grid.freqs, shape, basis)
        spill = np.zeros(shape)
        g_true = np.zeros(sys_basis.g_shape)
    elif args.systematic == 'groundspill':
        spill, truth = groundspill_cube(
            grid.freqs, shape, ripple_rms=ripple_rms,
            spill_level=spill_level, period=args.period,
            gradient=args.gradient, rng=rng)
        sys_basis, g_true = truth['basis'], truth['g_true']
    else:
        # 1/f and leakage are drawn from their own basis, with the relative
        # variances the model claims. For 1/f that is the KL spectrum, so the
        # injected realisation really does have the covariance being assumed.
        sys_basis, prior_var = build_basis(args, grid.freqs, shape, basis)
        spill, g_true = realise(sys_basis, prior_var, ripple_rms, rng=rng)

    noise_rms = np.sqrt(T_SYS ** 2 / (DEL_NU * DEL_T))
    noise = rng.normal(scale=noise_rms, size=shape)

    cube = (hi + foreground + spill + noise) * flag
    return dict(cube=cube, shape=shape, grid=grid, crop=crop, flag=flag,
                hi=hi, foreground=foreground, spill=spill,
                basis=sys_basis, g_true=g_true, noise_rms=noise_rms)


# ---------------------------------------------------------------------------
# One arm
# ---------------------------------------------------------------------------

def run_arm(args):
    np.random.seed(args.seed)
    on = args.arm == 'on'
    suffix = f'_{args.arm}{"fixedS" if args.fix_S else ""}_'
    sample_dir = os.path.join(args.out, 'samples')
    os.makedirs(sample_dir, exist_ok=True)

    truth = build_truth(args)
    cube, shape, grid = truth['cube'], truth['shape'], truth['grid']
    box_dims = grid.box_dims
    w = truth['flag'].flatten()
    print(grid.summary())
    if args.arm == 'clean':
        print(f"\ninjected     : HI rms {truth['hi'].std():.3e} K, "
              f"noise {truth['noise_rms']:.3e} K, NO systematic (control arm)")
    elif args.systematic == 'groundspill':
        print(f"\ninjected     : HI rms {truth['hi'].std():.3e} K, "
              f"ripple {args.ripple_rms:.3e} K, smooth spill "
              f"{args.spill_level:.3e} K, noise {truth['noise_rms']:.3e} K")
    else:
        # --spill-level is a ground-spill knob and is NOT applied here, so do
        # not print it: the log is the record of what was injected.
        print(f"\ninjected     : HI rms {truth['hi'].std():.3e} K, "
              f"{args.systematic} {args.ripple_rms:.3e} K rms, "
              f"noise {truth['noise_rms']:.3e} K")
        print(f"               (--spill-level does not apply to "
              f"{args.systematic} and was ignored)")
    print(f"foreground   : rms {truth['foreground'].std():.3e} K "
          f"({truth['foreground'].std() / truth['hi'].std():.0f}x the HI)")

    sig_k, idxs, kbin_meta = kbins_from_crop(cube, box_dims, max_bins=N_K_BINS)
    occupied = np.unique(idxs)
    occupied = occupied[occupied > 0]

    # Only a fixed-period ripple is a single k_parallel mode. Leakage is
    # periodic in lambda^2, so its period drifts across the band and its power
    # spreads over a range of k; 1/f is broadband by construction. For those
    # two "the contaminated bin" is not a well-defined idea, and the reported
    # bin is the peak rather than the whole story.
    k_ripple = ripple_wavenumber(args.period, grid.freqs[-1] - grid.freqs[0],
                                 box_dims[2])
    hit = int(np.argmin(np.abs(sig_k - k_ripple)))
    if args.systematic == 'groundspill':
        print(f'ripple       : period {args.period:g} MHz -> k_par '
              f'{k_ripple:.4f} Mpc^-1, nearest bin {hit} (k = {sig_k[hit]:.4f})')
    else:
        print(f'spread       : {args.systematic} is NOT a single k_parallel '
              f'mode; its power spreads across bins')
    print(f'k-bins       : ' + ', '.join(f'{k:.4f}' for k in sig_k))

    # S: the shipped starting point matches only the full 250-channel grid.
    if args.quick:
        S = np.full(int(np.prod(shape)), 1e30)
    else:
        S = load('S_starting_point_cropped.npy')
        if S.size != np.prod(shape):
            raise SystemExit(f'S has {S.size} entries, grid has '
                             f'{np.prod(shape)} voxels')

    n_freq = shape[2]
    evecs = np.linalg.qr(np.polynomial.legendre.legvander(
        np.linspace(-1, 1, n_freq), args.n_modes - 1))[0].T

    # How much of this ripple can the foreground block take on its own? The
    # systematic block can only ever recover the remainder, so this number is
    # the ceiling on what the run below can demonstrate. Worth printing on
    # every run -- it depends on the period AND the bandwidth, so a band cut
    # that looked harmless can quietly make the ripple unidentifiable.
    bandwidth = grid.freqs[-1] - grid.freqs[0]
    if args.systematic == 'groundspill':
        absorbed = float(period_scan(grid.freqs, [args.period], evecs)[0])
        print(f'identifiable : {bandwidth / args.period:.2f} cycles across the '
              f'band; the {args.n_modes}-mode foreground basis absorbs '
              f'{absorbed:.1%} of it')
    else:
        # Same question, asked of whatever templates this systematic uses:
        # how much of the basis lies inside the foreground span?
        probe, _ = build_basis(args, grid.freqs, shape, evecs)
        t = probe.spectral / np.linalg.norm(probe.spectral, axis=1)[:, None]
        absorbed = float(np.mean(np.sum((evecs @ t.T) ** 2, axis=0)))
        print(f'identifiable : the {args.n_modes}-mode foreground basis '
              f'absorbs {absorbed:.1%} of the {args.systematic} templates')
    if absorbed > 0.5:
        print(f'  WARNING: the foreground block already takes {absorbed:.0%} '
              f'of this ripple, so at most {1 - absorbed:.0%} is left for the '
              f'systematic block to find. Expect g to come back near zero -- '
              f'that is the correct answer, not a failure. Shorten --period '
              f'or widen the band.')
    print(f'               ceiling on recoverable amplitude: '
          f'{1 - absorbed:.1%} of the injected value')

    sys_basis = G = g_mean = None
    if on:
        # Built with the same parameters the systematic was injected with. A
        # real run does not know these; getting them wrong is the interesting
        # failure mode, and --period / --rm are the knobs to probe it with.
        sys_basis, prior_var = build_basis(args, grid.freqs, shape, evecs)
        G = (np.asarray(prior_var).ravel() * args.sys_prior ** 2
             * sys_basis.n_params)
        g_mean = np.zeros(sys_basis.g_shape)
        print(f'systematic   : {args.systematic} ON, {sys_basis.n_params} '
              f'params ({sys_basis.n_s} spatial x {sys_basis.n_t} spectral), '
              f'prior std {args.sys_prior:g} K')
    else:
        print(f'systematic   : {args.systematic} injected, NOT modelled '
              f'(s + f only)')

    d_2d = cube.reshape(-1, n_freq)
    s_true = Us(cube, True)
    rfft_len, rfft_shape = s_true.size, s_true.shape
    f_mean = (d_2d @ evecs.T).reshape(shape[0], shape[1], args.n_modes)
    f_len, f_shape = f_mean.size, f_mean.shape
    g_len = 0 if sys_basis is None else sys_basis.n_params
    total_len = 2 * rfft_len + f_len + g_len

    N = T_SYS ** 2 / (DEL_NU * DEL_T)
    N_inv, N_inv_scalar, Nw_inv = 1 / N, 1 / N, (1 / N) * w
    s_mean = np.zeros(rfft_shape)

    f_init = f_mean * np.random.normal(1.0, 0.05, f_mean.shape)
    F = np.diag(FCS(f_init.reshape(-1, args.n_modes)))

    x = np.concatenate([s_mean.real.flatten(), s_mean.imag.flatten(),
                        f_mean.flatten()]
                       + ([g_mean.flatten()] if on else []))

    L = LinearOperator(
        matvec=lambda v: construct_A(v, S, Nw_inv, F, w, evecs, rfft_len,
                                     rfft_shape, f_len, f_shape, shape,
                                     sys_basis=sys_basis, G=G).flatten(),
        shape=(total_len, total_len))
    precond = construct_preconditioner(S, N_inv_scalar, F, evecs, rfft_len,
                                       rfft_shape, f_len, f_shape, shape,
                                       sys_basis=sys_basis, G=G)

    meta = dict(arm=args.arm + ('fixedS' if args.fix_S else ''),
                systematic=args.systematic, absorbed_by_fg=absorbed,
                fix_S=bool(args.fix_S), thin=args.thin, n_modes=args.n_modes, n_samples=args.n_samples,
                burn=args.burn, seed=args.seed, tol=args.tol,
                quick=args.quick, cube_shape=list(shape),
                box_dims=list(box_dims), sig_k=sig_k.tolist(),
                ripple_k=float(k_ripple), ripple_bin=hit,
                ripple_rms=args.ripple_rms, spill_level=args.spill_level,
                period=args.period, gradient=args.gradient,
                sys_prior=args.sys_prior, g_true=truth['g_true'].tolist(),
                hi_rms=float(truth['hi'].std()), kbins=kbin_meta)
    print(f'S            : {"HELD at the prior (diagnostic)" if args.fix_S else "sampled each iteration"}')
    with open(os.path.join(args.out, f'run{suffix}meta.json'), 'w') as fh:
        json.dump(meta, fh, indent=2, default=str)

    print(f'\nsampling     : {args.n_samples} samples -> {sample_dir}')
    print(f'               signal cubes every {args.thin} samples '
          f'({args.n_samples // args.thin} of them, '
          f'{args.n_samples // args.thin * np.prod(shape) * 4 / 1e9:.1f} GB)')
    start = time.time()
    for rr in tqdm(range(args.n_samples)):
        x0 = x
        ws = np.fft.rfftn(np.random.normal(size=shape), norm='ortho').flatten()
        wf = np.random.normal(size=f_shape)
        wd = np.random.normal(size=shape).flatten()
        wg = np.random.normal(size=sys_basis.g_shape) if on else None

        b = construct_b(S, N_inv, F, w, s_mean, f_mean, evecs, cube,
                        ws, wf, wd, shape, sys_basis=sys_basis, G=G,
                        wg=wg, g_mean=g_mean)
        M_inv = LinearOperator((total_len, total_len), matvec=precond)
        x, code = lgmres(L, b.flatten(), x0=x0, rtol=args.tol, atol=0, M=M_inv)
        if code != 0:
            print(f'  [{rr}] lgmres did not converge (exit {code})')

        s = (x[:rfft_len].reshape(rfft_shape)
             + x[rfft_len:2 * rfft_len].reshape(rfft_shape) * 1j)
        s = Us(s, False)
        if rr % args.thin == 0:
            np.save(os.path.join(sample_dir, f's_cube{rr}{suffix}.npy'),
                    s.astype(np.float32))
        s = np.fft.fftn(s - np.mean(s), norm='ortho')

        binned_s, k_bins = bin_it(s, sig_k, idxs)
        if args.fix_S:
            # S is NOT updated. Report the binned power of the sampled signal
            # field -- the quantity SCS would have drawn from -- so the trace
            # stays comparable to the other arms.
            PkSample = np.array([np.mean(np.abs(b) ** 2) for b in binned_s])
        else:
            _, PkSample = SCS(np.concatenate(binned_s), np.concatenate(k_bins))
            S = np.zeros(len(idxs))
            for gg, b_i in enumerate(occupied):
                S[idxs == b_i] = PkSample[gg]
            S[idxs == 0] = 1e-12 * np.median(PkSample)
            S[kbin_meta['dc_index']] = 1e30
        np.save(os.path.join(sample_dir, f'Pk_trace{rr}{suffix}.npy'), PkSample)

        if on:
            g_off = 2 * rfft_len + f_len
            np.save(os.path.join(sample_dir, f'g_trace{rr}{suffix}.npy'),
                    x[g_off:g_off + g_len].reshape(sys_basis.g_shape))

        f_ms = x[2 * rfft_len:2 * rfft_len + f_len].reshape(f_shape).real
        F = np.diag(FCS((f_ms - f_mean).reshape(-1, args.n_modes)))
        precond = construct_preconditioner(S, N_inv_scalar, F, evecs, rfft_len,
                                           rfft_shape, f_len, f_shape, shape,
                                           sys_basis=sys_basis, G=G)
        gc.collect()

    dt = time.time() - start
    print(f'\n{args.n_samples} samples in {dt:.1f} s '
          f'({dt / args.n_samples:.2f} s/sample)')
    if on:
        report_g(args, truth, sample_dir, suffix)


def report_g(args, truth, sample_dir, suffix):
    """How well was the injected ``g`` recovered?"""
    g = np.array([np.load(os.path.join(sample_dir, f'g_trace{r}{suffix}.npy'))
                  for r in range(args.burn, args.n_samples)])
    mean, std = g.mean(axis=0), g.std(axis=0)
    print('\nground-spill amplitudes (K), posterior vs truth')
    print('  spatial  spectral      true     recovered      pull')
    for i in range(mean.shape[0]):
        for j in range(mean.shape[1]):
            t = truth['g_true'][i, j]
            pull = (mean[i, j] - t) / std[i, j] if std[i, j] > 0 else np.nan
            print(f'  {i:^7d}  {j:^8d}  {t:+.3e}  {mean[i, j]:+.3e}'
                  f' +- {std[i, j]:.1e}  {pull:+6.2f}')


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def summarise(args):
    """Compare the two arms. The truth cube is rebuilt, not reloaded."""
    truth = build_truth(args)
    metas, pk = {}, {}
    for arm in ('off', 'on'):
        path = os.path.join(args.out, f'run_{arm}_meta.json')
        if not os.path.exists(path):
            raise SystemExit(f'{path} not found -- run --arm {arm} first')
        metas[arm] = json.loads(Path(path).read_text())
        n, burn = metas[arm]['n_samples'], metas[arm]['burn']
        pk[arm] = np.array([
            np.load(os.path.join(args.out, 'samples',
                                 f'Pk_trace{r}_{arm}_.npy'))
            for r in range(burn, n)])

    if metas['off']['seed'] != metas['on']['seed']:
        print('WARNING: the two arms used different chain seeds, so part of '
              'the difference below is just chain noise')

    sig_k = np.array(metas['on']['sig_k'])
    hit = metas['on']['ripple_bin']

    _, idxs, kbin_meta = kbins_from_crop(truth['cube'], truth['grid'].box_dims,
                                         max_bins=N_K_BINS)
    occupied = np.unique(idxs)
    occupied = occupied[occupied > 0]
    flag = truth['flag']
    w2 = (flag ** 2).mean()

    def pk_footprint(cube):
        """P(k) restricted to the observed footprint, normalised by <w^2>.

        ``docs/STATUS.md``: 41% of the cube is inpainting, so the box average
        the ``Pk_trace`` files record mixes real sky with prior draws and sits
        up to 15x off at mid k. The footprint-restricted estimator is the one
        that agrees with the PCA transfer function (median ratio 1.03). Both
        are printed below; read the footprint one.
        """
        mu = cube[flag == 1].mean()
        A = np.fft.fftn((cube - mu) * flag, norm='ortho').flatten()
        p = (A * np.conj(A)).real
        return np.array([p[idxs == b].mean() for b in occupied]) / w2

    true_pk = pk_footprint(truth['hi'])

    # Footprint P(k) of the posterior-mean signal, per arm.
    pk_fp = {}
    for arm in ('off', 'on'):
        n, burn = metas[arm]['n_samples'], metas[arm]['burn']
        acc = None
        for r in range(burn, n):
            s = np.load(os.path.join(args.out, 'samples',
                                     f's_cube{r}_{arm}_.npy')).astype(float)
            acc = s if acc is None else acc + s
        pk_fp[arm] = pk_footprint(acc / (n - burn))

    print(f'\ninjected ripple at k = {metas["on"]["ripple_k"]:.4f} Mpc^-1 '
          f'-> bin {hit}')
    print(f'injected ripple RMS {metas["on"]["ripple_rms"]:.2e} K vs '
          f'H I rms {metas["on"]["hi_rms"]:.2e} K\n')

    print('  FOOTPRINT-RESTRICTED P(k)  (the one to read)')
    print('  bin       k     true P(k)      s+f  (ratio)      s+f+g  (ratio)')
    for i, k in enumerate(sig_k):
        a, b = pk_fp['off'][i], pk_fp['on'][i]
        mark = '  <-- ripple' if i == hit else ''
        print(f'  {i:^3d} {k:7.4f}  {true_pk[i]:.3e}  '
              f'{a:.3e} ({a / true_pk[i]:7.2f})  '
              f'{b:.3e} ({b / true_pk[i]:7.2f}){mark}')

    print('\n  BOX-AVERAGE P(k)  (what Pk_trace records; mixes in inpainting)')
    print('  bin       k          s+f          s+f+g')
    for i, k in enumerate(sig_k):
        mark = '  <-- ripple' if i == hit else ''
        print(f'  {i:^3d} {k:7.4f}  {pk["off"][:, i].mean():.3e}  '
              f'{pk["on"][:, i].mean():.3e}{mark}')

    off_r = pk_fp['off'][hit] / true_pk[hit]
    on_r = pk_fp['on'][hit] / true_pk[hit]
    print(f'\ncontaminated bin {hit}: s+f is {off_r:.2f}x the true H I power, '
          f's+f+g is {on_r:.2f}x')
    other = [i for i in range(len(sig_k)) if i != hit]
    drift = np.median(np.abs(pk_fp['on'][other] / pk_fp['off'][other] - 1))
    print(f'other bins: the two arms differ by a median of {drift * 100:.1f}% '
          f'-- the block should do nothing away from the ripple')


def main():
    args = parse_args()
    if args.summarise:
        summarise(args)
    elif args.arm:
        run_arm(args)
    else:
        raise SystemExit('give --arm on / --arm off, or --summarise')


if __name__ == '__main__':
    main()
