#!/usr/bin/env python
"""Known-answer test for the beam: build beamed mocks, and summarise the arms.

The comparison only means anything with an injected signal (see "The
comparison cannot work without injection" in docs/STATUS.md), so this builds
synthetic cubes on the live grid and mask::

    d = w * (B (HI + FG) + n)

HI is the simulated FastBox cube shipped in ``data/``; FG is the least-squares
Legendre model of the real L2021 cube (same as ``systematics_injection.py``),
so its amplitude and spectral shape are realistic; ``B`` is the beam; the
noise has the sampler's own ``T_sys``. A ``control`` mock is the same sky and
noise with no beam.

Build both mocks, with a Gaussian beam by default::

    python scripts/beam_injection.py build [--beam-fwhm 1.5 --beam-index 1]

or with any tabulated beam you have (format: ``imgibbs.load_beam_table``)::

    python scripts/beam_injection.py build --beam-table my_beam.npz

A smooth Gaussian exercises the deconvolution, the collapsed ``S`` step and
the preconditioner, but leaks almost no foreground (docs/STATUS.md, "The beam
in the MODEL"); a measured beam with spectral ripple is the harder test.
``build`` writes ``beam_args.txt`` beside the mocks -- the ``run_gibbs.py``
flags for the same beam -- so the sampler models exactly the beam the mock
was made with. Then sample the mocks (``submit_beam_injection.sh`` runs the
standard arms), then::

    python scripts/beam_injection.py summarise outputs/beam_injection

which compares each arm's P(k) chain with the true power of the injected HI,
both over every mode in a bin and over the beam-visible modes only.
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

from imgibbs import (                                          # noqa: E402
    beam_from_spec, beam_run_args, bin_it, data_path, kbins_from_crop, load,
    load_l2021_cube,
)
import imgibbs.grid as grid_mod                                # noqa: E402

CROP = (slice(33, 103), slice(14, 59), slice(0, 250))
N_K_BINS = 5
N_MODES = 6
T_SYS = 16.0
DEL_T = 1000.0
DEL_NU = (1712.0 - 856.0) / 4096 * 1e6
TRUTH_SEED = 20261019


def freqs_for(crop):
    """Channel frequencies without survey_grid's pyccl dependency."""
    dnu = (grid_mod.NU1 - grid_mod.NU0) / grid_mod.NCHAN
    return grid_mod.NU0 + np.arange(grid_mod.CH0, grid_mod.CH1)[crop[2]] * dnu


def legendre(n_freq, n_modes):
    return np.linalg.qr(np.polynomial.legendre.legvander(
        np.linspace(-1, 1, n_freq), n_modes - 1))[0].T


def build(args):
    real = load_l2021_cube()[CROP]
    shape = real.shape
    flag = (real != 0).astype(float)
    ev = legendre(shape[2], N_MODES)
    fg = ((real.reshape(-1, shape[2]) @ ev.T) @ ev).reshape(shape)
    hi = load('Fastbox_cube_cropped.npy')[:, :, :shape[2]]
    spec = (dict(table=os.path.abspath(args.beam_table)) if args.beam_table
            else dict(fwhm=args.beam_fwhm, index=args.beam_index,
                      nu_ref=args.beam_nu_ref))
    beam = beam_from_spec(spec, freqs_for(CROP), shape)
    rng = np.random.default_rng(TRUTH_SEED)
    noise = rng.normal(scale=np.sqrt(T_SYS ** 2 / (DEL_NU * DEL_T)),
                       size=shape)

    # Ground spill (optional) enters after the beam: it is far-sidelobe
    # pickup, not sky, so the model's g block sits outside B as well.
    if args.groundspill > 0:
        from imgibbs.systematics import groundspill_cube
        spill, gs = groundspill_cube(freqs_for(CROP), shape,
                                     ripple_rms=args.groundspill,
                                     spill_level=args.spill_level,
                                     period=args.gs_period,
                                     gradient=args.gs_gradient, rng=rng)
    else:
        spill, gs = np.zeros(shape), None

    os.makedirs(args.out, exist_ok=True)
    beamed = (beam.apply(hi + fg) + spill + noise) * flag
    control = (hi + fg + spill + noise) * flag
    # A flagged voxel is one that is exactly zero; make sure no real one is.
    for cube in (beamed, control):
        cube[(flag > 0) & (cube == 0)] = 1e-12
    np.save(os.path.join(args.out, 'mock_beamed.npy'), beamed)
    np.save(os.path.join(args.out, 'mock_control.npy'), control)
    np.save(os.path.join(args.out, 'truth_hi.npy'), hi)
    if gs is not None:
        np.save(os.path.join(args.out, 'truth_g.npy'), gs['g_true'])
    Path(args.out, 'beam_args.txt').write_text(' '.join(beam_run_args(spec))
                                               + '\n')
    meta = dict(beam=spec, truth_seed=TRUTH_SEED,
                n_modes_fg=N_MODES, hi_rms_K=float(hi.std()),
                fg_rms_K=float(fg[flag > 0].std()),
                groundspill=(None if gs is None else dict(
                    ripple_rms_K=args.groundspill,
                    spill_level_K=args.spill_level, period_MHz=args.gs_period,
                    gradient=args.gs_gradient,
                    run_gibbs_args=['--groundspill', '--gs-period',
                                    repr(args.gs_period)])),
                noise_rms_K=float(noise.std()))
    Path(args.out, 'mock_meta.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print(f'wrote mocks to {args.out}')


def true_power(hi, sig_k, idxs):
    """Mean |s_k|^2 per bin, in the same normalisation as the S draw."""
    s = np.fft.fftn(hi - hi.mean(), norm='ortho')
    binned, _ = bin_it(s, sig_k, idxs)
    return np.array([np.mean(np.abs(b) ** 2) for b in binned])


def summarise(args):
    hi = np.load(os.path.join(args.out, 'truth_hi.npy'))
    real = load_l2021_cube()[CROP]
    meta = json.loads(data_path('S_starting_point_cropped_meta.json').read_text())
    sig_k, idxs, _ = kbins_from_crop(real, tuple(meta['box_dims']),
                                     max_bins=N_K_BINS)
    mock = json.loads(Path(args.out, 'mock_meta.json').read_text())
    spec = mock.get('beam') or dict(table=mock['beam_table'])
    beam = beam_from_spec(spec, freqs_for(CROP), real.shape)
    vis = beam.observable_modes(args.threshold)
    p_all = true_power(hi, sig_k, idxs)
    p_vis = true_power(hi, sig_k, np.where(vis, idxs, 0))
    print('true HI power per bin, all modes    :', np.array2string(p_all, precision=3))
    print('true HI power per bin, visible modes:', np.array2string(p_vis, precision=3))
    print(f'(burn-in {args.burn} samples discarded)\n')

    rows = {}
    for cfg in sorted(glob.glob(os.path.join(args.out, '*', 'run*config.json'))):
        arm_dir = os.path.dirname(cfg)
        arm = os.path.basename(arm_dir)
        c = json.loads(Path(cfg).read_text())
        sfx = c.get('suffix') or f"_LP{c['n_modes']}_"
        traces = sorted(glob.glob(os.path.join(arm_dir, 'samples',
                                               f'Pk_trace*{sfx}.npy')),
                        key=lambda p: int(Path(p).name[8:].split('_')[0]))
        if len(traces) <= args.burn:
            print(f'{arm}: {len(traces)} samples, not past burn-in')
            continue
        pk = np.array([np.load(t) for t in traces])
        post = pk[args.burn:]
        mv = os.path.join(arm_dir, f'matvecs{sfx}.npy')
        mv = np.median(np.load(mv)) if os.path.exists(mv) else np.nan
        # Ground-spill recovery, where it was injected and modelled: the
        # posterior-mean g projected on the truth, as a fraction of it.
        g_rec = None
        g_file = os.path.join(args.out, 'truth_g.npy')
        g_traces = glob.glob(os.path.join(arm_dir, 'samples',
                                          f'g_trace*{sfx}.npy'))
        # A deflated basis is not the one g_true is expressed in, so the
        # amplitudes are not comparable; judge that arm by P(k).
        if (os.path.exists(g_file) and len(g_traces) > args.burn
                and not c.get('gs_deflate')):
            g_true = np.load(g_file).ravel()
            g_traces.sort(key=lambda p: int(Path(p).name[7:].split('_')[0]))
            g = np.array([np.load(f) for f in g_traces[args.burn:]])
            if c.get('gs_smooth'):
                # Spectral row 0 is the smooth template; g_true is the ripple.
                g = g.reshape(len(g), -1, c['gs_harmonics'] * 2 + 1)[:, :, 1:]
            g = g.reshape(len(g), -1)
            amp = g @ g_true / (g_true @ g_true)
            g_rec = (amp.mean(), amp.std())
        rows[arm] = (post.mean(0), post.std(0), pk, mv, g_rec)
        np.save(os.path.join(arm_dir, 'Pk_chain.npy'), pk)

    for arm, (mean, std, pk, mv, g_rec) in rows.items():
        print(f'{arm}: {len(pk)} samples, median {mv:.0f} matvecs/solve'
              + ('' if g_rec is None else
                 f'; ground spill recovered at {g_rec[0]:.3f}+-{g_rec[1]:.3f}'
                 f' of the injected amplitude'))
        print('   ratio to truth (all modes):     '
              + '  '.join(f'{m / t:6.3f}+-{s / t:.3f}'
                          for m, s, t in zip(mean, std, p_all)))
        print('   ratio to truth (visible modes): '
              + '  '.join(f'{m / t:6.3f}+-{s / t:.3f}'
                          for m, s, t in zip(mean, std, p_vis)))
        # Drift: posterior mean of the first and second halves after burn-in.
        h = len(pk[args.burn:]) // 2
        if h >= 20:
            first, second = pk[args.burn:args.burn + h].mean(0), pk[args.burn + h:].mean(0)
            print('   drift (2nd half / 1st half):    '
                  + '  '.join(f'{b / a:6.3f}' for a, b in zip(first, second)))


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = p.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build')
    b.add_argument('--beam-table', default=None,
                   help='measured beam, any imgibbs.load_beam_table layout '
                        '(radial table, grid transfer, profile pickle); '
                        'overrides the Gaussian')
    b.add_argument('--beam-fwhm', type=float, default=1.5,
                   help='Gaussian FWHM in degrees at --beam-nu-ref '
                        '(default 1.5)')
    b.add_argument('--beam-nu-ref', type=float, default=None,
                   help='reference frequency, MHz (default: band centre)')
    b.add_argument('--beam-index', type=float, default=1.0,
                   help='chromatic index (default 1)')
    b.add_argument('--groundspill', type=float, default=0.0,
                   help='also inject a ground-spill ripple of this rms, K, '
                        'outside the beam (default 0: none). Model it with '
                        'run_gibbs.py --groundspill.')
    b.add_argument('--spill-level', type=float, default=0.5,
                   help='rms of the smooth spill component, K (default 0.5)')
    b.add_argument('--gs-period', type=float, default=17.5,
                   help='ripple period, MHz (default 17.5)')
    b.add_argument('--gs-gradient', type=float, default=0.3,
                   help='scan-direction gradient relative to the constant '
                        '(default 0.3)')
    b.add_argument('--out', default='outputs/beam_injection')
    s = sub.add_parser('summarise')
    s.add_argument('out', nargs='?', default='outputs/beam_injection')
    s.add_argument('--burn', type=int, default=100)
    s.add_argument('--threshold', type=float, default=1e-3)
    args = p.parse_args()
    {'build': build, 'summarise': summarise}[args.cmd](args)


if __name__ == '__main__':
    main()
