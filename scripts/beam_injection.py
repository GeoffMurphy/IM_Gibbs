#!/usr/bin/env python
"""Known-answer test for the beam: build beamed mocks, and summarise the arms.

The comparison only means anything with an injected signal (see "The
comparison cannot work without injection" in docs/STATUS.md), so this builds
synthetic cubes on the live grid and mask::

    d = w * (B (HI + FG) + n)

HI is the simulated FastBox cube shipped in ``data/``; FG is the least-squares
Legendre model of the real L2021 cube (same as ``systematics_injection.py``),
so its amplitude and spectral shape are realistic; ``B`` is a tabulated
measured beam; the noise has the sampler's own ``T_sys``. A ``control`` mock
is the same sky and noise with no beam.

Build both mocks::

    python scripts/beam_injection.py build --beam-table data/beam_uhf_table.npz

then sample them with ``run_gibbs.py --data`` (``submit_beam_injection.sh``
runs the standard arms), then::

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
    bin_it, data_path, kbins_from_crop, load, load_beam_table,
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
    beam = load_beam_table(args.beam_table, freqs_for(CROP), shape)
    rng = np.random.default_rng(TRUTH_SEED)
    noise = rng.normal(scale=np.sqrt(T_SYS ** 2 / (DEL_NU * DEL_T)),
                       size=shape)

    os.makedirs(args.out, exist_ok=True)
    beamed = (beam.apply(hi + fg) + noise) * flag
    control = (hi + fg + noise) * flag
    # A flagged voxel is one that is exactly zero; make sure no real one is.
    for cube in (beamed, control):
        cube[(flag > 0) & (cube == 0)] = 1e-12
    np.save(os.path.join(args.out, 'mock_beamed.npy'), beamed)
    np.save(os.path.join(args.out, 'mock_control.npy'), control)
    np.save(os.path.join(args.out, 'truth_hi.npy'), hi)
    meta = dict(beam_table=str(args.beam_table), truth_seed=TRUTH_SEED,
                n_modes_fg=N_MODES, hi_rms_K=float(hi.std()),
                fg_rms_K=float(fg[flag > 0].std()),
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
    beam = load_beam_table(json.loads(Path(args.out, 'mock_meta.json')
                                      .read_text())['beam_table'],
                           freqs_for(CROP), real.shape)
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
        rows[arm] = (post.mean(0), post.std(0), len(pk), mv)
        np.save(os.path.join(arm_dir, 'Pk_chain.npy'), pk)

    for arm, (mean, std, n, mv) in rows.items():
        print(f'{arm}: {n} samples, median {mv:.0f} matvecs/solve')
        print('   ratio to truth (all modes): '
              + '  '.join(f'{m / t:6.3f}+-{s / t:.3f}'
                          for m, s, t in zip(mean, std, p_all)))


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = p.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build')
    b.add_argument('--beam-table', default=str(data_path('beam_uhf_table.npz')))
    b.add_argument('--out', default='outputs/beam_injection')
    s = sub.add_parser('summarise')
    s.add_argument('out', nargs='?', default='outputs/beam_injection')
    s.add_argument('--burn', type=int, default=100)
    s.add_argument('--threshold', type=float, default=1e-3)
    args = p.parse_args()
    {'build': build, 'summarise': summarise}[args.cmd](args)


if __name__ == '__main__':
    main()
