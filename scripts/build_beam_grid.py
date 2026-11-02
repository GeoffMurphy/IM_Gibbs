#!/usr/bin/env python
"""Transform a 2D primary-beam pattern onto the map's grid, for ``--beam-table``.

For anyone with a full (not azimuthally averaged) MeerKAT beam model in the
MATLAB v7.3 layout: complex Jones terms ``Jph``, ``Jpv``, ``Jqh``, ``Jqv`` of
shape ``(n_freq, n_phi, n_theta)`` on a polar grid ``ph`` x ``th`` (deg),
with ``freq`` (MHz). No such model is distributed with this repository.

``build_beam_table.py`` keeps only the azimuthal average, which is a circular
beam. This keeps the pattern's azimuthal structure -- its ellipticity -- by
evaluating

    B(k) = sum P(r) cos(k . r) dA / sum P(r) dA,   r = theta (cos phi, sin phi)

directly on the map's ``(kx, ky)`` grid, with ``P`` the Stokes-I power beam
``(|Jph|^2 + |Jpv|^2 + |Jqh|^2 + |Jqv|^2) / 2``, flat-sky, out to ``--rmax``.
The cosine keeps the point-symmetric part of the pattern (the operator must
be symmetric); the script reports how much asymmetric part is dropped.

The pattern is fixed on the sky at ``--orientation`` (phi = 0 along the
first map axis). A real scan rotates it with parallactic angle; that is what
``--rotate`` approximates, by averaging the transfer over orientations spread
uniformly across +-ROTATE degrees about ``--orientation``. ``--rotate 0`` is
the no-rotation limit, and ``--rotate 90`` covers every orientation of a
pattern with 180-degree symmetry, i.e. the azimuthal average. The output (``beam`` of shape ``(Nx, Ny, n_freq)``,
``freqs``) is derived from the input model, so keep it out of the
repository::

    python scripts/build_beam_grid.py BEAM.mat --out data/my_beam_grid.npz
    python scripts/run_gibbs.py 6 --beam-table data/my_beam_grid.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imgibbs import data_path, read_beam_profiles               # noqa: E402
from imgibbs.grid import PIX_DEG                                  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('mat', help='2D beam model, MATLAB v7.3 (HDF5)')
    p.add_argument('--out', default=str(data_path('beam_grid.npz')))
    p.add_argument('--nx', type=int, default=70)
    p.add_argument('--ny', type=int, default=45)
    p.add_argument('--pix-deg', type=float, default=PIX_DEG)
    p.add_argument('--fmin', type=float, default=960.0)
    p.add_argument('--fmax', type=float, default=1036.0)
    p.add_argument('--rmax', type=float, default=6.0,
                   help='cut the pattern at this radius, deg (default 6)')
    p.add_argument('--orientation', type=float, default=0.0,
                   help='rotate the pattern by this angle on the sky, deg')
    p.add_argument('--rotate', type=float, default=0.0,
                   help='average over orientations within +-ROTATE deg, to '
                        'mimic parallactic rotation during a scan (default 0)')
    p.add_argument('--n-rotate', type=int, default=13,
                   help='orientations sampled when --rotate > 0 (default 13)')
    p.add_argument('--check-profiles', default=None,
                   help='a pickle of azimuthal profiles of the same beam; '
                        'compare its profiles with this pattern\'s average')
    args = p.parse_args()

    import h5py
    with h5py.File(args.mat, 'r') as h:
        freq = np.asarray(h['freq']).ravel()
        ph = np.asarray(h['ph']).ravel()
        th = np.asarray(h['th']).ravel()
        lo = max(np.searchsorted(freq, args.fmin, side='right') - 1, 0)
        hi = min(np.searchsorted(freq, args.fmax, side='left') + 1, len(freq))
        nth = int(np.searchsorted(th, args.rmax, side='right'))
        # A closed azimuth grid repeats phi = 0 at 360; drop the repeat.
        nph = len(ph) - 1 if np.isclose(ph[-1] - ph[0], 360) else len(ph)
        P = np.zeros((hi - lo, nph, nth))
        for name in ('Jph', 'Jpv', 'Jqh', 'Jqv'):
            J = h[name][lo:hi, :nph, :nth]
            P += J['real'] ** 2 + J['imag'] ** 2
        P *= 0.5
    freq, th, ph = freq[lo:hi], th[:nth], ph[:nph]
    print(f'{Path(args.mat).name}: {len(freq)} channels {freq[0]:.2f}-'
          f'{freq[-1]:.2f} MHz, {nph} azimuths, {nth} radii to {th[-1]:g} deg')

    # Quadrature weights: trapezoid in theta times the polar area element.
    dth = np.gradient(th)
    wt = th * dth
    wt[[0, -1]] *= 0.5
    dph = np.deg2rad(360.0 / nph)
    # Midpoints of n equal slices of [-R, R]: at R = 90 the two ends are the
    # same orientation, so sampling both would count it twice.
    n = args.n_rotate if args.rotate > 0 else 1
    angles = (-args.rotate + (np.arange(n) + 0.5) * 2 * args.rotate / n
              + args.orientation)
    # Each orientation is the same pattern with phi shifted; stacking them as
    # extra quadrature points, each weighted 1/n, averages the transfer.
    phi = np.deg2rad(ph[None, :] + angles[:, None]).ravel()
    x = (th[None, :] * np.cos(phi)[:, None]).ravel()             # deg
    y = (th[None, :] * np.sin(phi)[:, None]).ravel()
    w = np.tile(np.broadcast_to(wt[None, :] * dph, (nph, nth)).ravel(),
                len(angles)) / len(angles)

    kx = 2 * np.pi * np.fft.fftfreq(args.nx, d=args.pix_deg)     # rad/deg
    ky = 2 * np.pi * np.fft.fftfreq(args.ny, d=args.pix_deg)
    KX, KY = (a.ravel() for a in np.meshgrid(kx, ky, indexing='ij'))

    beam = np.empty((args.nx, args.ny, len(freq)))
    asym = 0.0
    for c in range(len(freq)):
        pw = np.tile(P[c].ravel(), len(angles)) * w
        re = np.empty(KX.size)
        im = np.empty(KX.size)
        for s in range(0, KX.size, 256):
            phase = np.outer(KX[s:s + 256], x) + np.outer(KY[s:s + 256], y)
            re[s:s + 256] = np.cos(phase) @ pw
            im[s:s + 256] = np.sin(phase) @ pw
        norm = pw.sum()
        beam[:, :, c] = (re / norm).reshape(args.nx, args.ny)
        asym = max(asym, np.abs(im).max() / norm)

    # FWHM along the two map axes, from the pattern itself, for the record.
    prof = P / P[:, :1, :1]
    ax0 = 0.5 * (prof[:, 0] + prof[:, nph // 2])
    ax1 = 0.5 * (prof[:, nph // 4] + prof[:, 3 * nph // 4])

    def fwhm(pr):
        cut = np.argmax(pr < 0.3)
        return 2 * np.interp(0.5, pr[:cut][::-1], th[:cut][::-1])

    print(f'FWHM along phi = 0 / 90 deg: {fwhm(ax0[0]):.4f} / '
          f'{fwhm(ax1[0]):.4f} deg at {freq[0]:.2f} MHz, '
          f'{fwhm(ax0[-1]):.4f} / {fwhm(ax1[-1]):.4f} at {freq[-1]:.2f}')
    print(f'asymmetric part dropped: up to {asym:.2e} of B(0)')

    if args.check_profiles:
        fr, r, pr = read_beam_profiles(args.check_profiles, rmax=args.rmax)
        az = prof.mean(axis=1)
        diffs = []
        for c, f in enumerate(freq):
            i = int(np.argmin(np.abs(fr - f)))
            if abs(fr[i] - f) < 1e-6:
                diffs.append(np.abs(np.interp(th, r, pr[i]) - az[c]).max())
        print(f'azimuthal average vs {Path(args.check_profiles).name}: max '
              f'|difference| {max(diffs):.2e} over {len(diffs)} channels')

    np.savez(args.out, beam=beam, freqs=freq, rmax=args.rmax,
             orientation=args.orientation, rotate=args.rotate,
             pix_deg=args.pix_deg,
             source=str(Path(args.mat).name))
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
