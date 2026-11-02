#!/usr/bin/env python
"""Tabulate a primary-beam model as ``B(|k_perp|, nu)`` for ``--beam-table``.

For anyone with access to a MeerKAT primary-beam model in one of two
layouts. Such models are not distributed with this repository, and nothing
here depends on one -- every script runs with a Gaussian beam by default.

* **Jones ``.npz``** (the MeerKAT beam releases): complex voltage patterns
  ``beam`` of shape ``(4 pols, n_ant, n_freq, n, n)`` on a ``margin_deg``
  grid, with ``freq_MHz`` and ``antnames``. For every channel in the band it
  forms the Stokes-I power beam ``P = (|HH|^2 + |HV|^2 + |VH|^2 + |VV|^2)/2``
  and averages it azimuthally -- a scanned map averages over parallactic
  angle, so the effective map beam is closer to the azimuthal average than
  to the instantaneous, slightly elliptical pattern.
* **1-D profile pickle**: a dict with ``freq`` (MHz), ``th`` (deg) and
  azimuthally averaged power profiles ``P_v_th``, ``P_h_th`` of shape
  ``(n_freq, n_th)``; the intensity beam is their mean.

Either way the radial profile, cut at ``--rmax``, is Hankel-transformed to a
peak-normalised ``B(k)``.

The output (``k``, ``freq_MHz``, ``transfer``, plus profiles and FWHMs) is
derived from the input model, so it is gitignored under ``data/`` and should
be treated with the same access restrictions::

    python scripts/build_beam_table.py BEAM_MODEL.npz --out data/my_beam_table.npz
    python scripts/run_gibbs.py 6 --beam-table data/my_beam_table.npz

The pattern stops at the edge of the model's grid, so far sidelobes beyond
that are not in ``B``. That is the right split here anyway: far-sidelobe
pickup is the ground-spill systematic, which the model keeps outside the
beam.
"""

from __future__ import annotations

import argparse
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imgibbs import data_path, radial_transfer, read_beam_profiles  # noqa: E402
from imgibbs.beam import _profile_fwhm                         # noqa: E402


def open_beam(path):
    """Memory-map ``beam.npy`` inside an *uncompressed* ``.npz``.

    ``np.load`` would read all of it (several GB for a full-band model);
    the band needed here is a small fraction of the channels.
    """
    z = zipfile.ZipFile(path)
    info = z.getinfo('beam.npy')
    if info.compress_type != zipfile.ZIP_STORED:
        raise SystemExit('beam.npy is compressed; cannot memory-map it')
    with open(path, 'rb') as f:
        f.seek(info.header_offset)
        n, m = struct.unpack('<HH', f.read(30)[26:30])
        f.seek(info.header_offset + 30 + n + m)
        version = np.lib.format.read_magic(f)
        read = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                else np.lib.format.read_array_header_2_0)
        shape, fortran, dtype = read(f)
        offset = f.tell()
    if fortran:
        raise SystemExit('beam.npy is Fortran-ordered; not handled')
    return np.memmap(path, dtype=dtype, mode='r', offset=offset, shape=shape)


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('npz', help='MeerKAT primary-beam model: Jones .npz or '
                                 '1-D profile pickle')
    p.add_argument('--out', default=str(data_path('beam_table.npz')))
    p.add_argument('--fmin', type=float, default=960.0,
                   help='lowest frequency to tabulate, MHz (default 960)')
    p.add_argument('--fmax', type=float, default=1035.0,
                   help='highest frequency to tabulate, MHz (default 1035)')
    p.add_argument('--ant', type=int, default=0,
                   help='antenna index (default 0; the array average)')
    p.add_argument('--kmax', type=float, default=25.0,
                   help='largest k tabulated, rad/deg (default 25; the '
                        '0.3 deg grid needs 14.8)')
    p.add_argument('--nk', type=int, default=1001)
    p.add_argument('--rmax', type=float, default=6.0,
                   help='cut the radial profile here, deg (default 6, the '
                        'Jones grid edge). Power further out is pickup from '
                        'outside the field, not a transfer on in-map '
                        'structure.')
    args = p.parse_args()
    if not zipfile.is_zipfile(args.npz):
        return main_profiles(args)

    meta = np.load(args.npz)
    freq = meta['freq_MHz']
    margin = meta['margin_deg']
    ant = meta['antnames'][args.ant].decode()
    J = open_beam(args.npz)
    chans = np.where((freq >= args.fmin) & (freq <= args.fmax))[0]
    if len(chans) == 0:
        raise SystemExit(f'no channels in {args.fmin}-{args.fmax} MHz')
    print(f'{Path(args.npz).name}: {J.shape} {J.dtype}, antenna {ant!r}, '
          f'{len(chans)} channels {freq[chans[0]]:.3f}-{freq[chans[-1]]:.3f} '
          f'MHz')

    # Radial bins one grid step wide, out to the inscribed circle. The bin's
    # mean radius, not its centre, is used as its abscissa.
    dr = margin[1] - margin[0]
    X, Y = np.meshgrid(margin, margin, indexing='ij')
    R = np.hypot(X, Y).ravel()
    r_max = min(-margin[0], margin[-1], args.rmax)
    edges = np.arange(0.0, r_max + dr / 2, dr)
    idx = np.digitize(R, edges) - 1
    keep = (idx >= 0) & (idx < len(edges) - 1)
    counts = np.bincount(idx[keep], minlength=len(edges) - 1)
    r_mean = np.bincount(idx[keep], R[keep], len(edges) - 1) / counts

    k = np.linspace(0.0, args.kmax, args.nk)
    profiles, fwhm = [], []
    for c in chans:
        P = 0.5 * sum(np.abs(np.asarray(J[q, args.ant, c])) ** 2
                      for q in range(J.shape[0]))
        P = P.ravel()
        prof = np.bincount(idx[keep], P[keep], len(edges) - 1) / counts
        prof = prof / prof[0]
        profiles.append(prof)
        fwhm.append(2 * np.interp(0.5, prof[:np.argmax(prof < 0.3)][::-1],
                                  r_mean[:np.argmax(prof < 0.3)][::-1]))
    profiles = np.array(profiles)
    transfer = radial_transfer(r_mean, profiles, k)

    np.savez(args.out, k=k, freq_MHz=freq[chans], transfer=transfer,
             r_deg=r_mean, profile=profiles, fwhm_deg=np.array(fwhm),
             source=str(Path(args.npz).name), antenna=ant)
    fw = np.array(fwhm)
    print(f'FWHM {fw[0]:.4f} deg at {freq[chans[0]]:.2f} MHz -> {fw[-1]:.4f} '
          f'deg at {freq[chans[-1]]:.2f} MHz; '
          f'effective index {np.log(fw[0] / fw[-1]) / np.log(freq[chans[-1]] / freq[chans[0]]):.3f}')
    print(f'wrote {args.out}')


def main_profiles(args):
    """The 1-D profile-pickle layout (see the module docstring)."""
    freq, r, profiles = read_beam_profiles(args.npz, rmax=args.rmax)
    # Keep one tabulated channel either side of the band, so interpolation
    # onto the map's channels never extrapolates.
    lo = max(np.searchsorted(freq, args.fmin, side='right') - 1, 0)
    hi = min(np.searchsorted(freq, args.fmax, side='left') + 1, len(freq))
    fc, profiles = freq[lo:hi], profiles[lo:hi]
    print(f'{Path(args.npz).name}: {len(freq)} profiles; {len(fc)} channels '
          f'{fc[0]:.2f}-{fc[-1]:.2f} MHz, cut at {r[-1]:g} deg')
    fwhm = np.array([_profile_fwhm(r, pr) for pr in profiles])
    k = np.linspace(0.0, args.kmax, args.nk)
    transfer = radial_transfer(r, profiles, k)
    np.savez(args.out, k=k, freq_MHz=fc, transfer=transfer, r_deg=r,
             profile=profiles, fwhm_deg=fwhm,
             source=str(Path(args.npz).name), antenna='azimuthal profile')
    print(f'FWHM {fwhm[0]:.4f} deg at {fc[0]:.2f} MHz -> {fwhm[-1]:.4f} '
          f'deg at {fc[-1]:.2f} MHz; '
          f'effective index {np.log(fwhm[0] / fwhm[-1]) / np.log(fc[-1] / fc[0]):.3f}')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
