#!/usr/bin/env python
"""Tabulate a primary-beam model as ``B(|k_perp|, nu)`` for ``--beam-table``.

For anyone with access to a MeerKAT primary-beam model in the
``.npz`` layout used by the MeerKAT beam releases: complex Jones voltage
patterns ``beam`` of shape ``(4 pols, n_ant, n_freq, n, n)`` on a
``margin_deg`` grid, with ``freq_MHz`` and ``antnames``. Such models are not
distributed with this repository, and nothing here depends on one -- every
script runs with a Gaussian beam by default. For every channel in the band:

1. forms the Stokes-I power beam ``P = (|HH|^2 + |HV|^2 + |VH|^2 + |VV|^2)/2``;
2. averages it azimuthally -- a scanned map averages over parallactic angle,
   so the effective map beam is closer to the azimuthal average than to the
   instantaneous, slightly elliptical pattern;
3. Hankel-transforms the radial profile to a peak-normalised ``B(k)``.

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

from imgibbs import data_path, radial_transfer                 # noqa: E402


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
    p.add_argument('npz', help='MeerKAT primary-beam .npz')
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
    args = p.parse_args()

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
    r_max = min(-margin[0], margin[-1])
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


if __name__ == '__main__':
    main()
