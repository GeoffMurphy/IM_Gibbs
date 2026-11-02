"""A chromatic primary beam for the forward model.

With a beam the data model becomes::

    d = w * (B Us s + B Uf f + Ug g) + n

``B`` acts on the sky -- the 21cm signal and the foreground -- and on nothing
else. Ground spill and 1/f enter through the far sidelobes and the receiver,
after the main beam, so the systematics block sits outside it; polarisation
leakage is itself a beam effect and keeps its own spatial templates rather
than being convolved again. See "No beam in the MODEL" in ``docs/STATUS.md``.

What changes the physics is the chromaticity, not the smoothing
-----------------------------------------------------------------
The FWHM scales roughly as ``1/nu``, so across 970-1023 MHz it changes by ~5%.
A frequency-dependent beam turns spatial structure into spectral structure:
a foreground that is exactly smooth in frequency in every pixel acquires
spectral wiggles proportional to its own spatial gradients. That is the
assumption every absorbed-fraction number in :mod:`imgibbs.systematics`
rests on, so those numbers need re-measuring with a beam in place.
:func:`chromatic_leakage` measures the size of the effect on a given cube.

The operator
------------
``B`` is diagonal in ``(k_perp, nu)``: transform the two transverse axes,
multiply each channel by its own transfer function, transform back. It is a
real symmetric circulant, so ``B^T = B`` and the same call serves as the
adjoint. It is **not** diagonal in the 3D Fourier basis ``Us`` works in -- the
frequency dependence mixes ``k_parallel`` -- which is why the preconditioner
uses a band average for the signal block (see :meth:`Beam.signal_diagonal`).

Boundaries are periodic, matching the periodic box the signal ``s`` is defined
on. The footprint sits inside a bounding box with flagged edges, so wrapped
power lands on ``w = 0`` voxels for the most part; it is not zero-padded.

The beam is normalised to ``B(k_perp = 0) = 1``: it conserves the mean
brightness temperature in each channel, which is the convention a map in K
convolved with a peak-normalised beam follows.
"""

from __future__ import annotations

import numpy as np
import numpy.fft as fft

from .grid import PIX_DEG

#: FWHM = FWHM_TO_SIGMA * sigma for a Gaussian.
FWHM_TO_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


class Beam:
    """A beam given as a transfer function per channel on the transverse grid.

    Parameters
    ----------
    transfer : array, shape ``(Nx, Ny // 2 + 1, Nz)``
        ``B(k_perp, nu)`` on the half-plane ``rfftn(..., axes=(0, 1))``
        produces. Real and even in ``k_perp``, so the operator is symmetric.
    shape : tuple
        ``(Nx, Ny, Nz)`` of the cube the beam acts on.

    Build one with :func:`gaussian_beam` rather than by hand.
    """

    def __init__(self, transfer, shape):
        self.shape = tuple(int(n) for n in shape)
        nx, ny, nz = self.shape
        transfer = np.asarray(transfer, dtype=float)
        if transfer.shape != (nx, ny // 2 + 1, nz):
            raise ValueError(f'transfer has shape {transfer.shape}, expected '
                             f'{(nx, ny // 2 + 1, nz)} for a {self.shape} cube')
        self.transfer = transfer

    def apply(self, cube):
        """Convolve a real-space cube with the beam, channel by channel.

        Accepts the cube as ``shape`` or flattened, and returns the same
        layout it was given. ``B`` is symmetric, so this is also ``B^T``.
        """
        flat = np.ndim(cube) == 1
        c = np.reshape(cube, self.shape)
        ck = fft.rfftn(c, axes=(0, 1))
        out = fft.irfftn(ck * self.transfer, s=self.shape[:2], axes=(0, 1))
        return out.ravel() if flat else out

    adjoint = apply

    def full_transfer(self):
        """``B(k_perp, nu)`` on the full ``(Nx, Ny, Nz)`` transverse grid."""
        nx, ny, nz = self.shape
        full = np.empty(self.shape)
        nh = ny // 2 + 1
        full[:, :nh] = self.transfer
        # Mirror the half-plane: B(-kx, -ky) = B(kx, ky).
        ky_neg = np.arange(nh, ny)
        full[:, ky_neg] = self.transfer[(-np.arange(nx)) % nx][:, ny - ky_neg]
        return full

    def observable_modes(self, threshold):
        """Which 3D Fourier modes the beam leaves visible to the data.

        ``True`` where the band-averaged ``|B(k_perp)|^2`` is at least
        ``threshold``, on the full ``fftn`` grid, flattened -- the layout of
        ``idxs`` from :func:`imgibbs.kbins_from_crop`.

        Used to collapse the ``S`` step: modes the beam has removed carry no
        information about ``S``, so ``S`` is drawn from the visible modes
        only and the rest are marginalised. See ``docs/STATUS.md``, "The beam
        in the MODEL".
        """
        b2 = (self.full_transfer() ** 2).mean(axis=2)
        return np.broadcast_to((b2 >= threshold)[:, :, None],
                               self.shape).ravel()

    def signal_diagonal(self, rfft_shape):
        """Band-averaged ``|B(k_perp)|^2``, broadcast onto the ``Us`` layout.

        The diagonal of ``Us^T B^2 Us`` in the 3D Fourier basis is *exactly*
        the band mean of ``B(k_perp, nu)^2`` -- multiplying by a function of
        ``nu`` is a convolution in ``k_parallel``, and the zero-lag term of
        that convolution is the mean. The off-diagonal ``k_parallel`` mixing
        is what the preconditioner drops.
        """
        b2 = (self.full_transfer() ** 2).mean(axis=2)           # (Nx, Ny)
        return np.broadcast_to(b2[:, :, None], rfft_shape).ravel()


def gaussian_beam(freqs, shape, fwhm_ref, nu_ref, index=1.0,
                  pix_deg=PIX_DEG):
    """A circular Gaussian beam with ``FWHM(nu) = fwhm_ref * (nu_ref/nu)^index``.

    Parameters
    ----------
    freqs : array, length Nz
        Channel frequencies, MHz.
    shape : tuple
        ``(Nx, Ny, Nz)``.
    fwhm_ref : float
        FWHM in degrees at ``nu_ref``. **Take this from a measured MeerKAT
        beam, not from** ``1.22 lambda/D`` -- that formula assumes uniform
        illumination and is not the right width for a tapered feed.
    nu_ref : float
        Reference frequency for ``fwhm_ref``, MHz.
    index : float
        Chromatic index. ``1`` is diffraction scaling (FWHM ~ lambda);
        ``0`` gives an achromatic beam, which smooths but cannot mix
        spatial structure into frequency -- useful as a control.
    pix_deg : float or (float, float)
        Map pixel size in degrees along axes 0 and 1. Flat sky.

    Returns
    -------
    Beam
    """
    nx, ny, nz = (int(n) for n in shape)
    freqs = np.asarray(freqs, dtype=float)
    if freqs.shape != (nz,):
        raise ValueError(f'{freqs.size} frequencies for {nz} channels')
    px, py = np.broadcast_to(np.asarray(pix_deg, dtype=float), (2,))

    sigma_deg = beam_fwhm(freqs, fwhm_ref, nu_ref, index) / FWHM_TO_SIGMA
    # Angular wavenumbers in rad/deg on the rfftn(axes=(0, 1)) half-plane.
    kx = 2 * np.pi * fft.fftfreq(nx, d=px)
    ky = 2 * np.pi * fft.rfftfreq(ny, d=py)
    k2 = kx[:, None] ** 2 + ky[None, :] ** 2                   # (Nx, Ny//2+1)
    transfer = np.exp(-0.5 * k2[:, :, None] * sigma_deg[None, None, :] ** 2)
    return Beam(transfer, shape)


def radial_transfer(r_deg, profile, k):
    """Peak-normalised transfer function of a circular beam, by Hankel transform.

    ``B(k) = int P(r) J0(k r) r dr / int P(r) r dr``, so ``B(0) = 1``.

    Parameters
    ----------
    r_deg : array
        Radii, degrees, increasing from (near) zero.
    profile : array, ``(..., len(r_deg))``
        Power beam ``P(r)``; any normalisation.
    k : array
        Angular wavenumbers, rad/deg.

    Returns
    -------
    array, ``(..., len(k))``
    """
    from scipy.special import j0
    r = np.asarray(r_deg, dtype=float)
    kernel = j0(np.outer(k, r)) * r                         # (nk, nr)
    num = np.trapezoid(profile[..., None, :] * kernel, r, axis=-1)
    den = np.trapezoid(profile * r, r, axis=-1)[..., None]
    return num / den


def tabulated_beam(k_table, nu_table, transfer_table, freqs, shape,
                   pix_deg=PIX_DEG, width_scale=1.0):
    """A circular beam from a tabulated ``B(|k_perp|, nu)``.

    Linear interpolation in frequency between tabulated channels, then in
    ``|k_perp|`` onto the grid. This is how a *measured* beam enters the
    model, ripple and all -- see ``scripts/build_beam_table.py``.

    Parameters
    ----------
    k_table : array, ``(nk,)``
        Angular wavenumbers, rad/deg, increasing, covering the grid's
        ``|k_perp|`` (checked).
    nu_table : array, ``(nnu,)``
        Tabulated frequencies, MHz, increasing, covering ``freqs`` (checked).
    transfer_table : array, ``(nnu, nk)``
    freqs, shape, pix_deg :
        As for :func:`gaussian_beam`.
    width_scale : float
        Stretch the beam by this factor in angle (1.02 is 2% wider), i.e.
        evaluate ``B(width_scale * |k_perp|)``. For testing how wrong a
        model beam can be.
    """
    nx, ny, nz = (int(n) for n in shape)
    freqs = np.asarray(freqs, dtype=float)
    k_table = np.asarray(k_table, dtype=float)
    nu_table = np.asarray(nu_table, dtype=float)
    transfer_table = np.asarray(transfer_table, dtype=float)
    if freqs.shape != (nz,):
        raise ValueError(f'{freqs.size} frequencies for {nz} channels')
    if freqs.min() < nu_table[0] or freqs.max() > nu_table[-1]:
        raise ValueError(f'beam table covers {nu_table[0]:.2f}-'
                         f'{nu_table[-1]:.2f} MHz, grid needs '
                         f'{freqs.min():.2f}-{freqs.max():.2f}')
    px, py = np.broadcast_to(np.asarray(pix_deg, dtype=float), (2,))
    kx = 2 * np.pi * fft.fftfreq(nx, d=px)
    ky = 2 * np.pi * fft.rfftfreq(ny, d=py)
    kperp = width_scale * np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)
    if kperp.max() > k_table[-1]:
        raise ValueError(f'beam table reaches k = {k_table[-1]:.2f} rad/deg, '
                         f'grid needs {kperp.max():.2f}')

    # Frequency first: (nz, nk), then onto |k_perp| channel by channel.
    per_chan = np.stack([np.interp(freqs, nu_table, transfer_table[:, i])
                         for i in range(len(k_table))], axis=1)
    transfer = np.stack([np.interp(kperp, k_table, per_chan[c])
                         for c in range(nz)], axis=-1)
    return Beam(transfer, shape)


def grid_beam(transfer, file_freqs, freqs, shape):
    """A beam already tabulated on the map's own transverse grid.

    ``transfer`` is ``B(kx, ky, nu)`` on the full ``(Nx, Ny)`` ``fftfreq``
    grid, one plane per tabulated frequency -- e.g. a 2D (non-circular)
    beam pattern transformed by ``scripts/build_beam_grid.py``. It must be
    real and even in ``k_perp`` (a point-symmetric power pattern). Planes are
    interpolated linearly in frequency onto ``freqs`` unless they already
    match.
    """
    nx, ny, nz = (int(n) for n in shape)
    full = np.asarray(transfer, dtype=float)
    freqs = np.asarray(freqs, dtype=float)
    if full.ndim != 3 or full.shape[:2] != (nx, ny):
        raise ValueError(f'grid beam has shape {full.shape}, map needs '
                         f'({nx}, {ny}, n_freq)')
    if file_freqs is None:
        if full.shape[2] != nz:
            raise ValueError(f'grid beam has {full.shape[2]} planes and no '
                             f'frequencies; map has {nz} channels')
    else:
        file_freqs = np.asarray(file_freqs, dtype=float)
        if not (file_freqs.shape == freqs.shape
                and np.allclose(file_freqs, freqs, atol=1e-6)):
            if freqs.min() < file_freqs[0] or freqs.max() > file_freqs[-1]:
                raise ValueError(f'grid beam covers {file_freqs[0]:.2f}-'
                                 f'{file_freqs[-1]:.2f} MHz, map needs '
                                 f'{freqs.min():.2f}-{freqs.max():.2f}')
            idx = np.clip(np.searchsorted(file_freqs, freqs) - 1, 0,
                          len(file_freqs) - 2)
            a = ((freqs - file_freqs[idx])
                 / (file_freqs[idx + 1] - file_freqs[idx]))
            full = full[:, :, idx] * (1 - a) + full[:, :, idx + 1] * a
    return Beam(full[:, :ny // 2 + 1], shape)


def read_beam_profiles(path, rmax=6.0):
    """Azimuthally averaged power profiles from a pickle.

    The layout is a dict with ``freq`` (MHz), ``th`` (deg) and power
    profiles ``P_v_th``, ``P_h_th`` of shape ``(n_freq, n_th)``; the
    intensity beam is their mean. Returns ``freq, r_deg, profiles`` with
    each profile peak-normalised and cut at ``rmax`` degrees -- power further
    out is pickup from outside the field, not a transfer on in-map
    structure.
    """
    import pickle
    with open(path, 'rb') as fh:
        d = pickle.load(fh, encoding='latin1')
    freq = np.asarray(d['freq'], dtype=float).ravel()
    th = np.asarray(d['th'], dtype=float).ravel()
    prof = 0.5 * (np.asarray(d['P_v_th']) + np.asarray(d['P_h_th']))
    keep = th <= rmax
    prof = prof[:, keep]
    return freq, th[keep], prof / prof[:, :1]


def _profile_fwhm(r, profile):
    """FWHM in degrees of a peak-normalised radial profile."""
    cut = np.argmax(profile < 0.3)
    return 2 * float(np.interp(0.5, profile[:cut][::-1], r[:cut][::-1]))


def load_beam_table(path, freqs, shape, pix_deg=PIX_DEG, width_scale=1.0):
    """Build a :class:`Beam` from a beam file, whichever of three layouts:

    * a **radial table** ``.npz`` (``k``, ``freq_MHz``, ``transfer``), as
      written by ``scripts/build_beam_table.py`` -- a circular beam;
    * a **full-grid** ``.npz`` (``beam`` of shape ``(Nx, Ny, n_freq)``,
      optionally ``freqs`` or ``freq_MHz``), the transfer already on this
      map's grid -- any point-symmetric beam (see :func:`grid_beam`);
    * a **profile pickle** of azimuthally averaged power profiles (see
      :func:`read_beam_profiles`), Hankel-transformed here.

    A full Jones beam model is not read here -- it can be several GB --
    reduce it with ``scripts/build_beam_table.py`` first. The returned beam
    has a ``source`` attribute describing what was loaded.
    """
    import zipfile
    from pathlib import Path
    name = Path(path).name
    freqs = np.asarray(freqs, dtype=float)
    if zipfile.is_zipfile(path):
        with np.load(path) as t:
            files = set(t.files)
            if {'k', 'freq_MHz', 'transfer'} <= files:
                beam = tabulated_beam(t['k'], t['freq_MHz'], t['transfer'],
                                      freqs, shape, pix_deg,
                                      width_scale=width_scale)
                src = str(t['source']) if 'source' in files else name
                desc = f'radial table ({src})'
                if 'fwhm_deg' in files:
                    fw = width_scale * np.interp(freqs[[0, -1]],
                                                 t['freq_MHz'], t['fwhm_deg'])
                    desc += f', FWHM {fw[0]:.3f} -> {fw[1]:.3f} deg'
            elif 'margin_deg' in files or 'antnames' in files:
                raise ValueError(f'{name} is a full Jones beam model; reduce '
                                 f'it with scripts/build_beam_table.py first')
            elif 'beam' in files:
                if width_scale != 1.0:
                    raise ValueError('--beam-width-scale needs a radial '
                                     'table or profiles, not a grid beam')
                ff = next((t[k] for k in ('freqs', 'freq_MHz') if k in files),
                          None)
                beam = grid_beam(t['beam'], ff, freqs, shape)
                desc = f'grid transfer ({name})'
            else:
                raise ValueError(f'{name}: not a beam table, grid beam or '
                                 f'profile pickle (keys {sorted(files)})')
    else:
        fr, r, prof = read_beam_profiles(path)
        lo = max(np.searchsorted(fr, freqs.min(), side='right') - 1, 0)
        hi = min(np.searchsorted(fr, freqs.max(), side='left') + 1, len(fr))
        fr, prof = fr[lo:hi], prof[lo:hi]
        px, py = np.broadcast_to(np.asarray(pix_deg, dtype=float), (2,))
        kmax = width_scale * 2 * np.pi * np.hypot(0.5 / px, 0.5 / py)
        k = np.linspace(0.0, 1.05 * kmax, 1001)
        beam = tabulated_beam(k, fr, radial_transfer(r, prof, k), freqs,
                              shape, pix_deg, width_scale=width_scale)
        fw = width_scale * np.interp(freqs[[0, -1]], fr,
                                     [_profile_fwhm(r, q) for q in prof])
        desc = (f'azimuthal profiles ({name}), FWHM {fw[0]:.3f} -> '
                f'{fw[1]:.3f} deg')
    if width_scale != 1.0:
        desc += f', width x {width_scale:g}'
    beam.source = desc
    return beam


def beam_from_spec(spec, freqs, shape, pix_deg=PIX_DEG):
    """Build a beam from a small dict, so scripts can record and rebuild it.

    ``{'table': path}`` loads a tabulated beam (:func:`load_beam_table`);
    ``{'fwhm': deg, 'nu_ref': MHz, 'index': float}`` builds a Gaussian
    (``nu_ref`` defaults to the band centre, ``index`` to 1).
    """
    if spec.get('table'):
        return load_beam_table(spec['table'], freqs, shape, pix_deg)
    nu_ref = spec.get('nu_ref') or float(np.mean(freqs))
    return gaussian_beam(freqs, shape, spec['fwhm'], nu_ref,
                         index=spec.get('index', 1.0), pix_deg=pix_deg)


def beam_run_args(spec):
    """The ``run_gibbs.py`` flags that select the beam ``spec`` describes."""
    if spec.get('table'):
        return ['--beam-table', str(spec['table'])]
    out = ['--beam-fwhm', repr(float(spec['fwhm'])),
           '--beam-index', repr(float(spec.get('index', 1.0)))]
    if spec.get('nu_ref'):
        out += ['--beam-nu-ref', repr(float(spec['nu_ref']))]
    return out


def beam_fwhm(freqs, fwhm_ref, nu_ref, index=1.0):
    """``FWHM(nu)`` in degrees for the power-law scaling ``gaussian_beam`` uses."""
    return fwhm_ref * (nu_ref / np.asarray(freqs, dtype=float)) ** index


def chromatic_leakage(cube, beam, evecs, w=None):
    """How much spectral structure the beam adds that the foreground misses.

    Convolves ``cube`` with ``beam`` and returns the residual after projecting
    out the foreground basis ``evecs`` pixel by pixel -- the part of a smooth
    foreground that a chromatic beam pushes outside the span ``Uf`` can
    absorb. Compare its rms with the 21cm signal's.

    The same projection is applied to the unconvolved cube, and that residual
    is subtracted, so a cube that was not exactly in the span to begin with
    does not count against the beam.

    Parameters
    ----------
    cube : array, shape ``beam.shape``
        Usually a foreground fit -- a cube inside the span of ``evecs``.
    beam : Beam
    evecs : array, shape ``(n_modes, Nz)``, orthonormal rows
    w : array, optional
        Weights, ``shape`` or flat. Residual rms is taken over ``w > 0``.

    Returns
    -------
    residual : array, ``beam.shape``
        Beam-induced residual outside the foreground span.
    rms : float
        Its rms over the weighted voxels, in the units of ``cube``.
    """
    evecs = np.asarray(evecs)
    if not np.allclose(evecs @ evecs.T, np.eye(evecs.shape[0]), atol=1e-8):
        raise ValueError('evecs must have orthonormal rows')

    def outside(c):
        return c - (c @ evecs.T) @ evecs

    cube = np.reshape(cube, beam.shape)
    residual = outside(beam.apply(cube)) - beam.apply(outside(cube))
    mask = (np.ones(beam.shape, bool) if w is None
            else np.reshape(w, beam.shape) > 0)
    return residual, float(np.sqrt(np.mean(residual[mask] ** 2)))
