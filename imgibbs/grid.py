"""Survey geometry: the physical box behind a grid of map pixels and channels.

The mock generator, the sampler and the PCA benchmark all need the same
``(shape, box_dims, z_mid)``. They must agree: the signal covariance ``S`` is
indexed by k-bins derived from ``box_dims``, so if the two disagree the signal
prior is silently attached to the wrong wavenumbers. Deriving all of it here,
once, is what stops that drifting apart.

The survey constants are from Wang et al. (2021): the MeerKAT L band spans
856-1712 MHz over 4096 channels; the grid starts at channel 550 (the low
sub-band of the MeerKLASS 2021 L-band analysis) with 0.3 deg map pixels.
:func:`imgibbs.mock.mock_grid` is the usual way in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LINE_FREQ = 1420.405752   # MHz, rest frequency of the 21cm line
PIX_DEG = 0.3             # deg per map pixel
NU0, NU1, NCHAN = 856.0, 1712.0, 4096   # band edges (MHz) and channel count
CH0, CH1 = 550, 1050      # L-band channels a grid can draw on

#: The cosmology the simulated cubes are generated under. This mirrors
#: ``fastbox.box.default_cosmo`` so the geometry here needs no fastbox import,
#: but the two MUST agree -- the simulated P(k) is the truth curve the sampler
#: output is compared against. :func:`survey_grid` checks them when fastbox is
#: importable and raises if they have drifted apart.
DEFAULT_COSMO = dict(Omega_c=0.25, Omega_b=0.05,
                     h=0.7, n_s=0.95, sigma8=0.8,
                     transfer_function='eisenstein_hu')

@dataclass(frozen=True)
class SurveyGrid:
    """Physical geometry of one crop. Build it with :func:`survey_grid`."""

    shape: tuple[int, int, int]
    box_dims: tuple[float, float, float]   # Mpc
    chans: np.ndarray                      # L-band channel numbers
    freqs: np.ndarray                      # MHz
    z_lo: float
    z_hi: float
    z_mid: float
    D_M: float                             # Mpc, at z_mid
    dnu: float                             # MHz per channel

    @property
    def voxel(self) -> tuple[float, float, float]:
        """Voxel dimensions in Mpc. Strongly anisotropic: ~8 x 8 x 1."""
        return tuple(L / n for L, n in zip(self.box_dims, self.shape))

    def summary(self) -> str:
        Lx, Ly, Lz = self.box_dims
        vx, vy, vz = self.voxel
        return '\n'.join([
            f'shape         : {self.shape}',
            f'channel width : {self.dnu:.6f} MHz',
            f'channels      : {self.chans[0]} - {self.chans[-1]}  '
            f'({len(self.chans)} kept)',
            f'frequency     : {self.freqs[0]:.2f} - {self.freqs[-1]:.2f} MHz',
            f'redshift      : {self.z_lo:.4f} - {self.z_hi:.4f}  '
            f'(mid {self.z_mid:.4f})',
            f'D_M(z_mid)    : {self.D_M:.1f} Mpc',
            f'angular       : {self.shape[0] * PIX_DEG:.1f} x '
            f'{self.shape[1] * PIX_DEG:.1f} deg',
            f'box_dims      : ({Lx:.1f}, {Ly:.1f}, {Lz:.1f}) Mpc',
            f'voxel         : {vx:.2f} x {vy:.2f} x {vz:.2f} Mpc',
        ])


def survey_grid(crop, shape) -> SurveyGrid:
    """Work out the physical geometry of ``crop``.

    Parameters
    ----------
    crop : tuple of three slices
        Only ``crop[2]`` is read: it selects channels from ``CH0``-``CH1``.
        The transverse extent comes from ``shape``.
    shape : tuple (Nx, Ny, Nz)
        Shape of the cropped cube.

    Returns
    -------
    SurveyGrid

    Notes
    -----
    The transverse extent is the subtended angle times the comoving angular
    diameter distance at the mid redshift; the radial extent is the comoving
    depth of the band. This is a light-cone approximation -- a single ``D_M``
    stands in for the whole band, whereas 21 deg subtends 465 Mpc at z = 0.32
    and 647 Mpc at z = 0.46. Fine for setting a prior, worth revisiting for
    precision P(k).
    """
    import pyccl as ccl

    try:
        from fastbox.box import default_cosmo as _fb_cosmo
    except ImportError:
        pass                      # fastbox is only needed to generate cubes
    else:
        if dict(_fb_cosmo) != DEFAULT_COSMO:
            raise ValueError(
                'DEFAULT_COSMO has drifted from fastbox.box.default_cosmo:\n'
                f'  imgibbs : {DEFAULT_COSMO}\n'
                f'  fastbox : {dict(_fb_cosmo)}\n'
                'The simulated truth curve and the survey geometry would be '
                'built under different cosmologies.')

    dnu = (NU1 - NU0) / NCHAN
    chans = np.arange(CH0, CH1)[crop[2]]        # L-band channels retained
    freqs = NU0 + chans * dnu
    if len(freqs) != shape[2]:
        raise ValueError(
            f'frequency axis disagrees with the cropped cube: '
            f'{len(freqs)} channels from crop[2] vs shape[2]={shape[2]}')

    z = LINE_FREQ / freqs - 1.0
    z_lo, z_hi = float(z.min()), float(z.max())
    z_mid = 0.5 * (z_lo + z_hi)

    cosmo = ccl.Cosmology(**DEFAULT_COSMO)
    a = lambda zz: 1.0 / (1.0 + zz)   # noqa: E731

    # Radial: comoving distance across the band.
    Lz = abs(ccl.comoving_radial_distance(cosmo, a(z_hi))
             - ccl.comoving_radial_distance(cosmo, a(z_lo)))
    # Transverse: angle x comoving angular diameter distance at the midpoint.
    D_M = ccl.comoving_angular_distance(cosmo, a(z_mid))
    Lx = np.deg2rad(shape[0] * PIX_DEG) * D_M
    Ly = np.deg2rad(shape[1] * PIX_DEG) * D_M

    return SurveyGrid(
        shape=tuple(int(n) for n in shape),
        box_dims=(float(Lx), float(Ly), float(Lz)),
        chans=chans, freqs=freqs,
        z_lo=z_lo, z_hi=z_hi, z_mid=z_mid,
        D_M=float(D_M), dnu=dnu,
    )
