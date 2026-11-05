"""Mock observations: everything the sampler runs on, from public inputs only.

The data model the sampler fits is::

    d = w * (B (HI + FG) + systematic + n)

and this module builds every term of it:

* **HI** -- a FastBox lognormal realisation with linear redshift-space
  distortions (:func:`fastbox.tracers.generate_hi_mock`), in K.
* **FG** -- FastBox's :class:`~fastbox.foregrounds.ForegroundModel`: Galactic
  synchrotron and extragalactic point sources as Gaussian random amplitude
  maps with the Santos et al. (2005) angular power spectra, each scaled to
  the band by a spatially varying power law. Plus the CMB monopole.
* **B** -- a chromatic Gaussian primary beam (:func:`imgibbs.gaussian_beam`).
  It acts on the sky and on nothing else.
* **systematic** -- ground spill, 1/f or polarisation leakage, from the
  equations in :mod:`imgibbs.systematics`. Not beamed: they enter through
  the far sidelobes and the receiver.
* **n** -- white radiometer noise, ``T_sys / sqrt(dnu dt)`` per voxel.
* **w** -- a drift-scan footprint: a tilted band across the map with rounded
  ends, filling ~60% of the bounding box.

Every component has its own seed, all spawned from one master seed, so
changing the systematic (say) leaves the HI, foreground and noise
realisations untouched. FastBox draws from NumPy's global generator, so the
HI and foreground draws seed it and then restore it -- building a mock never
moves the sampler's own random stream.

The grid is the MeerKAT L band (856-1712 MHz over 4096 channels), channels
550 onward at 0.3 deg pixels -- see :mod:`imgibbs.grid`. The default
(70, 45, 250) is the grid every result in ``docs/STATUS.md`` was measured on.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np

from .beam import gaussian_beam
from .grid import NU0, NU1, NCHAN, PIX_DEG, survey_grid
from .systematics import (
    CompositeBasis, groundspill_cube, leakage_basis, onef_basis, realise,
)

#: The working grid: (RA, Dec, frequency) pixels.
MOCK_SHAPE = (70, 45, 250)

#: Radiometer noise, Wang et al. (2021) Table 1 for the L band, with the true
#: channel width and a nominal 1000 s per voxel.
T_SYS = 16.0                                # K
DEL_T = 1000.0                              # s
DEL_NU = (NU1 - NU0) / NCHAN * 1e6          # Hz, 0.208984 MHz

#: Gaussian beam FWHM in degrees at the band centre -- about the width of the
#: MeerKAT L-band main lobe near 1 GHz.
BEAM_FWHM = 1.5

T_CMB = 2.7255                              # K

#: Foreground components for :class:`fastbox.foregrounds.ForegroundModel`.
#: ``amp`` (mK^2) and ``beta`` set the angular power spectrum
#: ``C_ell = amp (ell / 1000)^beta``, Santos et al. (2005); ``monopole`` (mK)
#: and the spectral index ``alpha`` are at ``FG_FREQ_REF``. The amplitudes,
#: monopoles and mean indices are the ones FastBox's own paper figures use.
#: ``alpha_std`` / ``alpha_smooth`` (deg) give the index a smooth spatial
#: variation, so no two pixels share a spectrum.
FG_FREQ_REF = 130.0                         # MHz
FOREGROUNDS = {
    'synchrotron': dict(amp=700.0, beta=-2.4, monopole=133e3,
                        alpha=-2.80, alpha_std=0.1, alpha_smooth=1.0),
    'point_sources': dict(amp=57.0, beta=-1.1, monopole=26.7e3,
                          alpha=-2.07, alpha_std=0.05, alpha_smooth=0.5),
}

_SEED_KEYS = ('hi', 'foreground', 'systematic', 'noise')


@contextmanager
def _seeded(seed):
    """Seed NumPy's global generator for FastBox, then put it back."""
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def component_seeds(seed):
    """One independent seed per component, spawned from ``seed``."""
    children = np.random.SeedSequence(seed).spawn(len(_SEED_KEYS))
    return {k: int(c.generate_state(1)[0]) for k, c in zip(_SEED_KEYS, children)}


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def mock_grid(shape=MOCK_SHAPE):
    """The :class:`~imgibbs.SurveyGrid` for a mock of this shape.

    The band starts at channel 550 of the L band and runs for ``shape[2]``
    channels; the transverse size is ``shape[:2]`` pixels of 0.3 deg.
    """
    return survey_grid((slice(None), slice(None), slice(0, shape[2])), shape)


def drift_footprint(shape=MOCK_SHAPE, height=0.66, tilt=0.3, cap=0.12):
    """A drift-scan footprint: a tilted band with rounded ends.

    Parameters
    ----------
    shape : tuple
        ``(Nx, Ny, Nz)``; the mask is the same in every channel.
    height : float
        Band height as a fraction of ``Ny``.
    tilt : float
        Dec pixels the band centre drops per RA pixel.
    cap : float
        Fraction of ``Nx`` over which each end is rounded off.

    Returns
    -------
    array (Nx, Ny, Nz) of 0/1
    """
    nx, ny, nz = shape
    i = np.arange(nx) + 0.5
    centre = 0.5 * ny + tilt * (0.5 * nx - i)
    half = 0.5 * height * ny * np.ones(nx)
    # Elliptical caps over the first and last `cap * nx` columns.
    L = max(cap * nx, 1.0)
    edge = np.minimum(i, nx - i)
    inside = edge < L
    half[inside] *= np.sqrt(1.0 - ((L - edge[inside]) / L) ** 2)
    j = np.arange(ny) + 0.5
    m2 = np.abs(j[None, :] - centre[:, None]) <= half[:, None]
    return np.broadcast_to(m2[:, :, None], shape).astype(float)


# ---------------------------------------------------------------------------
# Sky
# ---------------------------------------------------------------------------

def _box(grid):
    from fastbox.box import CosmoBox, default_cosmo
    return CosmoBox(cosmo=default_cosmo, box_scale=grid.box_dims,
                    nsamp=grid.shape, redshift=grid.z_mid, realise_now=False)


def hi_cube(grid, seed=None, sigma_nl=120.0):
    """One FastBox HI realisation on ``grid``, in K.

    Lognormal density with the HI bias, linear redshift-space distortions with
    a ``sigma_nl`` km/s Fingers-of-God damping, times the mean brightness
    temperature. FastBox returns mK; this returns K.
    """
    from fastbox.box import default_cosmo
    from fastbox.tracers import generate_hi_mock
    # FastBox divides by k^2 = 0 at the DC mode and then zeroes it; the
    # warning is noise.
    with _seeded(seed), np.errstate(divide='ignore', invalid='ignore'):
        mK = generate_hi_mock(default_cosmo, box_scale=grid.box_dims,
                              nsamp=grid.shape, redshift=grid.z_mid,
                              sigma_nl=sigma_nl)
    return np.asarray(mK) / 1e3


def foreground_cube(grid, seed=None, components=FOREGROUNDS, cmb=True):
    """FastBox foregrounds on ``grid``, in K.

    Each component is an amplitude map from
    :meth:`~fastbox.foregrounds.ForegroundModel.realise_foreground_amp` and a
    spectral-index map from ``realise_spectral_index``, combined as
    ``A(x) (nu / FG_FREQ_REF)^alpha(x)`` on the grid's own channel
    frequencies. Every pixel is a power law, smooth in frequency, but no two
    share an index.

    Returns
    -------
    total : array (Nx, Ny, Nz), K
    parts : dict of the same, one per component (and ``'cmb'``)
    """
    from fastbox.foregrounds import ForegroundModel
    freqs = np.asarray(grid.freqs)
    parts = {}
    with _seeded(seed):
        fg = ForegroundModel(_box(grid))
        for name, p in components.items():
            amp = fg.realise_foreground_amp(amp=p['amp'], beta=p['beta'],
                                            monopole=p['monopole'])
            alpha = fg.realise_spectral_index(
                mean_spec_idx=p['alpha'], std_spec_idx=p['alpha_std'],
                smoothing_scale=p['alpha_smooth'])
            scale = (freqs / FG_FREQ_REF)[None, None, :] ** alpha[:, :, None]
            parts[name] = amp[:, :, None] * scale / 1e3          # mK -> K
    if cmb:
        parts['cmb'] = np.full(grid.shape, T_CMB)
    return sum(parts.values()), parts


def noise_rms(t_sys=T_SYS, del_nu=DEL_NU, del_t=DEL_T):
    """Radiometer noise per voxel, K."""
    return float(np.sqrt(t_sys ** 2 / (del_nu * del_t)))


# ---------------------------------------------------------------------------
# Systematics
# ---------------------------------------------------------------------------

#: Defaults for each systematic, overridable through ``make_mock(sys_kwargs=)``.
#: ``rms`` is the injected amplitude in K -- 1e-3 is about the HI rms.
SYSTEMATICS = {
    'groundspill': dict(rms=1e-3, spill_level=0.5, period=17.5, gradient=0.3),
    'onef': dict(rms=1e-3, alpha=1.0, beta=1.0, knee_cycles=1.0,
                 n_scan=2, n_spec=2),
    'leakage': dict(rms=1e-3, rm=1000.0),
}


def systematic_cube(name, grid, seed=None, **kwargs):
    """A systematic contaminant on ``grid``, from :mod:`imgibbs.systematics`.

    Returns ``(cube, basis, g_true)``: the cube in K, the
    :class:`~imgibbs.SystematicBasis` it lives in, and its exact amplitudes in
    that basis. For ground spill the cube also carries the smooth spill
    (``spill_level``), which is deliberately outside ``basis`` -- the
    foreground block absorbs it.
    """
    p = {**SYSTEMATICS[name], **kwargs}
    rng = np.random.default_rng(seed)
    freqs, shape = grid.freqs, grid.shape
    if name == 'groundspill':
        cube, truth = groundspill_cube(freqs, shape, ripple_rms=p['rms'],
                                       spill_level=p['spill_level'],
                                       period=p['period'],
                                       gradient=p['gradient'], rng=rng)
        return cube, truth['basis'], truth['g_true']
    if name == 'onef':
        basis, prior_var = onef_basis(freqs, shape, alpha=p['alpha'],
                                      beta=p['beta'],
                                      knee_cycles=p['knee_cycles'],
                                      n_scan=p['n_scan'], n_spec=p['n_spec'])
    elif name == 'leakage':
        basis = leakage_basis(freqs, shape, rm=p['rm'])
        prior_var = np.full(basis.g_shape, 1.0 / basis.n_params)
    else:
        raise ValueError(f'unknown systematic {name!r}; '
                         f'choose from {sorted(SYSTEMATICS)}')
    cube, g_true = realise(basis, prior_var, p['rms'], rng=rng)
    return cube, basis, g_true


def _systematic_names(systematic):
    """``None``, one name, ``'all'`` or a list -> a list of names (or None)."""
    if systematic is None:
        return None
    if isinstance(systematic, str):
        names = (list(SYSTEMATICS) if systematic == 'all'
                 else [s for s in systematic.split(',') if s])
    else:
        names = list(systematic)
    unknown = [n for n in names if n not in SYSTEMATICS]
    if unknown:
        raise ValueError(f'unknown systematic {unknown}; '
                         f"choose from {sorted(SYSTEMATICS)} or 'all'")
    if len(set(names)) != len(names):
        raise ValueError(f'systematic named twice: {names}')
    return names or None


def systematics_cube(systematic, grid, seed=None, sys_kwargs=None):
    """One systematic or several, summed. See :func:`systematic_cube`.

    ``systematic`` is a name, a list of names, or ``'all'``. With one name
    this *is* :func:`systematic_cube` (same seed, same cube) and
    ``sys_kwargs`` is its overrides. With several, ``sys_kwargs`` maps each
    name to its overrides, each systematic draws from its own stream --
    fixed by its place in :data:`SYSTEMATICS`, so adding or dropping one
    leaves the others unchanged -- and the basis is a
    :class:`~imgibbs.CompositeBasis` in the order given.

    Returns ``(cube, basis, g_true, kwargs)``; ``kwargs`` is the full set of
    parameters used, in the same layout as ``sys_kwargs``.
    """
    names = _systematic_names(systematic)
    if names is None:
        return np.zeros(grid.shape), None, None, {}
    sys_kwargs = sys_kwargs or {}
    if len(names) == 1:
        name = names[0]
        # Flat overrides, or keyed by name as for several.
        own = (sys_kwargs[name] if isinstance(sys_kwargs.get(name), dict)
               else sys_kwargs)
        kw = {**SYSTEMATICS[name], **own}
        cube, basis, g = systematic_cube(name, grid, seed, **kw)
        return cube, basis, g, kw
    unknown = set(sys_kwargs) - set(names)
    if unknown:
        raise ValueError(f'sys_kwargs for systematics not in the mock: '
                         f'{sorted(unknown)} (with several systematics, '
                         f'sys_kwargs is keyed by name)')
    streams = dict(zip(SYSTEMATICS, np.random.SeedSequence(seed).spawn(
        len(SYSTEMATICS))))
    cube, parts, gs, used = np.zeros(grid.shape), [], [], {}
    for name in names:
        kw = {**SYSTEMATICS[name], **sys_kwargs.get(name, {})}
        c, b, g = systematic_cube(name, grid,
                                  int(streams[name].generate_state(1)[0]),
                                  **kw)
        cube += c
        parts.append(b)
        gs.append(np.asarray(g).ravel())
        used[name] = kw
    return cube, CompositeBasis(parts, names), np.concatenate(gs), used


# ---------------------------------------------------------------------------
# The whole thing
# ---------------------------------------------------------------------------

@dataclass
class Mock:
    """A mock observation and every component that went into it.

    ``cube`` is what the sampler sees; the rest is the answer. Flagged voxels
    are exactly zero and observed ones never are, so ``cube != 0`` recovers
    ``mask`` -- the convention :func:`imgibbs.kbins_from_crop` relies on.
    """

    grid: object
    cube: np.ndarray
    mask: np.ndarray
    hi: np.ndarray
    foreground: np.ndarray
    systematic: np.ndarray
    noise: np.ndarray
    beam: object = None
    sys_basis: object = None
    g_true: np.ndarray = None
    config: dict = field(default_factory=dict)

    @property
    def shape(self):
        return self.grid.shape

    @property
    def sky(self):
        """HI + foreground as the telescope sees it: through the beam."""
        sky = self.hi + self.foreground
        return self.beam.apply(sky) if self.beam is not None else sky

    def summary(self):
        m = self.mask > 0
        lines = [
            f'grid         : {self.shape}, {self.grid.freqs[0]:.2f}-'
            f'{self.grid.freqs[-1]:.2f} MHz, z {self.grid.z_lo:.3f}-'
            f'{self.grid.z_hi:.3f}',
            f'footprint    : {m[:, :, 0].mean():.1%} of the map',
            f'HI           : rms {self.hi.std():.3e} K',
            f'foreground   : rms {self.foreground[m].std():.3e} K '
            f'({self.foreground[m].std() / self.hi.std():.0f}x the HI), '
            f'mean {self.foreground[m].mean():.3f} K',
            'beam         : ' + (
                'none' if self.beam is None else
                f"Gaussian, FWHM {self.config['beam_fwhm']:g} deg at "
                f"{self.config['beam_nu_ref']:.1f} MHz"),
            'systematic   : ' + self._systematic_summary(),
            f'noise        : rms {self.config["noise_rms"]:.3e} K',
        ]
        return '\n'.join(lines)

    def _systematic_summary(self):
        if self.sys_basis is None:
            return 'none'
        name, kw = self.config['systematic'], self.config['sys_kwargs']
        if isinstance(name, str):
            return (f"{name}, rms {kw['rms']:.1e} K, "
                    f"{self.sys_basis.n_params} amplitudes")
        each = ', '.join(f"{n} {kw[n]['rms']:.1e} K" for n in name)
        return (f'{each} (rms each); {self.systematic[self.mask > 0].std():.1e}'
                f' K together, {self.sys_basis.n_params} amplitudes')

    def save(self, path):
        """Write to ``path`` (``.npz``). Rebuild with :func:`load_mock`."""
        arrays = dict(cube=self.cube, mask=self.mask[:, :, 0], hi=self.hi,
                      foreground=self.foreground, systematic=self.systematic,
                      noise=self.noise)
        if self.g_true is not None:
            arrays['g_true'] = self.g_true
        np.savez(path, config=json.dumps(self.config), **arrays)

    @classmethod
    def load(cls, path):
        return load_mock(path)


def make_mock(shape=MOCK_SHAPE, seed=0, beam_fwhm=BEAM_FWHM, beam_nu_ref=None,
              beam_index=1.0, systematic=None, sys_kwargs=None,
              t_sys=T_SYS, del_t=DEL_T, footprint=None, foregrounds=True):
    """Build a complete mock observation.

    Parameters
    ----------
    shape : tuple
        ``(Nx, Ny, Nz)``. The k-binning is derived from the footprint, so any
        shape works -- but results in ``docs/STATUS.md`` are for the default.
    seed : int
        Master seed. Each component gets its own seed spawned from it.
    beam_fwhm : float or None
        Gaussian beam FWHM in degrees at ``beam_nu_ref`` (default: the band
        centre). ``None`` for no beam.
    beam_index : float
        Chromatic index, FWHM ~ nu^-index.
    systematic : str, list of str, or None
        ``'groundspill'``, ``'onef'``, ``'leakage'``, a list of them (or a
        comma-separated string), or ``'all'``. Several are summed, and
        ``sys_basis`` is then a :class:`~imgibbs.CompositeBasis`.
    sys_kwargs : dict
        Overrides for :data:`SYSTEMATICS`. With several systematics, keyed by
        name: ``{'leakage': {'rm': 500}}``.
    t_sys, del_t : float
        Radiometer noise.
    footprint : dict or array, optional
        Keyword arguments for :func:`drift_footprint`, or the mask itself.
    foregrounds : bool
        ``False`` for a signal + noise mock.

    Returns
    -------
    Mock
    """
    shape = tuple(int(n) for n in shape)
    grid = mock_grid(shape)
    seeds = component_seeds(seed)

    if footprint is None or isinstance(footprint, dict):
        mask = drift_footprint(shape, **(footprint or {}))
    else:
        mask = np.broadcast_to(np.asarray(footprint, float).reshape(
            shape[:2] + (-1,)), shape).astype(float)

    hi = hi_cube(grid, seeds['hi'])
    fg = (foreground_cube(grid, seeds['foreground'])[0] if foregrounds
          else np.zeros(shape))

    beam = None
    if beam_fwhm is not None:
        beam_nu_ref = (float(np.mean(grid.freqs)) if beam_nu_ref is None
                       else float(beam_nu_ref))
        beam = gaussian_beam(grid.freqs, shape, beam_fwhm, beam_nu_ref,
                             index=beam_index, pix_deg=PIX_DEG)

    names = _systematic_names(systematic)
    sys_cube, sys_basis, g_true, sys_kwargs = systematics_cube(
        names, grid, seeds['systematic'], sys_kwargs)
    systematic = (None if names is None else
                  names[0] if len(names) == 1 else names)

    sigma = noise_rms(t_sys, DEL_NU, del_t)
    noise = np.random.default_rng(seeds['noise']).normal(scale=sigma,
                                                         size=shape)

    sky = hi + fg
    if beam is not None:
        sky = beam.apply(sky)
    cube = (sky + sys_cube + noise) * mask
    # A flagged voxel is one that is exactly zero; make sure no real one is.
    cube[(mask > 0) & (cube == 0)] = 1e-12

    config = dict(shape=list(shape), seed=seed, seeds=seeds,
                  beam_fwhm=beam_fwhm, beam_nu_ref=beam_nu_ref,
                  beam_index=beam_index, systematic=systematic,
                  sys_kwargs=sys_kwargs,
                  t_sys=t_sys, del_t=del_t, del_nu=DEL_NU, noise_rms=sigma,
                  foregrounds=bool(foregrounds),
                  footprint=footprint if isinstance(footprint, dict) else
                  ('custom' if footprint is not None else {}))
    return Mock(grid=grid, cube=cube, mask=mask, hi=hi, foreground=fg,
                systematic=sys_cube, noise=noise, beam=beam,
                sys_basis=sys_basis, g_true=g_true, config=config)


def load_mock(path):
    """Read a :class:`Mock` written by :meth:`Mock.save`.

    The beam and systematic basis are rebuilt from the stored configuration
    rather than stored, so they are exactly what :func:`make_mock` would make.
    """
    with np.load(path) as z:
        config = json.loads(str(z['config']))
        arrays = {k: z[k] for k in z.files if k != 'config'}
    shape = tuple(config['shape'])
    grid = mock_grid(shape)
    beam = None
    if config['beam_fwhm'] is not None:
        beam = gaussian_beam(grid.freqs, shape, config['beam_fwhm'],
                             config['beam_nu_ref'],
                             index=config['beam_index'], pix_deg=PIX_DEG)
    sys_basis = None
    if config['systematic'] is not None:
        _, sys_basis, _, _ = systematics_cube(config['systematic'], grid,
                                              config['seeds']['systematic'],
                                              config['sys_kwargs'])
    mask = np.broadcast_to(arrays.pop('mask')[:, :, None], shape).astype(float)
    return Mock(grid=grid, mask=mask, beam=beam, sys_basis=sys_basis,
                g_true=arrays.pop('g_true', None), config=config, **arrays)


# ---------------------------------------------------------------------------
# A signal prior that is not the answer
# ---------------------------------------------------------------------------

def binned_power(cube, sig_k, idxs):
    """Mean ``|s_k|^2`` per k-bin, in the normalisation the ``S`` step uses.

    Mean-subtracted, orthonormal FFT, averaged over the modes of each bin --
    so it is directly comparable to the sampler's ``Pk_trace``.
    """
    s = np.fft.fftn(cube - cube.mean(), norm='ortho').ravel()
    p = np.abs(s) ** 2
    occupied = np.unique(idxs)
    occupied = occupied[occupied > 0]
    return np.array([p[idxs == b].mean() for b in occupied])


def expected_hi_power(grid, sig_k, idxs, n_real=8, seed=12345):
    """Mean binned HI power over ``n_real`` independent realisations.

    A starting point for ``S`` that is the *expected* power under the
    fiducial cosmology, not the power of the realisation in the mock. Use a
    ``seed`` different from the mock's. Returns ``(mean, std)`` per bin.
    """
    seeds = np.random.SeedSequence(seed).spawn(n_real)
    pk = np.array([binned_power(hi_cube(grid, int(s.generate_state(1)[0])),
                                sig_k, idxs) for s in seeds])
    return pk.mean(axis=0), pk.std(axis=0)


def injection_base(shape=MOCK_SHAPE, seed=0):
    """HI, foreground and footprint for the injection scripts.

    ``scripts/systematics_injection.py`` and ``scripts/beam_injection.py``
    add their own systematic and noise on top; their report and benchmark
    scripts rebuild the same truth by calling this with the same seed.
    Returns ``(hi, foreground, mask)``, all ``(Nx, Ny, Nz)``.
    """
    m = make_mock(shape, seed=seed, beam_fwhm=None)
    return m.hi, m.foreground, m.mask
