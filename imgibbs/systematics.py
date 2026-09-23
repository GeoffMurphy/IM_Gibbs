"""Instrumental systematics as a third component in the Gibbs model.

The sampler's data model is ``d = w * (Us s + Uf f) + n``. This module adds a
third additive term::

    d = w * (Us s + Uf f + Ug g) + n

where ``Ug`` is a **fixed, low-rank** basis of systematic templates and ``g``
is a short vector of amplitudes sampled jointly with everything else. The
first (and so far only) systematic implemented is ground spill.

Why a separate block at all
---------------------------
The foreground block already has per-pixel free amplitudes on ``n_modes``
smooth frequency modes, so *any* component that is smooth in frequency lies
inside its span, whatever its spatial structure. Measured on this grid, with
the orthonormalised Legendre basis the sampler uses, the fraction of a
template's power the foreground block can absorb is:

============================  ======  ======  ======
template                      n=6     n=10    n=20
============================  ======  ======  ======
smooth spill, ``nu^beta``     1.0000  1.0000  1.0000
ripple, 40 MHz period         0.9945  1.0000  1.0000
ripple, 20 MHz period         0.2872  0.9886  1.0000
ripple, 17.5 MHz period       0.1837  0.9378  1.0000
ripple, 10 MHz period         0.0577  0.2155  0.9987
ripple, 5 MHz period          0.0149  0.0526  0.1954
============================  ======  ======  ======

Reproduce with :func:`period_scan` on the (70, 45, 250) band; ``tests/`` pins
these values.

Two things follow, and they set the whole design of this module.

**The smooth part of ground spill is not identifiable, and does not need to
be.** It is absorbed to machine precision at any ``n_modes``. Putting a smooth
template in ``Ug`` would only make the system rank-deficient against ``Uf``.
:func:`spectral_templates` therefore omits it by default.

**The ripple is identifiable, and it lands where it hurts.** A ripple of
period ``P`` MHz is a pure ``k_parallel`` mode at ``k = 2*pi*B/(Lz*P)``; on the
current (70, 45, 250) grid (B = 52.04 MHz, Lz = 254.39 Mpc) that is

    P = 20 MHz -> k = 0.064,  17.5 -> 0.073,  10 -> 0.129 Mpc^-1

against k-bin centres ``[0.068, 0.160, 0.374, 0.875, 2.046]``. A standing wave
anywhere in the 10-20 MHz range sits in the lowest one or two signal bins.

Raising ``n_modes`` does absorb the ripple -- but ``docs/STATUS.md`` records
that ``n_modes = 20`` removes the 21cm signal too (T(k) 0.006-0.06) and
destroys convergence (tau_int 54-105). So "use more foreground modes" is not
available as a fix, which is the argument for modelling the systematic
explicitly with a tight prior instead of buying it more free polynomials.

The model
---------
Ground spill is far-sidelobe pickup of ~280 K ground emission. Two pieces:

* a **smooth envelope** ``eta(nu)``: the spillover fraction rises toward low
  frequency roughly as the beam solid angle, ``(nu/nu_ref)^beta`` with
  ``beta ~ -2``;
* a **standing-wave ripple** of period ``P``, set by the round trip between
  the dish and the subreflector/feed, modulating that envelope.

Spatially, a drift scan at fixed elevation sees a ground pattern that is
constant in the telescope frame, so to lowest order the contribution is the
same in every map pixel. What breaks that degeneracy is the azimuth range
covered within a scan, which maps onto the scan direction of the map. The
spatial model here is therefore a **low-order polynomial along the scan axis**
(constant + gradient by default) -- not per-pixel freedom, which would make
the systematic degenerate with the foreground all over again.

That gives ``n_s * n_t`` parameters -- four by default -- for a contaminant
that would otherwise sit on top of the lowest k-bins.

Nothing here is MeerKAT-specific: the period, the envelope index, the
amplitudes and the polynomial order are all arguments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Physical temperature of the ground, K. Very nearly a constant of nature for
#: this purpose -- ground emission is thermal at ~ambient.
T_GROUND = 280.0

#: Default standing-wave period in MHz. For a Gregorian dish this is
#: ``c / (2 * focal distance)``; ~17.5 MHz corresponds to ~8.5 m. It is a
#: default, not a constant -- set it from the instrument you are modelling.
RIPPLE_PERIOD_MHZ = 17.5

#: Default spectral index of the spillover envelope. The spilled power follows
#: the beam solid angle, which goes as lambda^2, hence -2.
ENVELOPE_INDEX = -2.0


# ---------------------------------------------------------------------------
# Frequency structure
# ---------------------------------------------------------------------------

def spillover_envelope(freqs, beta=ENVELOPE_INDEX, nu_ref=None):
    """The smooth part of the spillover fraction, normalised to 1 at ``nu_ref``.

    Parameters
    ----------
    freqs : array, MHz
    beta : float
        Spectral index. ``-2`` follows the beam solid angle.
    nu_ref : float, optional
        Reference frequency. Defaults to the band centre.

    Returns
    -------
    array, same shape as ``freqs``

    Notes
    -----
    On its own this is **invisible to the sampler** -- it lies entirely within
    the foreground basis (see the module docstring). It is here because it
    modulates the ripple, and because injecting it is how you demonstrate that
    the foreground block really does swallow it.
    """
    freqs = np.asarray(freqs, dtype=float)
    if nu_ref is None:
        nu_ref = 0.5 * (freqs[0] + freqs[-1])
    return (freqs / nu_ref) ** beta


def spectral_templates(freqs, period=RIPPLE_PERIOD_MHZ, n_harmonics=1,
                       beta=ENVELOPE_INDEX, include_smooth=False):
    """Frequency templates for the ground-spill block, unit RMS each.

    The ripple enters as a cosine/sine **pair** at each harmonic rather than
    as an amplitude and a phase. That is the whole reason this block can live
    inside the constrained realisation: amplitude and phase are a nonlinear
    pair, but the two quadratures are linear parameters, so the joint solve
    stays linear and no Metropolis step is needed. The period itself is *not*
    a linear parameter -- it is fixed here (see :func:`period_scan` if you
    need to search over it).

    Parameters
    ----------
    freqs : array, MHz
    period : float
        Standing-wave period in MHz.
    n_harmonics : int
        Number of harmonics to include. Harmonic ``m`` has period
        ``period / m``, i.e. twice the ``k_parallel``.
    beta : float
        Index of the envelope multiplying the ripple.
    include_smooth : bool
        Add the bare envelope as a template. **Off by default and you almost
        certainly want it off**: it is degenerate with the foreground block to
        machine precision, so including it adds a null direction that only the
        prior on ``G`` regularises.

    Returns
    -------
    array, shape ``(n_t, n_freq)``
        ``n_t = 2 * n_harmonics`` (+1 if ``include_smooth``). Each row is
        normalised to unit RMS over the band, so the matching amplitude in
        ``g`` is read directly in K.
    """
    freqs = np.asarray(freqs, dtype=float)
    if period <= 0:
        raise ValueError(f'period must be positive, got {period}')
    env = spillover_envelope(freqs, beta=beta)
    phase = 2 * np.pi * (freqs - freqs[0]) / period

    rows = []
    if include_smooth:
        rows.append(env)
    for m in range(1, int(n_harmonics) + 1):
        rows.append(env * np.cos(m * phase))
        rows.append(env * np.sin(m * phase))

    out = np.array(rows)
    return out / np.sqrt(np.mean(out ** 2, axis=1))[:, None]


def ripple_wavenumber(period, bandwidth, Lz):
    """``k_parallel`` in Mpc^-1 of a ripple of ``period`` MHz.

    A ripple is a single Fourier mode along the frequency axis, so it does not
    spread across k-bins -- it dumps all of its power into whichever bin
    contains this value. Use it to check, before running anything, whether a
    given standing wave lands on top of the signal bins.

    Parameters
    ----------
    period : float
        Ripple period, MHz.
    bandwidth : float
        Total band width, MHz (``freqs[-1] - freqs[0]``).
    Lz : float
        Comoving depth of the band, Mpc (``grid.box_dims[2]``).
    """
    return 2 * np.pi * bandwidth / (Lz * period)


def period_scan(freqs, periods, fg_basis):
    """Fraction of each ripple period the foreground block can absorb.

    This is the table in the module docstring, for an arbitrary band and
    foreground basis. Anything close to 1 is invisible to a systematics block
    -- not because it is harmless, but because the foreground already removes
    it. Anything close to 0 passes straight through the foreground clean and
    is what this module is for.

    Parameters
    ----------
    freqs : array, MHz
    periods : array
        Ripple periods, MHz.
    fg_basis : array, shape (n_modes, n_freq)
        The foreground basis, **orthonormal rows** (the sampler's ``evecs``).

    Returns
    -------
    array, one absorbed fraction in [0, 1] per period.
    """
    fg_basis = np.asarray(fg_basis, dtype=float)
    orth = np.abs(fg_basis @ fg_basis.T - np.eye(len(fg_basis))).max()
    if orth > 1e-8:
        raise ValueError(f'fg_basis rows are not orthonormal (max |B B^T - I| '
                         f'= {orth:.2e}); the projection below assumes they are')

    out = []
    for period in np.atleast_1d(periods):
        # Average the two quadratures: the answer should not depend on where
        # the band happens to cut the standing wave.
        tmpl = spectral_templates(freqs, period=period, n_harmonics=1)
        tmpl = tmpl / np.linalg.norm(tmpl, axis=1)[:, None]
        out.append(float(np.mean(np.sum((fg_basis @ tmpl.T) ** 2, axis=0))))
    return np.array(out)


# ---------------------------------------------------------------------------
# Spatial structure
# ---------------------------------------------------------------------------

def scan_templates(shape, order=1, scan_axis=0):
    """Low-order polynomial patterns along the scan direction, unit RMS each.

    Parameters
    ----------
    shape : tuple (Nx, Ny, Nz)
        Cube shape. Only the two spatial axes are used.
    order : int
        Highest polynomial degree. ``0`` is a constant offset (the pure
        drift-scan limit), ``1`` adds a gradient, and so on.
    scan_axis : {0, 1}
        Which spatial axis the telescope scanned along. For the MeerKLASS
        drift scan that is axis 0 (RA), since the sky moves through the beam
        in RA while elevation is held fixed.

    Returns
    -------
    array, shape ``(order + 1, Nx * Ny)``

    Notes
    -----
    Deliberately **not** per-pixel. Per-pixel amplitudes on a smooth spectral
    template are exactly what the foreground block already provides, so a
    systematic with that much spatial freedom is unidentifiable no matter what
    its frequency structure looks like. The physics agrees: ground pickup is
    fixed in the telescope frame and varies only with pointing, so it is
    smooth and low-order in map coordinates.
    """
    if scan_axis not in (0, 1):
        raise ValueError(f'scan_axis must be 0 or 1, got {scan_axis}')
    nx, ny = shape[0], shape[1]
    n_scan = (nx, ny)[scan_axis]

    coord = np.linspace(-1.0, 1.0, n_scan)
    vander = np.polynomial.legendre.legvander(coord, int(order)).T  # (order+1, n)

    rows = []
    for row in vander:
        pattern = (np.repeat(row[:, None], ny, axis=1) if scan_axis == 0
                   else np.repeat(row[None, :], nx, axis=0))
        rows.append(pattern.ravel())

    out = np.array(rows)
    return out / np.sqrt(np.mean(out ** 2, axis=1))[:, None]


# ---------------------------------------------------------------------------
# The basis operator
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SystematicBasis:
    """A separable systematic basis and the ``Ug`` operator built from it.

    Each template is an outer product ``spatial[s] x spectral[t]``, so the
    amplitude array ``g`` has shape ``(n_s, n_t)``. Separability is what keeps
    this cheap: applying ``Ug`` is two small matrix multiplies, and the Gram
    matrix needed by the preconditioner factorises exactly (see :meth:`gram`)
    instead of being assembled over a million voxels.

    Attributes
    ----------
    spatial : array, shape (n_s, Nx*Ny)
    spectral : array, shape (n_t, n_freq)
    shape : tuple (Nx, Ny, Nz)
    """

    spatial: np.ndarray
    spectral: np.ndarray
    shape: tuple

    def __post_init__(self):
        nx, ny, nz = self.shape
        if self.spatial.shape[1] != nx * ny:
            raise ValueError(f'spatial templates are {self.spatial.shape[1]} '
                             f'pixels, cube is {nx * ny}')
        if self.spectral.shape[1] != nz:
            raise ValueError(f'spectral templates are {self.spectral.shape[1]} '
                             f'channels, cube is {nz}')

    # -- shape metadata ----------------------------------------------------
    @property
    def n_s(self):
        return self.spatial.shape[0]

    @property
    def n_t(self):
        return self.spectral.shape[0]

    @property
    def g_shape(self):
        """Shape of the amplitude array ``g``."""
        return (self.n_s, self.n_t)

    @property
    def n_params(self):
        """Length of ``g`` flattened -- how much the solution vector grows."""
        return self.n_s * self.n_t

    # -- the operator ------------------------------------------------------
    def apply(self, g):
        """``Ug g``: amplitudes -> a flattened cube."""
        g = np.asarray(g).reshape(self.g_shape)
        return (self.spatial.T @ g @ self.spectral).ravel()

    def adjoint(self, y):
        """``Ug^T y``: a flattened cube -> amplitudes, shape ``(n_s, n_t)``."""
        y = np.asarray(y).reshape(-1, self.shape[2])
        return self.spatial @ y @ self.spectral.T

    def gram(self):
        """``Ug^T Ug`` as a dense ``(n_params, n_params)`` matrix.

        Because the templates are separable, the full Gram matrix is the
        Kronecker product of the two small ones -- exact, not an
        approximation. Ordering matches ``g.ravel()`` (spatial index slowest).
        """
        return np.kron(self.spatial @ self.spatial.T,
                       self.spectral @ self.spectral.T)

    def cube(self, g):
        """``Ug g`` reshaped back to ``(Nx, Ny, Nz)``, for plotting."""
        return self.apply(g).reshape(self.shape)


def groundspill_basis(freqs, shape, period=RIPPLE_PERIOD_MHZ, n_harmonics=1,
                      order=1, scan_axis=0, beta=ENVELOPE_INDEX,
                      include_smooth=False):
    """Build the ground-spill :class:`SystematicBasis` for one grid.

    The default is four parameters: {constant, scan gradient} x {cos, sin} at
    one standing-wave period.

    See :func:`spectral_templates` and :func:`scan_templates` for the
    arguments; ``include_smooth`` is off for the reason given in the module
    docstring.
    """
    return SystematicBasis(
        spatial=scan_templates(shape, order=order, scan_axis=scan_axis),
        spectral=spectral_templates(freqs, period=period,
                                    n_harmonics=n_harmonics, beta=beta,
                                    include_smooth=include_smooth),
        shape=tuple(int(n) for n in shape),
    )


# ---------------------------------------------------------------------------
# Injection
# ---------------------------------------------------------------------------

def groundspill_cube(freqs, shape, ripple_rms, spill_level=0.0,
                     period=RIPPLE_PERIOD_MHZ, beta=ENVELOPE_INDEX,
                     order=1, scan_axis=0, gradient=0.3, phase=0.0,
                     rng=None):
    """A ground-spill contaminant cube, in K, for injection.

    Parameters
    ----------
    freqs : array, MHz
    shape : tuple (Nx, Ny, Nz)
    ripple_rms : float
        RMS of the ripple over the cube, K. **This is the knob that matters.**
        For scale on the current grid: the simulated H I cube has an RMS of
        8.6e-4 K and the thermal noise 1.1e-3 K per voxel, so ``ripple_rms =
        1e-3`` is a systematic at roughly the level of the signal it sits on.
    spill_level : float
        RMS of the *smooth* spillover component, K. Defaults to zero. Setting
        it large (0.5 K is physically reasonable for a few 1e-3 of the beam on
        280 K ground) is a useful demonstration: it changes the recovered P(k)
        not at all, because the foreground block absorbs it completely.
    period, beta, order, scan_axis :
        As :func:`groundspill_basis`.
    gradient : float
        Amplitude of the scan-direction gradient relative to the constant
        term. ``0`` is the pure drift-scan limit -- spatially uniform, and
        therefore only distinguishable from the foreground by its spectrum.
    phase : float
        Standing-wave phase in radians, i.e. where the band cuts the wave.
    rng : numpy Generator, optional
        Only used if ``order > 1``, to fill the higher spatial terms.

    Returns
    -------
    cube : array (Nx, Ny, Nz), K
    truth : dict
        ``basis`` (the :class:`SystematicBasis` that spans the ripple) and
        ``g_true`` (the amplitudes of the ripple within it), so a recovery
        test has something exact to compare against. The smooth part is
        deliberately **not** in ``basis``.
    """
    freqs = np.asarray(freqs, dtype=float)
    rng = np.random.default_rng() if rng is None else rng

    basis = groundspill_basis(freqs, shape, period=period, n_harmonics=1,
                              order=order, scan_axis=scan_axis, beta=beta)

    # Spatial weights: constant + a gradient, then whatever is asked for above.
    weights = np.zeros(basis.n_s)
    weights[0] = 1.0
    if basis.n_s > 1:
        weights[1] = gradient
    if basis.n_s > 2:
        weights[2:] = gradient * rng.normal(scale=0.3, size=basis.n_s - 2)

    # cos/sin quadratures carry the phase.
    g_true = np.outer(weights, [np.cos(phase), np.sin(phase)])
    cube = basis.cube(g_true)
    # Scale amplitudes and cube by the SAME factor, so g_true stays the exact
    # answer for the cube that is returned. A recovery test is scored against
    # it, so the two must not drift apart.
    scale = ripple_rms / cube.std()
    cube = cube * scale
    g_true = g_true * scale

    if spill_level:
        smooth = SystematicBasis(
            spatial=basis.spatial,
            spectral=spectral_templates(freqs, period=period, n_harmonics=0,
                                        beta=beta, include_smooth=True),
            shape=basis.shape,
        )
        smooth_cube = smooth.cube(weights[:, None])
        cube = cube + smooth_cube * (spill_level / smooth_cube.std())

    return cube, {'basis': basis, 'g_true': g_true}
