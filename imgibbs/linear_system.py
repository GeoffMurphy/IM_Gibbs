"""The linear system ``Ax = b`` solved once per Gibbs iteration.

The joint solution vector packs three real blocks::

    x = [ Re(s), Im(s), f ]

where ``s`` is the 21cm signal in Fourier space (complex, split into real and
imaginary parts so the vector stays real for the Krylov solver) and ``f`` are
the foreground mode amplitudes.

``A`` is never formed explicitly -- :func:`construct_A` applies it as a
matrix-vector product, which is what makes the problem tractable at
~1.6 million voxels. Adding the omega terms to the right-hand side turns the
solve from a MAP/Wiener-filter estimate into a draw from the posterior; that
is the Gaussian Constrained Realisation step.

Systematics
-----------
Passing a :class:`imgibbs.systematics.SystematicBasis` as ``sys_basis`` adds a
fourth block, ``x = [Re(s), Im(s), f, g]``, for an instrumental systematic
with a fixed low-rank basis and a fixed prior covariance ``G``::

    d = w * (Us s + Uf f + Ug g) + n

``sys_basis=None`` is the default everywhere and leaves the three-block system
numerically identical to what it was, term for term -- the extra contributions
are added only when a basis is supplied.

``G`` is a **prior**, not a sampled covariance. The foreground ``F`` can be
drawn from an inverse-Wishart because there are ``Npix`` independent amplitude
vectors to estimate it from; ``g`` is a single vector of a handful of numbers,
so its covariance is not identified by the data and has to come from
instrument characterisation. See ``imgibbs/systematics.py``.

Beam
----
Passing a :class:`imgibbs.beam.Beam` as ``beam`` convolves the sky -- signal
and foreground, not the systematic -- with a chromatic primary beam::

    d = w * (B Us s + B Uf f + Ug g) + n

``B`` is symmetric, so the adjoint of ``B Us`` is ``Us^T B`` and the same
call serves both ways. ``beam=None`` leaves the beamless system untouched:
the beam is a separate code path, not a multiplication by one. See
``imgibbs/beam.py``.
"""

import numpy as np
import numpy.fft as fft


def construct_Uf(n_modes, shape, freqs):
    """Build a power-law x log-polynomial foreground basis.

    An alternative to the PCA and Legendre bases used in the notebooks:
    ``(nu/nu_ref)^beta * log10(nu/nu_ref)^(l+1)``. Not used by the current
    sampler configuration, kept as a modelling option.
    """
    beta = -2.4
    ref_freq = 130

    row = []
    for ll in range(0, shape[0]):
        if ll > n_modes - 1:
            row.append(0 * freqs)
        else:
            row.append(((freqs / ref_freq)**beta)
                       * (np.log10(freqs / ref_freq))**(ll + 1))

    Uf = np.broadcast_to(np.array(row), (shape[0],) + np.array(row).shape)
    return Uf[:, :n_modes, :][0]


def Uf(evecs, amps, transpose):
    """Apply the foreground projection operator, or its transpose."""
    if transpose:
        gn_fit = amps @ evecs.T
    else:
        gn_fit = (amps @ evecs).flatten()

    return gn_fit.real


def Us(s, transpose):
    """Apply the signal operator: an orthonormal real FFT, or its inverse.

    ``transpose=True`` is the forward transform (real space -> Fourier),
    ``False`` the inverse.

    .. warning::
       ``irfftn`` is called without ``s=``, so it infers the last-axis length
       and always returns an EVEN number of channels. This is exact for the
       500- and 250-channel cubes used here, but silently drops a channel if
       you ever re-channelise to an odd count. See ``docs/STATUS.md``.
    """
    if transpose:
        return fft.rfftn(s, norm='ortho')
    else:
        return fft.irfftn(s, norm='ortho')


def construct_A(x, S, Nw_inv, F, w, evecs, rfft_len, rfft_shape, f_len,
                f_shape, shape, sys_basis=None, G=None, beam=None):
    """Evaluate the LHS of ``Ax = b`` as a matrix-vector product.

    With ``sys_basis`` supplied, ``x`` carries a fourth block ``g`` of length
    ``sys_basis.n_params`` and the returned vector grows to match. The
    signal-foreground part of the product is unchanged.

    With ``beam`` supplied, the signal and foreground enter the data through
    ``B``; the systematic does not.
    """
    x0_recon = (x[0:rfft_len].reshape(rfft_shape)
                + x[rfft_len:2 * rfft_len].reshape(rfft_shape) * 1j)
    x1_recon = x[2 * rfft_len:2 * rfft_len + f_len].reshape(f_shape)

    # Truncate S from full fftn shape (Nx,Ny,Nz) to rfft shape (Nx,Ny,Nz//2+1)
    S = S.reshape(shape)[:, :, :shape[2] // 2 + 1].flatten()

    # Take only the real part for the foreground amplitudes
    x1_recon = x1_recon.real

    if beam is not None:
        return _construct_A_beam(x, x0_recon, x1_recon, S, Nw_inv, F, evecs,
                                 rfft_len, rfft_shape, f_len, shape,
                                 sys_basis, G, beam)

    A00 = (((1 / S) * x0_recon.flatten()).reshape(rfft_shape)
           + Us(((Nw_inv) * Us(x0_recon, False).flatten()).reshape(shape), True))

    A01 = Us(((Nw_inv) * Uf(evecs, x1_recon, False)).reshape(shape), True)

    A10 = Uf(evecs, ((Nw_inv) * Us(x0_recon, False).flatten()).reshape(shape), True)

    A11 = ((1 / F) * x1_recon
           + Uf(evecs, ((Nw_inv) * Uf(evecs, x1_recon, False)).reshape(shape), True))

    Ax0 = A00 + A01
    Ax1 = A10 + A11

    if sys_basis is None:
        return np.concatenate([Ax0.real.flatten(), Ax0.imag.flatten(),
                               Ax1.real.flatten()])

    g_off = 2 * rfft_len + f_len
    x2_recon = x[g_off:g_off + sys_basis.n_params].real
    Ug_g = sys_basis.apply(x2_recon)

    # The systematic's contribution to the signal and foreground rows, and the
    # signal and foreground contributions to its own row.
    A02 = Us(((Nw_inv) * Ug_g).reshape(shape), True)
    A12 = Uf(evecs, ((Nw_inv) * Ug_g).reshape(shape), True)
    A20 = sys_basis.adjoint((Nw_inv) * Us(x0_recon, False).flatten())
    A21 = sys_basis.adjoint((Nw_inv) * Uf(evecs, x1_recon, False))
    A22 = ((1 / G).reshape(sys_basis.g_shape) * x2_recon.reshape(sys_basis.g_shape)
           + sys_basis.adjoint((Nw_inv) * Ug_g))

    Ax0 = Ax0 + A02
    Ax1 = Ax1 + A12
    Ax2 = A20 + A21 + A22

    return np.concatenate([Ax0.real.flatten(), Ax0.imag.flatten(),
                           Ax1.real.flatten(), Ax2.real.flatten()])


def _construct_A_beam(x, x0_recon, x1_recon, S, Nw_inv, F, evecs, rfft_len,
                      rfft_shape, f_len, shape, sys_basis, G, beam):
    """``construct_A`` with the sky passed through the beam.

    Written as "model, weight, adjoint" rather than block by block: the
    beamed sky ``B (Us s + Uf f)`` is formed once, the systematic added
    outside the beam, the weighted residual pulled back through each block's
    adjoint. Two beam applications per product instead of eight.
    """
    sky = beam.apply(Us(x0_recon, False).flatten()
                     + Uf(evecs, x1_recon, False))
    if sys_basis is not None:
        g_off = 2 * rfft_len + f_len
        x2_recon = x[g_off:g_off + sys_basis.n_params].real
        sky = sky + sys_basis.apply(x2_recon)

    r = Nw_inv * sky
    rB = beam.apply(r).reshape(shape)

    Ax0 = (((1 / S) * x0_recon.flatten()).reshape(rfft_shape)
           + Us(rB, True))
    Ax1 = (1 / F) * x1_recon + Uf(evecs, rB, True)
    out = [Ax0.real.flatten(), Ax0.imag.flatten(), Ax1.real.flatten()]

    if sys_basis is not None:
        Ax2 = ((1 / G).reshape(sys_basis.g_shape)
               * x2_recon.reshape(sys_basis.g_shape)
               + sys_basis.adjoint(r))
        out.append(Ax2.real.flatten())

    return np.concatenate(out)


def construct_preconditioner(S, N_inv_scalar, F, evecs, rfft_len, rfft_shape,
                             f_len, f_shape, shape, sys_basis=None, G=None,
                             beam=None, fg_block=None):
    """Build a block-diagonal preconditioner ``M^-1`` for ``Ax = b``.

    Drops the off-diagonal (signal-foreground coupling) blocks of A, keeping::

        M = [S^{-1} + N^{-1},   0                        ]
            [0,                 F^{-1} + U_f^T N^{-1} U_f]

    With uniform noise and ortho FFTs (unitary U_s):

    - Signal block inverse: ``1 / (1/S[k] + N_inv)`` -- diagonal, per mode
    - Foreground block inverse: ``(diag(1/F) + N_inv * evecs @ evecs.T)^-1``
      -- a small dense matrix, inverted once

    Parameters
    ----------
    S : 1D array, length Nx*Ny*Nz
        Signal covariance per voxel (full cube).
    N_inv_scalar : float
        1/sigma^2, the inverse noise variance (scalar for uniform noise).
    F : 1D array, length n_modes
        Diagonal foreground covariance (per foreground mode).
    evecs : 2D array, shape (n_modes, n_freq)
        Foreground basis, orthonormal rows.
    rfft_len, rfft_shape, f_len, f_shape, shape :
        Shape metadata for unpacking the solution vector x.
    beam : Beam, optional
        With a beam the signal block becomes ``1 / (1/S + N_inv <|B|^2>)``,
        where ``<|B|^2>`` is the band average per ``k_perp`` -- the exact
        diagonal of ``Us^T B^2 Us``, dropping the ``k_parallel`` mixing a
        chromatic beam introduces. The foreground block stays *exact* for
        uniform noise: ``Uf^T B^2 Uf`` is diagonal in ``k_perp``, so it is
        one small ``n_modes x n_modes`` matrix per transverse mode, inverted
        once and applied in transverse Fourier space.
    fg_block : MaskedForegroundBlock, optional
        With a beam *and* a mask, the uniform-coverage foreground block above
        is not enough -- on the live grid LGMRES does not converge with it
        (see :class:`MaskedForegroundBlock`). Supplying one replaces the
        foreground block for its stiff low-order modes with an exact-mask
        factorisation; the remaining modes keep the Fourier block.

    Returns
    -------
    apply : callable
        Applies ``M^-1`` to a vector x.
    """
    # Signal block: truncate S from full fftn shape to rfft shape.
    S_rfft = S.reshape(shape)[:, :, :shape[2] // 2 + 1].flatten()
    n_modes = evecs.shape[0]
    if beam is None:
        M00_inv = 1.0 / (1.0 / S_rfft + N_inv_scalar)

        # Foreground block: small (n_modes x n_modes) matrix, invert once.
        M11 = np.diag(1.0 / F[:n_modes]) + N_inv_scalar * (evecs @ evecs.T)
        M11_inv = np.linalg.inv(M11)

        def apply_f(f):
            return (f @ M11_inv.T).flatten()
    else:
        M00_inv = 1.0 / (1.0 / S_rfft
                         + N_inv_scalar * beam.signal_diagonal(rfft_shape))

        # One block per transverse mode, over the modes the dense
        # exact-mask block (if any) does not already handle.
        lo = 0 if fg_block is None else fg_block.n_dense
        hi = np.arange(lo, n_modes)
        if len(hi):
            e_hi = evecs[hi]
            b2 = beam.full_transfer() ** 2                      # (Nx, Ny, Nz)
            M11 = (np.diag(1.0 / F[hi])
                   + N_inv_scalar * np.einsum('mz,xyz,nz->xymn', e_hi, b2, e_hi))
            M11_inv = np.linalg.inv(M11)                  # (Nx, Ny, n_hi, n_hi)

        def apply_f(f):
            out = np.empty_like(f)
            if lo:
                out[:, :, :lo] = fg_block.solve(f[:, :, :lo])
            if len(hi):
                fk = fft.fft2(f[:, :, lo:], axes=(0, 1), norm='ortho')
                out[:, :, lo:] = fft.ifft2((M11_inv @ fk[..., None])[..., 0],
                                           axes=(0, 1), norm='ortho').real
            return out.flatten()

    # Systematic block: n_params x n_params, and n_params is a handful. The
    # Gram matrix comes from SystematicBasis.gram(), which factorises it
    # exactly rather than assembling it over every voxel.
    if sys_basis is not None:
        M22 = (np.diag(1.0 / np.asarray(G).ravel())
               + N_inv_scalar * sys_basis.gram())
        M22_inv = np.linalg.inv(M22)
        g_off = 2 * rfft_len + f_len
        n_g = sys_basis.n_params

    def apply(x):
        re_s = M00_inv * x[0:rfft_len]
        im_s = M00_inv * x[rfft_len:2 * rfft_len]
        f = x[2 * rfft_len:2 * rfft_len + f_len].reshape(f_shape)
        f_out = apply_f(f)
        if sys_basis is None:
            return np.concatenate([re_s, im_s, f_out])
        g_out = M22_inv @ x[g_off:g_off + n_g]
        return np.concatenate([re_s, im_s, f_out, g_out])

    return apply


class MaskedForegroundBlock:
    """Exact-mask preconditioner for the stiff foreground modes under a beam.

    The foreground block of ``A`` is

        A_ff = F^-1 + N^-1 Uf^T B W B Uf

    With no beam it is local to each pixel, and with a beam but no mask it is
    diagonal in ``k_perp``; the default preconditioner exploits one or the
    other. With both it is neither, and the prior precision of the low-order
    modes is ~1e9 below the data's (``N^-1 F_0 ~ 1.4e9`` on the live grid), so
    any coverage the preconditioner gets wrong becomes a continuum of
    eigenvalues across nine decades. Measured on a sub-box of the real mask
    (``docs/STATUS.md``), condition number of the preconditioned block:
    uniform-coverage Fourier 1.6e6; exact mask, achromatic beam 790; exact
    mask, beam in 5 channel groups 1.5.

    So this forms the low-order modes' block exactly in pixel space:

        K = sum_g B_g diag(Omega_g) B_g,  Omega_g[p, a, b] = sum_{nu in g} e_a e_b w

    with ``B_g`` the beam averaged over a group of channels -- chromaticity
    matters here, achromatic is not good enough -- and Cholesky-factorises
    ``N^-1 K + F^-1``. ``K`` is fixed for a run (beam, mask and basis do not
    change); only ``F`` does, and a factor built with a stale ``F`` still
    preconditions well (20 vs 14 iterations with ``F`` off by 20%), so
    :meth:`factor` need only be called every so often.

    Memory: ``K`` and its factor are each ``(Npix * n_dense)^2`` doubles --
    0.7 GB apiece at ``n_dense=3`` on the live grid, 2.9 GB at 6.

    Parameters
    ----------
    beam : Beam
    w : array, ``beam.shape`` or flat
        The weights the operator uses (the mask).
    evecs : array, ``(n_modes, Nz)``
    N_inv : float
    n_dense : int
        Number of lowest-order modes treated exactly. The rest keep the
        Fourier block. 3 is enough on the live grid (``N^-1 F_m`` is ~1e9,
        1e5, 1e2 for modes 0-2 and ~10 above).
    n_groups : int
        Channel groups for the chromatic beam. ``Nz`` is exact.
    """

    def __init__(self, beam, w, evecs, N_inv, n_dense=3, n_groups=5):
        nx, ny, nz = beam.shape
        self.shape = beam.shape
        self.n_dense = int(min(n_dense, evecs.shape[0]))
        self.N_inv = float(N_inv)
        npix = nx * ny
        md = self.n_dense
        w = np.reshape(w, (npix, nz))
        ix, iy = np.unravel_index(np.arange(npix), (nx, ny))
        dx = (ix[:, None] - ix[None, :]) % nx
        dy = (iy[:, None] - iy[None, :]) % ny
        K = np.zeros((npix, md, npix, md))
        for grp in np.array_split(np.arange(nz), int(n_groups)):
            kern = fft.irfftn(beam.transfer[:, :, grp].mean(axis=2),
                              s=(nx, ny), axes=(0, 1))
            Bg = kern[dx, dy]                          # symmetric circulant
            e = evecs[:md, grp]
            Om = np.einsum('az,bz,pz->pab', e, e, w[:, grp])
            for a in range(md):
                for b in range(a, md):
                    blk = (Bg * Om[:, a, b][None, :]) @ Bg
                    K[:, a, :, b] += blk
                    if b != a:
                        K[:, b, :, a] += blk
        self.K = self.N_inv * K.reshape(npix * md, npix * md)
        self._inv = None

    def factor(self, F):
        """(Re)build the inverse with the current foreground covariance ``F``.

        Forms the explicit inverse (Cholesky then ``dpotri``) rather than
        keeping the factor: it is applied ~25 times per solve and reused
        across many solves, and a dense matvec is ~4x faster than a pair of
        triangular solves. Accuracy is ample for a preconditioner.
        """
        from scipy.linalg import lapack
        npix = self.shape[0] * self.shape[1]
        M = np.asfortranarray(self.K)                 # copy, Fortran order
        M[np.diag_indices_from(M)] += np.tile(
            1.0 / np.asarray(F)[:self.n_dense], npix)
        c, info = lapack.dpotrf(M, lower=1, overwrite_a=1, clean=0)
        if info != 0:
            raise np.linalg.LinAlgError(f'dpotrf failed (info={info})')
        inv, info = lapack.dpotri(c, lower=1, overwrite_c=1)
        if info != 0:
            raise np.linalg.LinAlgError(f'dpotri failed (info={info})')
        # dpotri fills the lower triangle only.
        inv = np.tril(inv)
        inv += np.tril(inv, -1).T
        self._inv = np.ascontiguousarray(inv)

    def solve(self, f_lo):
        """Apply the inverse to the low-order amplitudes, ``(Nx, Ny, n_dense)``."""
        if self._inv is None:
            raise RuntimeError('call factor(F) first')
        return (self._inv @ np.reshape(f_lo, -1)).reshape(np.shape(f_lo))


def construct_b(S, N_inv, F, w, s_mean, f_mean, evecs, data_cube, ws, wf, wd,
                shape, sys_basis=None, G=None, wg=None, g_mean=None,
                beam=None, weighted_data=None):
    """Build the RHS of ``Ax = b``.

    Combines the data term, the prior-mean terms, and the omega (``ws``,
    ``wf``, ``wd``) terms that supply the stochastic scatter. Setting the
    omegas to zero reduces the solve to the MAP/Wiener-filter solution.

    With ``sys_basis`` supplied, ``wg`` is the systematic's omega draw --
    standard normal, shape ``sys_basis.g_shape``. ``g_mean`` defaults to zero:
    a systematic has no known sign, so the prior is centred on "absent" and
    the data has to argue it into existence.

    With ``beam`` supplied, the weighted data and its omega term reach the
    signal and foreground rows through ``B`` (``B^T = B``), and the
    systematic row directly.

    ``weighted_data``, if given, replaces ``N^-1 w d + (N^-1 w)^1/2 wd`` --
    the weighted data plus its noise draw -- for a noise model that is not
    diagonal (:meth:`MarginalNoise.weighted_data`).
    """
    rfft_shape = np.shape(s_mean)

    # Truncate S from full fftn shape (Nx,Ny,Nz) to rfft shape (Nx,Ny,Nz//2+1)
    S = S.reshape(shape)[:, :, :shape[2] // 2 + 1].flatten()

    r = (weighted_data if weighted_data is not None
         else (N_inv) * w * data_cube.flatten() + (np.sqrt(N_inv)) * np.sqrt(w) * wd)

    if beam is not None:
        return _construct_b_beam(S, F, s_mean, f_mean, evecs, r, ws, wf,
                                 shape, rfft_shape, sys_basis, G, wg, g_mean,
                                 beam)

    b0 = (Us(r.reshape(shape), True)
          + ((1 / S) * s_mean.flatten()).reshape(rfft_shape)
          + ((1 / np.sqrt(S)) * ws).reshape(rfft_shape))

    b1 = (Uf(evecs, r.reshape(shape), True)
          + ((1 / F) * f_mean)
          + ((1 / np.sqrt(F)) * wf))

    if sys_basis is None:
        return np.concatenate([b0.real.flatten(), b0.imag.flatten(),
                               b1.real.flatten()])

    G = np.asarray(G).reshape(sys_basis.g_shape)
    if g_mean is None:
        g_mean = np.zeros(sys_basis.g_shape)
    if wg is None:
        wg = np.zeros(sys_basis.g_shape)

    b2 = (sys_basis.adjoint(r)
          + (1 / G) * np.asarray(g_mean).reshape(sys_basis.g_shape)
          + (1 / np.sqrt(G)) * np.asarray(wg).reshape(sys_basis.g_shape))

    return np.concatenate([b0.real.flatten(), b0.imag.flatten(),
                           b1.real.flatten(), b2.real.flatten()])


def _construct_b_beam(S, F, s_mean, f_mean, evecs, r, ws, wf, shape,
                      rfft_shape, sys_basis, G, wg, g_mean, beam):
    """``construct_b`` with the sky passed through the beam. ``r`` is the
    weighted data plus its noise draw."""
    rB = beam.apply(r).reshape(shape)

    b0 = (Us(rB, True)
          + ((1 / S) * s_mean.flatten()).reshape(rfft_shape)
          + ((1 / np.sqrt(S)) * ws).reshape(rfft_shape))
    b1 = (Uf(evecs, rB, True)
          + ((1 / F) * f_mean)
          + ((1 / np.sqrt(F)) * wf))
    out = [b0.real.flatten(), b0.imag.flatten(), b1.real.flatten()]

    if sys_basis is not None:
        G = np.asarray(G).reshape(sys_basis.g_shape)
        if g_mean is None:
            g_mean = np.zeros(sys_basis.g_shape)
        if wg is None:
            wg = np.zeros(sys_basis.g_shape)
        b2 = (sys_basis.adjoint(r)
              + (1 / G) * np.asarray(g_mean).reshape(sys_basis.g_shape)
              + (1 / np.sqrt(G)) * np.asarray(wg).reshape(sys_basis.g_shape))
        out.append(b2.real.flatten())

    return np.concatenate(out)


class MarginalNoise:
    """Noise weights with a systematic block integrated out.

    Sampling ``(s, f, g)`` jointly leaves a near-null direction wherever a
    template can be traded against ``B Us s`` or ``B Uf f``: the solver
    barely moves along it (docs/STATUS.md, finding 9 of "The beam in the
    MODEL"). Integrating ``g`` out instead treats the systematic as
    correlated noise, ``N' = N_w + Ug G Ug^T``, solves for ``(s, f)`` with
    it, and then draws ``g | s, f, d`` exactly. The joint posterior is the
    same; the troublesome direction becomes ``n_params`` outlying
    eigenvalues, which a Krylov solver absorbs in a few iterations.

    With masked voxels ``N_w^-1 = w N^-1`` has zeros, so ``N'`` is never
    formed; its inverse is, by Woodbury, a rank-``n_params`` update::

        N'^-1 = W - W Ug P^-1 Ug^T W,    P = G^-1 + Ug^T W Ug,  W = w N^-1

    ``obj * y`` applies ``N'^-1`` to a flattened cube, so an instance stands
    in for the weight vector ``Nw_inv`` in :func:`construct_A`.

    Parameters
    ----------
    Nw_inv : array, length Nx*Ny*Nz
        ``w N^-1``.
    sys_basis : SystematicBasis or TemplateBasis
    G : array, length ``n_params``
        Prior variance of each amplitude. The prior mean is zero.
    """

    # Make ``array * obj`` defer to __rmul__ instead of broadcasting obj.
    __array_ufunc__ = None

    def __init__(self, Nw_inv, sys_basis, G):
        self.W = np.asarray(Nw_inv, dtype=float).ravel()
        self.basis = sys_basis
        self.G = np.asarray(G, dtype=float).ravel()
        eye = np.eye(sys_basis.n_params)
        # Rows are W Ug e_i, so WU @ y = Ug^T W y and WU.T @ a = W Ug a.
        self.WU = np.array([self.W * sys_basis.apply(e) for e in eye])
        P = np.diag(1 / self.G) + np.array(
            [sys_basis.adjoint(row).ravel() for row in self.WU])
        self.P = 0.5 * (P + P.T)
        self.chol = np.linalg.cholesky(self.P)

    def _solve_P(self, v):
        return np.linalg.solve(self.P, v)

    def __mul__(self, y):
        y = np.asarray(y).ravel()
        return self.W * y - self.WU.T @ self._solve_P(self.WU @ y)

    __rmul__ = __mul__

    def weighted_data(self, data_cube, wd, wg):
        """``N'^-1 d`` plus a draw with covariance ``N'^-1``.

        ``wd`` (one per voxel) and ``wg`` (one per amplitude) are standard
        normal. With ``z = W d + W^1/2 wd``, the returned
        ``z - W Ug P^-1 (Ug^T z - G^-1/2 wg)`` has mean ``N'^-1 d`` and noise
        covariance ``W - W Ug P^-1 Ug^T W = N'^-1``.
        """
        z = (self.W * np.asarray(data_cube).ravel()
             + np.sqrt(self.W) * np.asarray(wd).ravel())
        v = (self.basis.adjoint(z).ravel()
             - np.asarray(wg).ravel() / np.sqrt(self.G))
        return z - self.WU.T @ self._solve_P(v)

    def preconditioner(self, M1_apply, C, adjoint_weight):
        """Correct a preconditioner for the rank-``n_params`` update.

        With ``g`` integrated out, ``A = A0 - C P^-1 (w C)^T``, where ``A0``
        is the system with plain ``W`` weights -- what ``M1_apply`` (from
        :func:`construct_preconditioner`) approximates the inverse of. ``C``
        has rows ``A_x^T W Ug e_i`` as :func:`construct_b` forms them
        (``A_x`` the model operator on ``(s, f)``). The packed real FFT
        makes ``A`` non-symmetric: the true adjoint of the ``s`` block counts
        every ``k_z`` strictly between the DC and Nyquist planes twice, and
        ``w = adjoint_weight`` carries that factor (see
        :func:`rfft_adjoint_weight`).

        ``M1`` alone leaves ``n_params`` directions with near-zero
        eigenvalues, which a warm-started solve does not resolve. Woodbury
        applied to ``M1`` cancels them::

            M^-1 r = u + Y K^-1 (w C) u,   u = M1 r,  Y = M1 C^T,
            K = P - (w C) Y

        exact when ``M1 = A0^-1``.

        Parameters
        ----------
        M1_apply : callable
            Approximation to ``A0^-1``.
        C : array, shape (n_params, len(x))
            Rows ``A_x^T W Ug e_i``; independent of ``S`` and ``F``.
        adjoint_weight : array, length len(x)
        """
        CR = C * adjoint_weight
        Y = np.array([M1_apply(c) for c in C])
        K = self.P - CR @ Y.T

        def apply(r):
            u = M1_apply(r)
            return u + Y.T @ np.linalg.solve(K, CR @ u)
        return apply

    def draw(self, resid, eta):
        """A draw of ``g | s, f, d``: ``N(P^-1 Ug^T W resid, P^-1)``.

        ``resid`` is the data minus the (beamed) sky model, flattened;
        ``eta`` is standard normal, length ``n_params``. Returns ``g`` in
        ``sys_basis.g_shape``.
        """
        mean = self._solve_P(self.WU @ np.asarray(resid).ravel())
        g = mean + np.linalg.solve(self.chol.T, np.asarray(eta).ravel())
        return g.reshape(self.basis.g_shape)


def rfft_adjoint_weight(rfft_shape, f_len):
    """Weights turning the packed ``rfftn`` rows into true adjoint rows.

    ``x`` packs ``s`` as the real and imaginary parts of its half spectrum.
    Every ``k_z`` strictly between the DC and Nyquist planes stands for
    itself and its mirror, so the adjoint of ``irfftn`` in that packing is
    ``rfftn`` times 2 there and times 1 on the two edge planes (only DC when
    ``Nz`` is odd). The foreground block is unweighted.
    """
    nzh = rfft_shape[2]
    c = np.full(rfft_shape, 2.0)
    c[:, :, 0] = 1.0
    # irfftn's default length is 2 * (nzh - 1): an even Nz with a Nyquist plane.
    c[:, :, -1] = 1.0
    return np.concatenate([c.ravel(), c.ravel(), np.ones(f_len)])
