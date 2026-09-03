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
                f_shape, shape):
    """Evaluate the LHS of ``Ax = b`` as a matrix-vector product."""
    x0_recon = (x[0:rfft_len].reshape(rfft_shape)
                + x[rfft_len:2 * rfft_len].reshape(rfft_shape) * 1j)
    x1_recon = x[2 * rfft_len:2 * rfft_len + f_len].reshape(f_shape)

    # Truncate S from full fftn shape (Nx,Ny,Nz) to rfft shape (Nx,Ny,Nz//2+1)
    S = S.reshape(shape)[:, :, :shape[2] // 2 + 1].flatten()

    # Take only the real part for the foreground amplitudes
    x1_recon = x1_recon.real

    A00 = (((1 / S) * x0_recon.flatten()).reshape(rfft_shape)
           + Us(((Nw_inv) * Us(x0_recon, False).flatten()).reshape(shape), True))

    A01 = Us(((Nw_inv) * Uf(evecs, x1_recon, False)).reshape(shape), True)

    A10 = Uf(evecs, ((Nw_inv) * Us(x0_recon, False).flatten()).reshape(shape), True)

    A11 = ((1 / F) * x1_recon
           + Uf(evecs, ((Nw_inv) * Uf(evecs, x1_recon, False)).reshape(shape), True))

    Ax0 = A00 + A01
    Ax1 = A10 + A11

    return np.concatenate([Ax0.real.flatten(), Ax0.imag.flatten(),
                           Ax1.real.flatten()])


def construct_preconditioner(S, N_inv_scalar, F, evecs, rfft_len, rfft_shape,
                             f_len, f_shape, shape):
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

    Returns
    -------
    apply : callable
        Applies ``M^-1`` to a vector x.
    """
    # Signal block: truncate S from full fftn shape to rfft shape.
    S_rfft = S.reshape(shape)[:, :, :shape[2] // 2 + 1].flatten()
    M00_inv = 1.0 / (1.0 / S_rfft + N_inv_scalar)

    # Foreground block: small (n_modes x n_modes) matrix, invert once.
    n_modes = evecs.shape[0]
    M11 = np.diag(1.0 / F[:n_modes]) + N_inv_scalar * (evecs @ evecs.T)
    M11_inv = np.linalg.inv(M11)

    def apply(x):
        re_s = M00_inv * x[0:rfft_len]
        im_s = M00_inv * x[rfft_len:2 * rfft_len]
        f = x[2 * rfft_len:2 * rfft_len + f_len].reshape(f_shape)
        f_out = (f @ M11_inv.T).flatten()
        return np.concatenate([re_s, im_s, f_out])

    return apply


def construct_b(S, N_inv, F, w, s_mean, f_mean, evecs, data_cube, ws, wf, wd,
                shape):
    """Build the RHS of ``Ax = b``.

    Combines the data term, the prior-mean terms, and the omega (``ws``,
    ``wf``, ``wd``) terms that supply the stochastic scatter. Setting the
    omegas to zero reduces the solve to the MAP/Wiener-filter solution.
    """
    rfft_shape = np.shape(s_mean)

    # Truncate S from full fftn shape (Nx,Ny,Nz) to rfft shape (Nx,Ny,Nz//2+1)
    S = S.reshape(shape)[:, :, :shape[2] // 2 + 1].flatten()

    b0 = (Us(((N_inv) * w * data_cube.flatten()
              + (np.sqrt(N_inv)) * np.sqrt(w) * wd).reshape(shape), True)
          + ((1 / S) * s_mean.flatten()).reshape(rfft_shape)
          + ((1 / np.sqrt(S)) * ws).reshape(rfft_shape))

    b1 = (Uf(evecs, ((N_inv) * w * data_cube.flatten()
                     + (np.sqrt(N_inv)) * np.sqrt(w) * wd).reshape(shape), True)
          + ((1 / F) * f_mean)
          + ((1 / np.sqrt(F)) * wf))

    return np.concatenate([b0.real.flatten(), b0.imag.flatten(),
                           b1.real.flatten()])
