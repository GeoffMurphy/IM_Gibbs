"""Conditional samplers for the signal and foreground covariances.

These are the two Gibbs steps that follow the constrained realisation. Given
the current signal sample, :func:`signal_covariance_sampler` draws a new P(k)
from its inverse-gamma conditional in each k-bin; given the current foreground
amplitudes, :func:`foreground_covariance_sampler` draws a new F from an
inverse-Wishart.
"""

import numpy as np
from scipy.stats import invgamma, invwishart


# --------------------------------------------------------------------------
# Signal covariance
# --------------------------------------------------------------------------

def signal_covariance_sampler(s, k, alpha_offset=0):
    """Draw the diagonal signal covariance S from its conditional posterior.

    Parameters
    ----------
    s : array
        21cm coefficients in comoving Fourier space.
    k : array
        Corresponding absolute wavenumbers, same length as ``s``. Modes
        sharing a k value are pooled into one bin.
    alpha_offset : float
        Added to the default ``alpha = N/2 - 1``.

    Returns
    -------
    Pk : array, shape (len(s),)
        The drawn P(k) broadcast back onto every mode.
    Pk_out : list, one entry per bin
        The per-bin draw.
    """
    assert len(s) == len(k), 'Both arrays must be of the same length.'

    # Sort by k so equal-k modes are contiguous, then chunk on the boundaries.
    sorted_indices = np.argsort(k)
    shuffled_k = k[sorted_indices]
    shuffled_s = s[sorted_indices]

    chunked_k, chunked_s = chunk_arrays(shuffled_k, shuffled_s)
    assert len(chunked_k) == len(chunked_s), 'Both lists must be of the same length.'
    Nmodes = len(chunked_k)
    sigmaSq = np.zeros(Nmodes)
    Nkvec = np.zeros(Nmodes)
    for i in range(Nmodes):
        Nkvec[i] = len(chunked_k[i])
        assert len(chunked_k[i]) > 2, \
            'The number of modes of the same k must be greater than 2.'
        sigmaSq[i] = np.sum(np.abs(chunked_s[i])**2)

    PkSamples = samplingPk(Nkvec, sigmaSq, alpha_offset=alpha_offset)
    Pk_out = PkSamples

    Pk_chuncked = []
    for i in range(Nmodes):
        Pk_chuncked.append(PkSamples[i] * np.ones(len(chunked_k[i])))

    Pk_sorted = recover_from_chunking(Pk_chuncked)
    Pk = recover_from_sorting(Pk_sorted, sorted_indices)

    return Pk, Pk_out


def samplingPk(Nvec, sigma2vec, alpha_offset=0):
    """Draw one inverse-gamma sample per k-bin.

    Parameters
    ----------
    Nvec : array
        ``Nvec[i]`` is the number of modes in bin i.
    sigma2vec : array
        ``sum |s|^2`` within bin i.
    alpha_offset : float
        Added to the default ``alpha = N/2 - 1``.
    """
    assert len(Nvec) == len(sigma2vec)
    alpha = (0.5 * Nvec - 1) + alpha_offset

    beta = 0.5 * sigma2vec
    dim = len(Nvec)
    result = []
    for i in range(dim):
        result.append(invgamma.rvs(a=alpha[i], scale=beta[i]))

    return result


def chunk_arrays(first_array, second_array):
    """Split two parallel sorted arrays wherever the first one changes value."""
    assert len(first_array) == len(second_array), \
        'Both arrays must be of the same length.'

    chunked_first_array = []
    chunked_second_array = []
    current_start = 0

    for i in range(1, len(first_array)):
        if first_array[i] != first_array[current_start]:
            chunked_first_array.append(first_array[current_start:i])
            chunked_second_array.append(second_array[current_start:i])
            current_start = i

    chunked_first_array.append(first_array[current_start:])
    chunked_second_array.append(second_array[current_start:])

    return chunked_first_array, chunked_second_array


def recover_from_chunking(chunked_array):
    """Flatten a list of chunks back into one array."""
    return np.concatenate(chunked_array)


def recover_from_sorting(sorted_array, sorted_indices):
    """Undo the argsort applied by :func:`signal_covariance_sampler`."""
    original_array = np.empty_like(sorted_array)
    original_array[sorted_indices] = sorted_array
    return original_array


# --------------------------------------------------------------------------
# Foreground covariance
# --------------------------------------------------------------------------

def foreground_covariance_sampler(fmat):
    """Draw the foreground covariance F from an inverse-Wishart conditional.

    Parameters
    ----------
    fmat : array, shape (Npix, Nmodes)
        Mean-subtracted foreground coefficients. Pass ``f - f_mean``: at
        initialisation ``f == f_mean`` and the scatter matrix would be
        singular, but inside the sampling loop the drawn f has moved away
        from the mean.

    Returns
    -------
    cov_sample : array, shape (Nmodes, Nmodes)
    """
    Npix = fmat.shape[0]
    Nmodes = fmat.shape[1]

    p = Nmodes                # dimension of the scale matrix
    nu = Npix                 # degrees of freedom, must be >= p
    assert nu >= p, ('Degrees of freedom must be greater than or equal to '
                     'the dimension of the scale matrix.')

    Psi = np.zeros((p, p))
    for i in range(Npix):
        fi = fmat[i, :]
        Psi += np.outer(fi, fi)

    return invwishart.rvs(df=nu, scale=Psi)
