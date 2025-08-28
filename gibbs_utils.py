import numpy as np
from scipy.stats import invgamma, invwishart
import numpy.fft as fft

####################### Signal Covariance Sampler #######################
def signal_covariance_sampler(s, k):
    """
    s: vector of 21cm coefficients (in comoving Fourer space)
    k: vector of corresponding absolute wavenumbers

    output: 
    vector of the diagonal elements of the sample of the covariance matrix S.
    shape: (len(s),)
    """
    assert len(s) == len(k), "Both arrays must be of the same length."

    # Sort the k vector.
    sorted_indices = np.argsort(k)
    shuffled_k = k[sorted_indices]
    shuffled_s = s[sorted_indices]

    # Chunk the arrays
    chunked_k, chunked_s = chunk_arrays(shuffled_k, shuffled_s)
    assert len(chunked_k) == len(chunked_s), "Both lists must be of the same length."
    Nmodes = len(chunked_k)
    sigmaSq = np.zeros(Nmodes)
    Nkvec = np.zeros(Nmodes)
    for i in range(Nmodes):
        Nkvec[i] = len(chunked_k[i])
        assert len(chunked_k[i]) > 2, "The number of modes of the same k must be greater than 2."
        sigmaSq[i] = np.sum(np.abs(chunked_s[i])**2)

    PkSamples = samplingPk(Nkvec, sigmaSq)
    Pk_out = sigmaSq
    #Pk_out = PkSamples
    
    Pk_chuncked = []
    for i in range(Nmodes):
        Pk_chuncked.append(PkSamples[i] * np.ones(len(chunked_k[i])))
    
    Pk_sorted = recover_from_chunking(Pk_chuncked)
    Pk = recover_from_sorting(Pk_sorted, sorted_indices)

    return Pk, Pk_out


def samplingPk(Nvec, sigma2vec):
    """
    Both Nvec and sigma2vec are vectors of the same length.
    Nvec: 
        len(Nvec): number of different distributions/statistical parameters. 
        Nvec[i]: number of randam variables of characterized by the i-th statistical parameter

    sigma2vec: vector of parameters.
    """
    assert len(Nvec) == len(sigma2vec)
    alpha = (0.5 * Nvec - 1) # default    
    
    beta = 0.5 * sigma2vec
    dim = len(Nvec)
    result = []
    for i in range(dim):
        result.append(invgamma.rvs(a=alpha[i], scale=beta[i]))

    return result


def chunk_arrays(first_array, second_array):
    # Ensure both arrays have the same length
    assert len(first_array) == len(second_array), "Both arrays must be of the same length."

    # Initialize the list of chunks for both arrays
    chunked_first_array = []
    chunked_second_array = []
    
    # Initialize the current chunk start
    current_start = 0

    # Iterate over the array to find chunks
    for i in range(1, len(first_array)):
        if first_array[i] != first_array[current_start]:
            # If the current value is different from the chunk start, cut the chunk for both arrays
            chunked_first_array.append(first_array[current_start:i])
            chunked_second_array.append(second_array[current_start:i])
            current_start = i  # Update the chunk start to the current index

    # Add the last chunk
    chunked_first_array.append(first_array[current_start:])
    chunked_second_array.append(second_array[current_start:])

    return chunked_first_array, chunked_second_array


def recover_from_chunking(chunked_array):
    # Flatten the list of chunks into a single array
    original_array = np.concatenate(chunked_array)
    return original_array


def recover_from_sorting(sorted_array, sorted_indices):
    # Create an empty array of the same shape as sorted_array
    original_array = np.empty_like(sorted_array)
    
    # Place elements from sorted_array into their original positions
    original_array[sorted_indices] = sorted_array
    
    return original_array


####################### Foreground Covariance Sampler #######################

def foreground_covariance_sampler(fmat):
    """
    fmat: matrix of "foreground coefficients - Mean" in comoving Fourier space
    shape: (Npix, Nmodes)

    output:
    sample of the covariance matrix of the foreground coefficients.
    shape: (Nmodes, Nmodes)
    """
    Npix = fmat.shape[0]
    Nmodes = fmat.shape[1]

    p = Nmodes # Dimension of the scale matrix
    nu = Npix - p - 1 # Degrees of freedom, must be greater than or equal to dimension of the scale matrix
    assert nu >= p, "Degrees of freedom must be greater than or equal to the dimension of the scale matrix."

    Psi = np.zeros((p, p))

    # Compute the scale matrix
    for i in range(Npix):
        fi = fmat[i, :]
        Psi += np.outer(fi, fi)
    
    """Normalise"""
    Psi = Psi/Npix
    
    # Draw a sample from the inverse Wishart distribution
    cov_sample = invwishart.rvs(df=nu, scale=Psi)
    return cov_sample

#############################################################################

""" Binner for the S covariance sampler """

import fastbox
from fastbox.box import CosmoBox, default_cosmo
from fastbox.foregrounds import ForegroundModel
import time, sys

box = CosmoBox(cosmo=default_cosmo, box_scale=(1e3/0.678,1e3/0.678,1e3/0.678), nsamp=128, 
               redshift=0.39, realise_now=False)

def define_bins(cube,nbins):
    sig_k, sig_pk_true, sig_stddev, idxs = box.binned_power_spectrum(
    delta_x=np.fft.fftn(cube,norm='ortho'), nbins=nbins)

    return sig_k, idxs

def bin_it(cube, sig_k, idxs):
    binned_s = []
    k_bins = []

    unique_idxs = np.unique(idxs)
    
    for gg in range(0,len(unique_idxs)):
        where = np.where(idxs==unique_idxs[gg])
        sses = cube.flatten()[where]
        binned_s.append(sses)
    
        k_bins.append(np.ones(len(sses)) * sig_k[gg])

    return binned_s, k_bins