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

    Pk_chuncked = []
    for i in range(Nmodes):
        Pk_chuncked.append(PkSamples[i] * np.ones(len(chunked_k[i])))
    
    Pk_sorted = recover_from_chunking(Pk_chuncked)
    Pk = recover_from_sorting(Pk_sorted, sorted_indices)

    return Pk, PkSamples


def samplingPk(Nvec, sigma2vec):
    """
    Both Nvec and sigma2vec are vectors of the same length.
    Nvec: 
        len(Nvec): number of different distributions/statistical parameters. 
        Nvec[i]: number of randam variables of characterized by the i-th statistical parameter

    sigma2vec: vector of parameters.
    """
    assert len(Nvec) == len(sigma2vec)
    alpha = 0.5 * Nvec - 1
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

    # Draw a sample from the inverse Wishart distribution
    cov_sample = invwishart.rvs(df=nu, scale=Psi)
    return cov_sample

####################### S Binner #######################

def k_vecs(nsamp):
    """
    Define the k-vectors for each voxel (from fastbox)
    """
    
    scale_x, scale_y, scale_z = 2e3, 2e3, 2e3
    #nsamp = 128
    x = np.linspace(-0.5*scale_x, 0.5*scale_x, nsamp) # in Mpc
    y = np.linspace(-0.5*scale_y, 0.5*scale_y, nsamp) # in Mpc
    z = np.linspace(-0.5*scale_z, 0.5*scale_z, nsamp) # in Mpc
    
    Lx = x[-1] - x[0]
    Ly = y[-1] - y[0]
    Lz = z[-1] - z[0]
    
    N = nsamp
    kmin = 2.*np.pi/np.max([Lx, Ly, Lz])
    kmax = 2.*np.pi*np.sqrt(3.)*N/np.min([Lx, Ly, Lz])
    
    Kx = np.zeros((N,N,N))
    Ky = np.zeros((N,N,N))
    Kz = np.zeros((N,N,N))
    
    NN = ( N*fft.fftfreq(N, 1.) ).astype("i")
    
    for i in NN:
        Kx[i,:,:] = i
        Ky[:,i,:] = i
        Kz[:,:,i] = i
        
    #k = 2.*np.pi * np.sqrt(  (Kx/Lx)**2. + (Ky/Ly)**2. + (Kz/Lz)**2.)

    #---------------

    all_k = [] # Find the corresponding |k| for all Fourier modes

    Kx_flat = Kx.flatten()
    Ky_flat = Ky.flatten()
    Kz_flat = Kz.flatten()
        
    for qq in range(len(Kx.flatten())):
        all_k.append(np.sqrt(Kx_flat[qq]**2 + Ky_flat[qq]**2 + Kz_flat[qq]**2))
        
    return(np.array(all_k))



####################### Binner #######################


def binner(s_flat, nbins, cube_len):

    k = k_vecs(cube_len) # Get the k vectors

    k_min, k_max = min(k.flatten()), max(k.flatten()) # Bin range
    
    kbins = np.linspace(k_min, k_max, nbins+1) # Define the bins

    binned_s_out = [] # Groups all the s Fourier modes which fall in the same bin

    k_bins_out = [] # output the k bins corresponding to each individual s

    k_flat = k.flatten()

    idxs_track = [] # Track the indices.
    
    for i in range(0,len(kbins)-1):

        # Find the indices of the modes in a particular bin
        idxs = np.where(np.logical_and(k_flat >= kbins[i], k_flat < kbins[i+1]))

        # Append the modes grouped by the bin they're in
        binned_s_out.append( s_flat[idxs] )
        
        if len(s_flat[idxs]) < 3:
            print('< 3 modes in a bin')
            
        
        k_bins_out.append( np.zeros(len(s_flat[idxs])) + kbins[i] )

        idxs_track.append(idxs)
        
    return binned_s_out, k_bins_out, idxs_track


'''def broadcast_S(binned_s_sum, S_len, idxs_track):
    # Output the full S matrix
    S = np.ones(S_len) # fix

    for rr in range(0, len(idxs_track)):
        S[idxs_track[rr]] = binned_s_sum[rr]

    return S'''
    
