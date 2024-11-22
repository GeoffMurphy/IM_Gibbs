import numpy as np
import numpy.fft as fft
import time
import random


def construct_Uf(n_modes, shape, freqs):
    """
    Make the U_f operator
    """
    
    beta = -2.4
    ref_freq = 130

    row = []
    for ll in range(0,shape[0]):
        if (ll > n_modes-1):
            row.append( 0*freqs)
        else:
            row.append( ((freqs/ref_freq)**beta) * (np.log10(freqs/ref_freq))**(ll+1))

    '''for ll in range(0,shape[0]):
       row.append( ((freqs/ref_freq)**beta) * (np.log10(freqs/ref_freq))**(ll+1))'''

    Uf = np.broadcast_to(np.array(row),(shape[0],)+np.array(row).shape)
    return Uf[:,:n_modes,:][0]


def Uf(Uf_op, amps, transpose):
    """
    Apply the Uf operator
    """

    if transpose == True:
        gn_fit = (amps@np.linalg.pinv(Uf_op))
    else:
        gn_fit = (amps@Uf_op).flatten()

    #gn_fit[:,:,3:] = 0
    
    return gn_fit


def Us(s, transpose):
    """
    Apply U_s (no fft scaling)
    """

    if transpose:
        return fft.rfftn(s)
    else:
        return fft.irfftn(s)


def construct_A(x0_recon, x1_recon, S, Nw, F, w, Uf_op):
    """
    Evaluate the LHS of Ax = b
    """

    # Track some shapes
    length = np.shape(x0_recon)[0]
    shape = (length,length,length)
    rfft_shape = np.shape(x0_recon)
    rfft_len = len(x0_recon.flatten())

    # Take only the first few S covariance elements, to match the length of an rfft vector (?)
    S = S[0:rfft_len]

    # Invert noise cov, and replace infinities with zero, if there are any
    Nw_inv = np.nan_to_num(1/Nw,posinf=0)

    # Take only the real part for the foreground amplitudes
    x0_recon = x0_recon
    x1_recon = x1_recon.real


    
    Ax0 = ( ((1/S)*x0_recon.flatten()).reshape(rfft_shape) + Us( ((Nw_inv)*Us(x0_recon, False).flatten()).reshape(shape) , True) +
            Us(((Nw_inv)*Uf(Uf_op, x1_recon, False)).reshape(shape), True) )

    Ax1 = ( Uf(Uf_op, ((Nw_inv)*Us(x0_recon, False).flatten()).reshape(shape), True ) +
           (1/F)*x1_recon + Uf(Uf_op, ((Nw_inv)*Uf(Uf_op, x1_recon, False)).reshape(shape), True) )


    return np.concatenate([Ax0.flatten(), Ax1.flatten()])
    
def construct_b(S, N, F, w, s_mean, f_mean, Uf_op, data_cube, ws, wf, wd):
    """
    RHS of the linear equation
    """
    
    # Invert noise cov, and replace infinities with zero, if there are any
    N_inv = np.nan_to_num(1/N,posinf=0)

    # Track some shapes
    length = np.shape(data_cube)[0]
    shape = (length,length,length)
    rfft_shape = np.shape(s_mean)
    rfft_len = len(s_mean.flatten())

    # Take only the first few S covariance elements, to match the length of an rfft vector (?)
    S = S[0:rfft_len]
    
    b0 = Us(((N_inv)*w*data_cube.flatten() + (np.sqrt(N_inv))*np.sqrt(w)*wd ).reshape(shape), True) + ((1/S)*s_mean.flatten()).reshape(rfft_shape) + ((1/np.sqrt(S))*ws).reshape(rfft_shape)

    b1 = Uf(Uf_op, ((N_inv)*w*data_cube.flatten() + (np.sqrt(N_inv))*np.sqrt(w)*wd ).reshape(shape),True) + ((1/F)*f_mean) + ((1/np.sqrt(F))*wf)

    #return b0
    #return b1
    return np.concatenate([b0.flatten(), b1.flatten()])















