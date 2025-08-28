import numpy as np
import numpy.fft as fft
import time
import random

def Uf(evecs, amps, transpose):
    '''
    The foreground operator. 
    The forward operation, f.Uf, 
    returns a model foreground cube, while the
    reverse returns a vector of amplitudes.
    '''
    if transpose:
        gn_fit = amps @ evecs.T

    else:
        gn_fit = (amps @ evecs).flatten()
    
    return gn_fit.real


def Us(s, transpose):
    """
    The signal operator:
    The forward operation is an inverse rfft which
    returns a real field, while
    the reverse operation is a forward rfft which
    assumes a real-valued input.
    """
    
    if transpose:
        return fft.rfftn(s,norm='ortho') #* 1/np.sqrt(n)
    else:
        return fft.irfftn(s,norm='ortho') #* 1/np.sqrt(n) #/ n


def construct_A(x, S, Nw_inv, F, w, evecs, rfft_len, rfft_shape, f_len, f_shape):
    """
    Evaluate the LHS of Ax = b
    """

    x0_recon = x[0:rfft_len].reshape(rfft_shape) + x[rfft_len:2*rfft_len].reshape(rfft_shape)*1j
    x1_recon = x[2*rfft_len:2*rfft_len + f_len].reshape(f_shape) 
    
    # Track some shapes
    length = np.shape(x0_recon)[0]
    shape = (length,length,length)
 
    S = S[0:rfft_len]

    x0_recon = x0_recon
    x1_recon = x1_recon.real

    ### ---------------------- ###
    A00 = ((1/S)*x0_recon.flatten()).reshape(rfft_shape) + Us( ((Nw_inv)*Us(x0_recon,False).flatten()).reshape(shape) , True)

    A01 = Us(((Nw_inv)*Uf(evecs, x1_recon, False)).reshape(shape), True) 

    A10 = Uf(evecs, ((Nw_inv)*Us(x0_recon, False).flatten()).reshape(shape), True )

    A11 = ( (1/F)*x1_recon + Uf(evecs, ((Nw_inv)*Uf(evecs, x1_recon, False)).reshape(shape), True) )

    Ax0 = A00 + A01
    Ax1 = A10 + A11


    return np.concatenate([Ax0.real.flatten(), Ax0.imag.flatten(), Ax1.real.flatten()])
    
def construct_b(S, N_inv, F, w, s_mean, f_mean, evecs, data_cube, ws, wf, wd):
    """
    RHS of the linear equation
    """
    
    length = np.shape(data_cube)[0]
    shape = (length,length,length)
    rfft_shape = np.shape(s_mean)
    rfft_len = len(s_mean.flatten())

    S = S[0:rfft_len]

    b0 = Us(((N_inv)*w*data_cube.flatten() + (np.sqrt(N_inv))*np.sqrt(w)*wd ).reshape(shape), True) + ((1/S)*s_mean.flatten()).reshape(rfft_shape) + ((1/np.sqrt(S))*ws).reshape(rfft_shape)

    b1 = ( Uf(evecs, ((N_inv)*w*data_cube.flatten() + (np.sqrt(N_inv))*np.sqrt(w)*wd ).reshape(shape),True) + ((1/F)*f_mean) + ((1/np.sqrt(F))*wf) )

    return np.concatenate([b0.real.flatten(), b0.imag.flatten(), b1.real.flatten()])















