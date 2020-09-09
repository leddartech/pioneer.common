from scipy.interpolate import griddata
from scipy.fftpack import dct, idct

import numpy as np

def inpaintn(x,m=100, x0=None, alpha=2):
    """ This function interpolates the input (2-dimensional) image 'x' with missing values (can be NaN of Inf).  It is based on a recursive process
        where at each step the discrete cosine transform (dct) is performed of the residue, multiplied by some weights, and then the inverse dct is taken.
        The initial guess 'x0' for the interpolation can be provided by the user, otherwise it starts with a nearest neighbor filling.   

        Args
            INPUTS:
                x (numpy array) - is the image with missing elements (eiher np.nan or np.inf) from which you want to perform interpolation
                m (int) - is the number of iteration; default=100
                x0 (numpy array) - can be your initial guess; defaut=None
                alpha (float) - some input number used as a power scaling; default=2

            OUT:
                y (numpy array) - is the interpolated image wrt proposed method
    """

    sh = x.shape
    ids0 = np.isfinite(x)
    if ids0.all(): #Nothing to interpolate...
        return x

    # Smoothness paramaters:
    s0 = 3
    s1 = -6
    s = np.logspace(s0,s1,num=m)
   
    # Relaxation factor:
    rf = 2

    # Weight matrix, here we add some basis vectors to Lambda depending on original size of 'x':
    Lambda = np.zeros(sh, float)
    u0 = np.cos(np.pi*np.arange(0,sh[0]).reshape((sh[0],1))/sh[0])
    u1 = np.cos(np.pi*np.arange(0,sh[1]).reshape((1,sh[1]))/sh[1])
    Lambda = np.add(np.add(Lambda,u0),u1)
    Lambda = 2*(2-Lambda)
    Lambda = Lambda**alpha

    # Starting interpolation: 
    if x0 is None:
        y = initial_nn(x)
    else:
        y = np.copy(x0)


    for mu in range(m):
        Gamma = 1/(1+s[mu]*Lambda)
        a = np.copy(y)
        a[ids0] = (x-y)[ids0]+y[ids0]
        y = rf*idct(Gamma*dct(a, norm='ortho'), norm='ortho')+(1-rf)*y
    
    y[ids0] = x[ids0]
    return y




def initial_nn(x, method="nearest"):
    """ Function that provides a grid-interpolation for a 2d images with missing elements (either np.nan, or np.inf).

        (this is basically a direct call to griddata)

        Args
            INPUTS:
                x (numpy array) - a 2-dimensional image
                method (string) - is one of "nearest", "linear", "cubic".
            OUT:
                y (numpy array) - same image as x but where missing elements interpolated.
    
    """
    reso = x.shape
    grid_m, grid_n = np.mgrid[0:reso[0]-1:reso[0] + 0j, 0:reso[1]-1:reso[1] + 0j]
    ids0 = np.isfinite(x)
    if ids0.all(): #Nothing to interpolate...
        return x
    points = np.where( ids0)
    y = griddata(points, x[points], (grid_m, grid_n), method = method, fill_value = 0.0, rescale = False)
    return y







    
    


