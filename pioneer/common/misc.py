from scipy.ndimage import gaussian_filter
from scipy.constants import speed_of_light

import numpy as np

class Signal(object):
    """ Simple class to implement a signal/slot system """
    def __init__(self):
        self._slots = set()
        self._reintering = False
    
    def connect(self, slot):
        if slot is not self: #prevent 1st level of recursion
            self._slots.add(slot)

    def __call__(self, *args, **kwargs):

        if self._reintering:
            raise RuntimeError(f"Reucursion detected in {self}, aborting!")

        self._reintering = True
        try:
            for s in self._slots:
                s(*args, **kwargs)
        finally:
            self._reintering = False

def gaussian_kernel(shape, sigma, dtype = np.float32):
    dirac = np.zeros(shape, dtype = dtype)

    e = dirac
    for s in shape[:-1]:
        e = e[s//2, ...]
    e[shape[-1]//2] = 1
    kernel = gaussian_filter(dirac, sigma = sigma, mode = 'constant', cval = 0)
    kernel /= np.sum(kernel)
    return kernel

def distance_per_sample(adc_freq, oversampling = 1, binning = 1):
    return speed_of_light / (adc_freq * oversampling / binning) / 2

def weighted_avg(kernel, kernel_mask, scalars, scalars_mask):
    masked_kernel = kernel.flat[kernel_mask]
    return np.dot(scalars[scalars_mask], masked_kernel/masked_kernel.sum())

def map_angles_minus_pi_to_pi(angles):
    a = np.array(angles)
    mask = (a < -np.pi) | (a > np.pi)
    a[mask] = a[mask] % (2 * np.pi) - np.pi
    return a

def colors_to_reflectances(colors):
    ''' converts colors (Nx4) to reflectances (N), respecting opacity
    '''
    return np.mean(colors[:,:3]*colors[:, 3,None], axis = 1)