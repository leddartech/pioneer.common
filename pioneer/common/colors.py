from pioneer.common.logging_manager import LoggingManager

from matplotlib import pyplot as plt

import matplotlib
import numpy as np

def amplitudes_to_color(amplitudes, power=1.0, mask=None, log_normalize=False):
    
    cm = plt.get_cmap('viridis')
    if mask is None:
        mask = np.ones(amplitudes.shape, dtype=np.bool)

    amplitudes = np.power(amplitudes, power)
    masked_amps = amplitudes[mask]
    if masked_amps.size == 0:
        LoggingManager.instance().warning('Provided mask is empty')
        dmin, dmax = amplitudes.min(), amplitudes.max()
    else:
        dmin, dmax = masked_amps.min(), masked_amps.max()

    if log_normalize:
        norm = matplotlib.colors.LogNorm(dmin, dmax)
    else:
        norm = matplotlib.colors.Normalize(dmin, dmax)

    colors = norm(amplitudes)
    colors = cm(colors)
    colors = (colors[:, :3]*255).astype('u1')

    return colors