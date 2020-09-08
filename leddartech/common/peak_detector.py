try:
    from scipy.signal import find_peaks
    FIND_PEAKS_AVAILABLE = True
except:
    FIND_PEAKS_AVAILABLE = False

import copy
import numpy as np
import time
import warnings

for message in ['some peaks have','invalid value encountered in','All-NaN slice encountered','divide by zero']:
    warnings.filterwarnings('ignore', message=message)

class PeakDetector(object):

    def __init__(self, nb_detections_max:int=3, min_amplitude:float=0, min_distance:float=0, verbose:bool=False):

        self.nb_detections_max = nb_detections_max
        self.min_amplitude = min_amplitude
        self.min_distance = min_distance
        self.verbose = verbose

        if self.nb_detections_max > 1 and not FIND_PEAKS_AVAILABLE:
            print('Warning: scipy.signal.find_peaks is not available. Thus, nb_detections_max is set back to 1.')
            self.nb_detections_max = 1

    @staticmethod
    def _fit(max_plus_neighbors):
        """Separate method to fit the parabola, so we can call it in parallel."""
        try:
            coeffs =  np.polyfit(np.array([-1,0,1]), max_plus_neighbors, deg=2)
        except:
            coeffs =  np.array([1,0,1]).astype('float64')
        return coeffs

    # @staticmethod
    def _find_peaks(self, trace):
        peaks = find_peaks(trace, height=1)
        return peaks[0][np.argsort(-peaks[1]['peak_heights'])][:self.nb_detections_max]


    def __call__(self, traces):

        start = time.time()
        echoes = {}

        if type(traces)==np.ndarray:
            traces = {'data':traces}
        if 'time_base_delays' not in traces:
            traces['time_base_delays'] = 0
        elif traces['time_base_delays'] is None:
            traces['time_base_delays'] = 0

        k = 1 #The k^th neighbor both sides is selected for the fit
        ind_distance_zero = int(max([np.floor(np.min(-traces['time_base_delays'])/traces['distance_scaling'])-k,0]))

        # The number of samples each sides of a peak considered in the width and skew calculation
        pulse_window = 20

        if traces['data'].ndim == 1:
            traces['data'] = np.expand_dims(traces['data'], axis=0)

        padded_traces = np.pad(traces['data'],((0,0),(pulse_window,pulse_window+1))) #For pulse width and skew

        if self.nb_detections_max == 1:
            ind_max = np.argmax(traces['data'][:,2*k+ind_distance_zero:-2*k],axis=-1)+2*k+ind_distance_zero #Simplest peak detector: argmax
            echoes['indices'] = np.array(range(len(ind_max))).astype('u4')
            ind = np.indices(ind_max.shape)
            amp_5pts = np.vstack([traces['data'][ind,ind_max+ind_k] for ind_k in [-2*k,-k,0,k,2*k]]).T
            amp_pulse = np.vstack([padded_traces[ind,ind_max+ind_pulse+pulse_window] for ind_pulse in np.arange(-pulse_window, pulse_window+1, dtype=int)]).T

        else:
            peaks = [self._find_peaks(traces['data'][ch,2*k+ind_distance_zero:-2*k]) for ch in range(traces['data'].shape[0])]
            indices, ind_max = [], []
            for ch in range(len(peaks)):
                for i in range(len(peaks[ch])):
                    indices.append(ch)
                    ind_max.append(peaks[ch][i])
            ind_max = np.array(ind_max)+2*k+ind_distance_zero
            echoes['indices'] = np.array(indices).astype('u4')

            amp_5pts = np.empty(5)
            for i, ch in enumerate(echoes['indices']):
                amp_5pts = np.vstack([amp_5pts,traces['data'][ch,ind_max[i]-2*k:ind_max[i]+2*k+1][::k]])
            amp_5pts = amp_5pts[1:]

            amp_pulse = np.empty(2*pulse_window+1)
            for i, ch in enumerate(echoes['indices']):
                amp_pulse = np.vstack([amp_pulse,[padded_traces[ch,ind_max[i]+ind_pulse+pulse_window] for ind_pulse in np.arange(-pulse_window, pulse_window+1, dtype=int)]])
            amp_pulse = amp_pulse[1:]

        if amp_5pts.ndim == 1:
            return {'amplitudes':0, 'distances':0, 'indices':0, 'widths':0, 'skews':0}

        # Amplitudes - highest value
        echoes['amplitudes'] = amp_5pts[:,2]

        # Distances - interpolated from 3 highest values
        a = k*(2*amp_5pts[:,2] - amp_5pts[:,1] - amp_5pts[:,3])
        b = k**2 * (amp_5pts[:,1] - amp_5pts[:,3])
        distance_corrections = -b/(2*a+1e-9)
        distance_corrections = np.clip(distance_corrections,-0.5*k,0.5*k)

        # Time base delays
        echoes['distances'] = (ind_max+distance_corrections)*traces['distance_scaling']
        if type(traces['time_base_delays']) in [int,float] or traces['time_base_delays'].size == 1:
            tbd = np.full(len(echoes['distances']), traces['time_base_delays'])
        else:
            tbd = traces['time_base_delays'][echoes['indices']]
        echoes['distances'] += tbd  

        # Widths and skews - from intersections (left and right) with half amplitude
        all_ind = np.arange(amp_pulse.shape[0])
        diff_left = amp_pulse[:,:pulse_window] - echoes['amplitudes'][:,None]/2
        smallest_non_neg_ind_left = np.argmin(np.where(diff_left > 0, diff_left, np.inf), axis=1)
        corr_left = diff_left[all_ind,smallest_non_neg_ind_left]/(diff_left[all_ind,smallest_non_neg_ind_left] - diff_left[all_ind,smallest_non_neg_ind_left-1])
        width_left = pulse_window - smallest_non_neg_ind_left - corr_left
        diff_right = amp_pulse[:,pulse_window:] - echoes['amplitudes'][:,None]/2
        smallest_non_neg_ind_right = np.argmin(np.where(diff_right > 0, diff_right, np.inf), axis=1)
        corr_right = -diff_right[all_ind,smallest_non_neg_ind_right]/(diff_right[all_ind,smallest_non_neg_ind_right] - diff_right[all_ind,smallest_non_neg_ind_right-1])
        width_right = smallest_non_neg_ind_right + corr_right
        echoes['widths'] = (width_left + width_right)*traces['distance_scaling']
        echoes['widths'][~np.isfinite(echoes['widths'])] = 0
        echoes['widths'] = np.clip(echoes['widths'], 0, 2*pulse_window*traces['distance_scaling'])
        echoes['skews'] = (width_left - width_right)*traces['distance_scaling']
        echoes['skews'][~np.isfinite(echoes['skews'])] = 0
        echoes['skews'] = np.clip(echoes['skews'], -pulse_window*traces['distance_scaling'], pulse_window*traces['distance_scaling'])


        filter = list(np.where(echoes['distances'] <= self.min_distance)[0])
        filter += list(np.where(echoes['amplitudes'] <= self.min_amplitude)[0])
        for key in echoes.keys():
            echoes[key] = np.delete(echoes[key],filter)

        if self.verbose:
            print('Peak detection time : {:d}ms.'.format(int((time.time() - start)*1e3)))

        return echoes

