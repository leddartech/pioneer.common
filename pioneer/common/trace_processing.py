from pioneer.common.linalg import fit_cubic
from pioneer.common.types.calibration import SaturationCalibration

from enum import Enum
from scipy.signal import convolve2d
from typing import List

import numpy as np

class TraceProcessing():

    def __init__(self, priority:int=0):
        self.priority = priority

    def __call__(self, traces):
        return traces



class TraceProcessingCollection():

    def __init__(self, list_trace_processing:List[TraceProcessing]=[]):

        priorities = [trace_processing.priority for trace_processing in list_trace_processing]
        self.apply_order = np.argsort(priorities)
        self.list_trace_processing = list_trace_processing

    def __call__(self, traces):
        for i in self.apply_order:
            traces = self.list_trace_processing[i](traces)
        return traces


class RemoveStaticNoise(TraceProcessing):

    def __init__(self, static_noise):
        super(RemoveStaticNoise, self).__init__(priority=TraceProcessingPriorities.RemoveStaticNoise.value)
        self.static_noise = static_noise

    def __call__(self, traces):
        if self.static_noise is not None:
            traces['data'] -= self.static_noise
        return traces


class Realign(TraceProcessing):

    def __init__(self, target_time_base_delay=None):
        super(Realign, self).__init__(priority=TraceProcessingPriorities.Realign.value)
        self.target_time_base_delay = target_time_base_delay

    def __call__(self, traces):
        if type(traces['time_base_delays']) in [int, float, type(None)] and self.target_time_base_delay is None:
            return traces

        offsets_nb_pts = -traces['time_base_delays'] / traces['distance_scaling']
        offsets_nb_pts -= np.min(offsets_nb_pts)

        if self.target_time_base_delay is not None:
            offset = (np.max(traces['time_base_delays']) - self.target_time_base_delay) / traces['distance_scaling']
            if offset < 0:
                offsets_nb_pts -= offset

        while np.max(offsets_nb_pts) > 0:
            ind = np.where(offsets_nb_pts//1==0.0)[0] #where offset between 0 and 1
            traces['data'][ind,:-1] = np.multiply(traces['data'][ind,:-1].T, 1-offsets_nb_pts[ind]).T + np.multiply(traces['data'][ind,1:].T, offsets_nb_pts[ind]).T
            traces['time_base_delays'][ind] += offsets_nb_pts[ind]*traces['distance_scaling']
            ind = np.where(offsets_nb_pts > 1.0)[0] #where offset > 1
            traces['data'][ind,:-1] = traces['data'][ind,1:]
            traces['time_base_delays'][ind] += traces['distance_scaling']
            offsets_nb_pts -= 1

        return traces



class ZeroBaseline(TraceProcessing):

    def __init__(self):
        super(ZeroBaseline, self).__init__(priority=TraceProcessingPriorities.ZeroBaseline.value)

    def __call__(self, traces):
        traces['data'] = traces['data'].astype('float64')
        traces['data'] -= np.mean(traces['data'][:,:10], axis=-1)[...,None]
        return traces



class Clip(TraceProcessing):

    def __init__(self, min_value=0, max_value=np.inf):
        super(Clip, self).__init__(priority=TraceProcessingPriorities.Clip.value)
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, traces):
        traces['data'] = np.clip(traces['data'], self.min_value, self.max_value)
        return traces



class CutInterval(TraceProcessing):

    def __init__(self, min_indice:int=0, max_indice:int=-1):
        super(CutInterval, self).__init__(priority=TraceProcessingPriorities.CutInterval.value)
        self.min_indice = min_indice
        self.max_indice = max_indice

    def __call__(self, traces):
        traces['data'] = traces['data'][...,self.min_indice:self.max_indice]
        traces['time_base_delays'] += self.min_indice*traces['distance_scaling']
        return traces



class Smooth(TraceProcessing):

    def __init__(self):
        super(Smooth, self).__init__(priority=TraceProcessingPriorities.Smooth.value)

    def __call__(self, traces):
        smoothing_kernel = np.expand_dims(traces['trace_smoothing_kernel'], 0)
        if traces['data'].ndim == 1:
            traces['data'] = np.expand_dims(traces['data'],0)
            traces['data'] = convolve2d(traces['data'], smoothing_kernel, mode = "same", boundary='symm')[0]
        else:
            traces['data'] = convolve2d(traces['data'], smoothing_kernel, mode = "same", boundary='symm')
        
        return traces



class Desaturate(TraceProcessing):

    def __init__(self, saturation_calibration:SaturationCalibration=None):
        super(Desaturate, self).__init__(priority=TraceProcessingPriorities.Desaturate.value)
        self.saturation_calibration = saturation_calibration

    def __call__(self, traces):
        traces['data'] = traces['data'].astype('float64')
        saturation_value = traces['data'].max()

        if saturation_value == 0:
            return traces

        where_plateau = np.where(traces['data'] == saturation_value)

        channels, ind, sizes = np.unique(where_plateau[0], return_index=True, return_counts=True)
        positions = where_plateau[1][ind]

        for channel, position, size in zip(channels, positions, sizes):
            if size > 5 and position > 2 and position + size + 2 < traces['data'].shape[-1]:

                # x axis for the fit
                x = np.arange(0,traces['data'][channel].shape[0])*traces['distance_scaling']
                x = x[position:position + size]

                # Before plateau
                x1 = (position - 1.5) * traces['distance_scaling']
                y1 = (traces['data'][channel][position - 1] + traces['data'][channel][position - 2])/2 - saturation_value
                dy1 = (traces['data'][channel][position - 1] - traces['data'][channel][position - 2])/traces['distance_scaling']

                # After plateau
                x2 = x1 + (size + 2.5)*traces['distance_scaling']
                y2 = (traces['data'][channel][position + size + 2] + traces['data'][channel][position + size + 1])/2 - saturation_value
                dy2 = (traces['data'][channel][position + size + 2] - traces['data'][channel][position + size + 1])/traces['distance_scaling']


                if self.saturation_calibration is not None:

                    start_plateau_position = traces['distance_scaling']*(position - 
                                                (traces['data'][channel][ position ] - traces['data'][channel][position-1]) \
                                            /(traces['data'][channel][position-1] - traces['data'][channel][position-2]))
                    end_plateau_position = traces['distance_scaling']*(position + size - 1 +
                                                (traces['data'][channel][position+size-1] - traces['data'][channel][position+size]) \
                                            /(traces['data'][channel][position+size] - traces['data'][channel][position+size+1]))

                    x0, y0 = self.saturation_calibration(end_plateau_position - start_plateau_position)
                    x0 += start_plateau_position

                    # Cubic fit between start of plateau and peak
                    a1, b1, c1, d1 = fit_cubic(p1=(x0,y0), p2=(x1,y1), d1=0, d2=dy1)
                    # Cubic fit between peak and end of plateau 
                    a2, b2, c2, d2 = fit_cubic(p1=(x0,y0), p2=(x2,y2), d1=0, d2=dy2)
                    
                    ind_x0 = np.argmax((x-x0)>0)
                    traces['data'][channel][position:position + size][:ind_x0] += a1*x[:ind_x0]**3 + b1*x[:ind_x0]**2 + c1*x[:ind_x0] + d1
                    traces['data'][channel][position:position + size][ind_x0:] += a2*x[ind_x0:]**3 + b2*x[ind_x0:]**2 + c2*x[ind_x0:] + d2

                else:
                    # Cubic fit between start of plateau and peak
                    a, b, c, d = fit_cubic(p1=(x1,y1), p2=(x2,y2), d1=dy1, d2=dy2)
                    traces['data'][channel][position: position+size] += a*x**3 + b*x**2 + c*x + d

        return traces


class Decimate(TraceProcessing):
    def __init__(self, factor:int=1):
        super(Decimate, self).__init__(priority=TraceProcessingPriorities.Decimate.value)
        self.factor = factor

    def __call__(self, traces):
        traces['data'] = traces['data'][:,::self.factor]
        traces['distance_scaling'] *= self.factor
        return traces


class Binning(TraceProcessing):
    def __init__(self, factor:int=1):
        super(Binning, self).__init__(priority=TraceProcessingPriorities.Binning.value)
        self.factor = factor

    def __call__(self, traces):
        min_len = int(np.floor(traces['data'].shape[-1]/self.factor))
        traces['data'] = sum([traces['data'][:,i::self.factor][:,:min_len] for i in range(self.factor)])/self.factor
        traces['time_base_delays'] += 0.5*(self.factor-1)*traces['distance_scaling']
        traces['distance_scaling'] *= self.factor
        return traces


class TraceProcessingPriorities(Enum):
    Desaturate = 0
    RemoveStaticNoise = 1
    CutInterval = 2
    Realign = 3
    ZeroBaseline = 4
    Clip = 5
    Smooth = 6
    Binning = 7
    Decimate = 8