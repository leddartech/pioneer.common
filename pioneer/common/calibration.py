from pioneer.common.logging_manager import LoggingManager
from pioneer.common import clouds, plane, banks, images
from pioneer.common import interpolator as Ir

try:
    import leddar
except:
    LoggingManager.instance().warning("Could not import 'leddar', no live sensor can be used!")

from datetime import datetime
from future.utils import viewitems
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import numpy as np
import pickle
import re
import time

MAX_RANGE_320_MHZ = 299792458/3.2e8*512/2 #speed_of_light / ADC_freq  * n_samples / 2 (light roundtrip)

__packages = []
def echoes_callback(leddar_echoes):
    __packages.append(leddar_echoes)

def connect_and_configure(ip = "192.168.0.20", data_thread_delay = 1000, callback = echoes_callback):
    dev = leddar.Device()
    if not dev.connect(ip, leddar.device_types["LeddarAuto"], 48630):
        print("connection failed!")
        raise RuntimeError("Connection failed!")
    dev.set_callback_echo(callback)
    dev.set_data_mask(leddar.data_masks["PDM_ECHOES"])
    dev.set_data_thread_delay(data_thread_delay)
    return dev

def acquire_and_save(dev, acquisition_time = 1, suffix = ""):
    specs = banks.get_specs(dev)

    dev.start_data_thread()
    time.sleep(acquisition_time)
    dev.stop_data_thread()

    print("acquired {0} frames in {1} seconds ({2} fps)".format(len(__packages), acquisition_time, len(__packages)/acquisition_time))
    data = {"specs": specs, "packages": __packages}

    with open("./data/dataset" + suffix + ".pkl", 'wb') as f:
        pickle.dump(data, f)

    return data

def wall_distances(wall_distance, angles):
    cos_angles = np.cos(angles)
    return wall_distance / cos_angles[:, 0] / cos_angles[:, 1]

def wall_distances_2(wall_distance, angles):
    directions = clouds.directions(angles)
    return plane.intersect_rays([0,0,0], directions, np.array([0,0,1,-wall_distance]))

def avg_distances(packages, v, h, interpolate=False):
    avg_distances = np.zeros((v*h,))
    counts = np.zeros(avg_distances.shape, np.uint32)
    
    if interpolate:
        print('Warning: empty echoes are interpolated')
        n=0
        for p in packages:
            completed_d = np.full((v*h,), np.nan)
            d = p['data']
            mask = images.maximum_amplitude_mask(d)
            d = d[mask]
            selection = d["indices"]
            completed_d[selection] = d['distances']
            completed_d = Ir.initial_nn(completed_d.reshape((v,h))).reshape((v*h,))
            avg_distances += completed_d
            n+=1
        return avg_distances/n
    
    else:
        for p in packages:
            d = p['data']
            mask = images.maximum_amplitude_mask(d)
            d = d[mask]
            selection = d["indices"]
            avg_distances[selection] += d['distances']
            counts[selection] += 1
            
        ids_valid = np.where(counts>0)
        avg_distances[ids_valid] = avg_distances[ids_valid]/counts[ids_valid]

    return avg_distances

    

def connect_and_set_apd_bias(value = 170):
    '''
        use this after you performed a './LCA3 -b all' on your unit. 160 to 190 range seems right for the 3DF
    '''
    dev = leddar.Device()
    if not dev.connect("192.168.0.20", leddar.device_types["LeddarAuto"], 48630):
        raise RuntimeError("Connection failed!")

    dev.set_property_value(leddar.property_ids['ID_APD_VBIAS_VOLTAGE_T0'], str(value))

    print("ID_SENSIVITY:" + dev.get_property_value(leddar.property_ids['ID_SENSIVITY']))
    print("ID_APD_VBIAS_VOLTAGE_T0:" + dev.get_property_value(leddar.property_ids['ID_APD_VBIAS_VOLTAGE_T0']))
    print("ID_APD_VBIAS_MULTI_FACTOR:" + dev.get_property_value(leddar.property_ids['ID_APD_VBIAS_MULTI_FACTOR']))
    print("ID_APD_VBIAS_T0:" + dev.get_property_value(leddar.property_ids['ID_APD_VBIAS_T0']))
    dev.disconnect()

def calibrate(packages, wall_distance, specs, interpolate=False):
    v, h, v_fov, h_fov = (specs[x] for x in ["v", "h", "v_fov", "h_fov"])
    print('Calib specs (v,h,v_fov,h_fov):',v,h,v_fov,h_fov)

    if 'angles' in specs:
        angles = specs['angles']
    else:
        angles = clouds.angles(v, h, v_fov, h_fov)
    
    avg = avg_distances(packages, v, h, interpolate)

    return wall_distances_2(wall_distance, angles) - avg


def set_offset_to_base_temperature(state_packages, offsets, gain_factor, T_base):
    T_0 = 0
    n = 0
    for omega in state_packages:
        T_0 += omega['system_temp']
        n += 1
    return offsets + gain_factor*T_0/n-gain_factor*T_base


def distance_correction_temperature(echo, temperature, gain_factor, T_fix):
    echo['data']['distances'] = echo['data']['distances']- gain_factor*(-T_fix+temperature)
    return echo


def amplitude_correction_temperature(echo, temperature, gain_factor, threshold):
    if temperature > threshold:
        echo['data']['amplitudes'] = echo['data']['amplitudes']*(1 + gain_factor*(temperature-threshold))
    return echo


class SaturationCalibration():
    # TODO: document when the final form has converged.

    def __init__(self, distance_coeff:float, amplitude_coeffs:tuple):
        self.distance_coeff = distance_coeff
        self.amplitude_coeffs = amplitude_coeffs

    def __call__(self, plateau_size):
        distance = self.distance_coeff*plateau_size
        amplitude = self.amplitude_coeffs[0]*np.exp(self.amplitude_coeffs[1]*plateau_size)
        return distance, amplitude


'''
FIXME: The following does no belong in configuration.py
'''

def flatten_yml_configurations_section(yml_value):
    cfg_flat = {}
    for key, value in yml_value['configurations'].items():
        if isinstance(value, dict):
            cfg_flat = {**cfg_flat, **value}
        else:
            cfg_flat[key] = value
    return cfg_flat

def read_configuration(dev):
        configuration = {'timestamp' : int(datetime.utcnow().timestamp() * 1e6)}
        for (name,prop) in leddar.property_ids.items():
            try:
                configuration[name] = dev.get_property_value(prop)
            except:
                pass
        calibration = {}
        for (name,prop) in leddar.calib_types.items():
            try:
                calibration[name] = dev.get_calib_values(prop)
            except:
                pass
        configuration['calibration'] = calibration

        return configuration

def configure_properties(dev, cfg_flat):
    for name,value in cfg_flat.items():
        if name in leddar.property_ids:
            dev.set_property_value(name, str(value))

def set_data_mask(dev, data_masks):

            #data masks:
        data_masks = re.split('[;, ]', data_masks.strip())

        data_mask = 0
        for dm in data_masks:
            data_mask |= leddar.data_masks[dm]
        
        dev.set_data_mask(data_mask)

def prepend_ts_msb(ts, prev_ts_64 = None, ts_local = None):
    if ts_local is None:
        t = datetime.utcnow().timestamp()
        ts_local = int(t * 1e6) # to microseconds

    msb = ts_local & 0xFFFFFFFF00000000

    if prev_ts_64 is not None:
        prev_ts = prev_ts_64 & 0x00000000FFFFFFFF
        prev_msb = prev_ts_64 & 0xFFFFFFFF00000000
        if msb > prev_msb and prev_ts < ts:
            msb = prev_msb #the msb changed before the 32bit overflow, use the previous one

    return msb | int(ts)




    
