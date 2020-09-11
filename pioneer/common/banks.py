from pioneer.common import clouds, plane
from pioneer.common.logging_manager import LoggingManager

import ctypes
import numpy as np
import os
import pickle
import platform
import traceback

def get_bank_offsets(v, h):
    '''
    For the LCA3 MEMs, The vertical fov is divided in 3 (256 scan lines divided in 3 sections due to fpga memory constraints = 86 + 85 + 85),
    and the horizontal fov is divided in 4 (64 APD = 16ADC * 4 mux)
    each of theese banks share the same timestamp due to the way trace sequence are gathered (and accumulated)
    '''
    h_offsets = (0, 16, 32, 48, 64)
    if v == 8: #LCA2
        return (tuple(range(v+1)), h_offsets[0:3])
    if v in (16, 64, 128, 312): #LCA3_3DF, LCA3_MEMS_Eagle 
        return (tuple(range(v+1)), h_offsets)
    elif v == 172: #LCA3_MEMS_Appolo
        return ((0, 86, 86+86), h_offsets)
    elif v == 256: #LCA3_MEMS
        return ((0, 86, 86+85, 86+85+85), h_offsets)
    return None

def add_timestamp_offsets(echo_package, sensor_name, specs, n_accumulations, n_oversampling, n_basepointcount):

    try:
        scan_direction = 0 if 'scan_direction' not in echo_package else echo_package['scan_direction'] #backward compatible for packages without 'scan_direction'
        sampled_to_ordered = specs['sampled_to_ordered_inv'] if scan_direction == 1 else specs['sampled_to_ordered']
        sensor_type = sensor_name.split('_')[0]
        rv, to_us_coeff = get_ts_offsets(sensor_type, sampled_to_ordered, specs['v'], specs['h'], n_accumulations, n_oversampling, n_basepointcount, scan_direction)
        echo_package['data']['timestamps'] = rv[echo_package['data']['indices']]
        echo_package['data'] = np.sort(echo_package['data'], order = ('timestamps') )
        echo_package['timestamps_to_us_coeff'] = to_us_coeff
        echo_package['timestamps'] = echo_package['data']['timestamps'].astype('u8') * to_us_coeff + echo_package['timestamp']

        echo_package['eof_timestamp'] = int(int(rv.max()) * to_us_coeff + echo_package['timestamp']) #FIXME: in fact actual end would be a little after max(), but this is close enough
    except:
        traceback.print_exc()
    
    return echo_package


def add_timestamp_offsets_to_trace(trace_package, echo_package, sensor_name, specs, n_accumulations, n_oversampling, n_basepointcount):

    try:
        scan_direction = 0 if 'scan_direction' not in echo_package else echo_package['scan_direction'] #backward compatible for packages without 'scan_direction'
        sampled_to_ordered = specs['sampled_to_ordered_inv'] if scan_direction == 1 else specs['sampled_to_ordered']
        sensor_type = sensor_name.split('_')[0]
        rv, to_us_coeff = get_ts_offsets(sensor_type, sampled_to_ordered, specs['v'], specs['h'], n_accumulations, n_oversampling, n_basepointcount, scan_direction)
   
        trace_package['timestamps'] = rv.astype('u8') * to_us_coeff + trace_package['timestamp']
    except:
        traceback.print_exc()
    
    return echo_package


# FIXME: theese timings should be obtained from the sensor.
TS_OFFSETS_128 = {"T_LASER_PULSE_ns": 2500
                ,"T_MIRROR_ns"     : 400000
                ,"T_LCPG_ns"       : 1500000
                ,"N_LCPG"          : 16}

TS_OFFSETS_312 = {"T_LASER_PULSE_ns": 2500
                ,"T_MIRROR_ns"     : 1e9 * 0x02FA00*.3 /160e6 #1219200 # modif pierre pour les timing eagle 80 ok pour support 4Hz le 11/07/2019
                ,"T_LCPG_ns"       : 1e9 * 0x08A980*.5 /160e6 #3548000
                ,"N_LCPG"          : 13}

TS_OFFSETS_B0 = {"T_LASER_ns": 700
                ,"T_CPU_ns":182400
                ,"T_QSPI_CLOCK_ns":25}

_ts_offsets_cache = {}
def get_ts_offsets(sensor_type, sampled_to_ordered, v, h, n_accumulations, n_oversampling = 1, n_basepointcount = 64, scan_direction = 0, timings = None, dtype = np.dtype('u2')):
    '''
        Returns an array of offsets that should be added to the timestamp of an echo package to obtain the actual 
        timestamp of a single echo. 
        Returns: a tuple with two elements:
                    tuple[0] = the array of offsets
                    tuple[1] = the coefficient to convert to microseconds 
    '''
    global _ts_offsets_cache

    
    if (v, h, n_accumulations, n_oversampling, scan_direction) not in _ts_offsets_cache:

        v_offsets, h_offsets = get_bank_offsets(v, h)

        ts_offsets_ns = np.zeros((v,h), 'u8')
        
        if sensor_type == 'lca2': #v=8, h=32
            
            if timings is None:
                timings = TS_OFFSETS_B0

            ts_offsets_ns = np.zeros((v,h))
            
            points_per_trace = n_basepointcount*n_oversampling
            t_data_transfer_ns = timings["T_QSPI_CLOCK_ns"]*(points_per_trace*64.+17.)
            t_process_ns = 3500.*points_per_trace + 72900.

            # There are 16 (2*v) blocks of acquisition per frame.
            for n in range(2*v):
                # The indices of the first block --> [0,1::2]
                #                    second block -> [0,0::2]
                #                    third block --> [1,1::2]
                #                    fourth block -> [1,0::2]
                #                    fifth block --> [2,1::2]
                #                    nth block ----> [n//2,(n+1)%2::2]

                # The offset of the middle of the nth block is given by
                # offset = (2n+1)/2 * acquitision_time + n*delay
                # where the delay is the time between each block.
                ts_offsets_ns[int(n//2),int((n+1)%2)::2] = \
                    (2.*n+1.)/2.*timings["T_LASER_ns"]*n_accumulations*n_oversampling \
                    + n*(t_data_transfer_ns + timings["T_CPU_ns"] + t_process_ns)

            flat_ts_offsets_ns = ts_offsets_ns.flatten()
            max_ts = flat_ts_offsets_ns.max()
            to_ns_coeff = int(np.ceil(max_ts/np.iinfo(dtype).max))

            _ts_offsets_cache[(v, h, n_accumulations, n_oversampling, scan_direction)] = \
                ((flat_ts_offsets_ns[sampled_to_ordered] // to_ns_coeff).astype(dtype)\
                    ,to_ns_coeff/1000)

        elif sensor_type == 'pixell': #v=8, h=96
            pass


        elif sensor_type == 'eagle': #LCA3_MEMS_Eagle_128 or LCA3_MEMS_Eagle_312

            if timings is None:
                timings = TS_OFFSETS_128 if v == 128 else TS_OFFSETS_312

            n_lasers_pulses = 0
            
            cumul_delays_ns = -timings["T_LCPG_ns"] #since 0%16 == 0

            for v_b in range(len(v_offsets)-1):
                v_from, v_to = v_offsets[v_b], v_offsets[v_b+1]
                v_size = v_to - v_from
                
                if v_b%timings["N_LCPG"] == 0: # at each 16 angles, the LCPG moves
                    cumul_delays_ns += timings["T_LCPG_ns"]
                else:
                    cumul_delays_ns += timings["T_MIRROR_ns"] # mirror movement only
                
                for h_b in range(len(h_offsets)-1):
                    h_from, h_to = h_offsets[h_b], h_offsets[h_b+1]
                    h_size = h_to - h_from

                    ts_offsets_ns[v_from:v_to, h_from:h_to] = (cumul_delays_ns + int(n_lasers_pulses * timings["T_LASER_PULSE_ns"]))
                    n_lasers_pulses += n_accumulations
            
            flat_ts_offsets_ns = (ts_offsets_ns + int(n_accumulations * timings["T_LASER_PULSE_ns"]/2)).flatten() # center in the middle of the interval

            max_ts = flat_ts_offsets_ns.max()

            to_ns_coeff = int(np.ceil(max_ts/np.iinfo(dtype).max))
            
             # make the data fit in dtype, and divide by 1000 since our target resolution is microseconds
            _ts_offsets_cache[(v, h, n_accumulations, n_oversampling, scan_direction)] = \
                ((flat_ts_offsets_ns[sampled_to_ordered] // to_ns_coeff).astype(dtype)\
                , to_ns_coeff/1000)
                
        else:
            raise RuntimeError(f"Could not get timestamps offset per channel for the sensor: {sensor_type}. Unsupported.")

    return _ts_offsets_cache[(v, h, n_accumulations, n_oversampling, scan_direction)]

def get_bank_indices(ordering, v_bank, h_bank, v_offsets, h_offsets):
    v, h = (v_offsets[-1], h_offsets[-1])
    return ordering.reshape(v, h, order = 'C')[v_offsets[v_bank]:v_offsets[v_bank+1], h_offsets[h_bank]:h_offsets[h_bank+1]].flatten()


def get_bank_range(v_bank, h_bank, v_offsets, h_offsets):
    row_start = v_offsets[v_bank] * h_offsets[-1]
    v_size = (v_offsets[v_bank+1] - v_offsets[v_bank])
    return (row_start + v_size * h_offsets[h_bank], row_start + v_size * h_offsets[h_bank+1])

_ordering_cache = {}
def get_sampling_ordering_flat(v, h):

    global _ordering_cache
    if (v,h) not in _ordering_cache:
        if v in [8, 16, 64, 128, 172, 256, 312]: 
            if platform.system() == "Windows":
                ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), "leddar_utils_lcax.dll"))
                lca_math = ctypes.cdll.leddar_utils_lcax
            else:
                ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), "libleddar_utils_lcax.so"))
                lca_math = ctypes.CDLL("libleddar_utils_lcax.so")

            lca_math.MATH_GenIndexMap(v, h)

            fMATH_CopyMapTrc2Mem = lca_math.MATH_CopyMapTrc2Mem
            fMATH_CopyMapTrc2Mem.restype = None
            fMATH_CopyMapTrc2Mem.argtypes = [ctypes.c_size_t, np.ctypeslib.ndpointer(ctypes.c_uint16, flags="C_CONTIGUOUS"), ctypes.c_bool]


            sampled_to_ordered = np.zeros((v*h), dtype = np.uint16, order = 'C')
            fMATH_CopyMapTrc2Mem(v*h, sampled_to_ordered, False)

            sampled_to_ordered_inv = np.zeros((v*h), dtype = np.uint16, order = 'C')
            fMATH_CopyMapTrc2Mem(v*h, sampled_to_ordered_inv, True)

            fMATH_CopyMapMem2Trc = lca_math.MATH_CopyMapMem2Trc
            fMATH_CopyMapMem2Trc.restype = None
            fMATH_CopyMapMem2Trc.argtypes = [ctypes.c_size_t, np.ctypeslib.ndpointer(ctypes.c_uint16, flags="C_CONTIGUOUS"), ctypes.c_bool]

            ordered_to_sampled = np.zeros((v*h), dtype = np.uint16, order = 'C')
            fMATH_CopyMapMem2Trc(v*h, ordered_to_sampled, False)
            ordered_to_sampled_inv = np.zeros((v*h), dtype = np.uint16, order = 'C')
            fMATH_CopyMapMem2Trc(v*h, ordered_to_sampled_inv, False)
            _ordering_cache[(v,h)] = (ordered_to_sampled, sampled_to_ordered, ordered_to_sampled_inv, sampled_to_ordered_inv)
        else:
            LoggingManager.instance().warning("Unsupported v resolution")
            return None, None
    
    return _ordering_cache[(v,h)]
    

def fill_specs(v,h,v_fov, h_fov):
    try:
        ordered_to_sampled, sampled_to_ordered, ordered_to_sampled_inv, sampled_to_ordered_inv = get_sampling_ordering_flat(v,h)
    except:
        return {"v": v, "h" : h, "v_fov":  v_fov, "h_fov" : h_fov}

    return {"v": v, "h" : h, "v_fov":  v_fov, "h_fov" : h_fov
    , "ordered_to_sampled" : ordered_to_sampled, "sampled_to_ordered" : sampled_to_ordered
    , "ordered_to_sampled_inv": ordered_to_sampled_inv, "sampled_to_ordered_inv": sampled_to_ordered_inv
    , "bank_offsets" : get_bank_offsets(v, h)}

def extract_specs(getter):

    return fill_specs(int(getter("ID_VERTICAL_CHANNEL_NBR")), int(getter("ID_HORIZONTAL_CHANNEL_NBR")), float(getter("ID_VFOV")), float(getter("ID_HFOV")))


def extract_intrisic_calibration(getter):
    return lambda n: getter('calibration')[n]

def extract_intrinsics_modules_angles(getter):
    """return 'mu,nu' from Pixell Sensor.
    """
    getter = extract_intrisic_calibration(getter)
    return dict(ID_CHANNEL_ANGLE_AZIMUT=np.deg2rad(getter("ID_CHANNEL_ANGLE_AZIMUT")), ID_CHANNEL_ANGLE_ELEVATION=np.deg2rad(getter("ID_CHANNEL_ANGLE_ELEVATION")))

def extract_intrinsics_static_noise(getter):
    """Return static noise from intrinsic calibration in the sensor.
    """
    getter = extract_intrisic_calibration(getter)
    return getter('ID_STATIC_NOISE')

def extract_intrinsics_timebase_delays(getter):
    """Return timebase_delays from intrinsic calibration in the sensor.
    """
    getter = extract_intrisic_calibration(getter)
    return np.array(getter('ID_TIMEBASE_DELAY'))


def get_specs(dev):
    d = extract_specs(dev.get_property_value)
    v, h = d['v'], d['h']
    ordered_to_sampled, sampled_to_ordered, ordered_to_sampled_inv, sampled_to_ordered_inv = get_sampling_ordering_flat(v, h)

    d["ordered_to_sampled"] = ordered_to_sampled
    d["sampled_to_ordered"] = sampled_to_ordered
    d["ordered_to_sampled_inv"] = ordered_to_sampled_inv
    d["sampled_to_ordered_inv"] = sampled_to_ordered_inv
    d["bank_offsets"] = get_bank_offsets(v, h)

    return d



