from collections import OrderedDict
from typing import Tuple

import fnmatch
import os

def parse_sensor_name(name:str) -> Tuple[str, str]:
    """Parse a sensor name

        Args:
            name: the sensor name, e.g. 'pixell_tfc'
        Returns:
            A tuple with sensor's name and sensor's position id, e.g. ('pixell', 'tfc')
    """

    sensor_type, position = name.split('_')
    return sensor_type, position

def parse_datasource_name(full_ds_name:str) -> Tuple[str, str, str]:
    """Parse an expanded datasource name

        Args:
            name: the datasource name, e.g. 'pixell_tfc_ech'
        Returns:
            A tuple with datasource's sensor type, datasource's sensor position id 
            and datasource's datasource type, , e.g. ('pixell', 'tfc', 'ech')
    """
    sensor_type, position, ds_type = full_ds_name.split('_')
    return sensor_type, position, ds_type

def extract_sensor_id(full_ds_name:str) -> str:
    """Parse an expanded datasource name for sensor name

        Args:
            name: the datasource name, e.g. 'pixell_tfc_ech'
        Returns:
            The sensor name, e.g. 'pixell_tfc'
    """
    sensor_type, position, _ = full_ds_name.split('_')
    return f"{sensor_type}_{position}"

def referential_name(name:str) -> str:
    """Extracts the referential name from a sensor name or datasource name"""

    if name == 'world':
        return name

    split = name.split('_')
    
    if(len(split) == 2):
        return name
    if(len(split) == 3):
        sensor_type, position, datasource = split
        return f"{sensor_type}_{position}"
    
    raise RuntimeError("Bad name format, expected 2 or 3 '_'")

def slice_to_range(s:slice, length:int) -> range:
    """Converts a slice object to a range
        Args: 
            s: the slice object
            length: the number of elements in the collection the slice is to be used with
        Returns:
            The range instance
    """
    return range(0 if s.start is None else s.start
                 , length if s.stop is None else s.stop
                 , 1 if s.step is None else s.step)

def expand_wildcards(labels, ds_names):
    """Expands datasource names containing wildcard '*' characters

        Args:
            labels: a list of string that may contain multiple widlcard '*' characters
            ds_names: the exhaustive list of datasource names in the Platform instance
        Returns:
            A list of all expanded datasource names that matches the provided list of wildcards
    """ 
    # use and order preserving collection to preserve the ordering of the
    # provided labels
    expanded_labels = OrderedDict()
    for label in labels:
        c = len(expanded_labels)
        for ds_name in ds_names:
            if fnmatch.fnmatch(ds_name, label):
                expanded_labels[ds_name] = True

    return list(expanded_labels.keys())