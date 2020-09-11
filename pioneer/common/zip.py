import pioneer.common.constants as Constants

from distutils.version import LooseVersion
from ruamel.std import zipfile

import cv2
import io
import numpy as np
import pickle


class ZipFileWriter(object):

    def __init__(self, filepath, mode='w'):
        self.filepath = filepath
        self.archive = zipfile.ZipFile(self.filepath, mode=mode)
        self.is_opened = True

    def __del__(self):
        self.close()
    
    def reopen(self, mode='a'):
        if not self.is_opened:
            self.archive = zipfile.ZipFile(self.filepath, mode=mode)
            self.is_opened = True

    def close(self):
        if self.is_opened:
            self.archive.close()
            self.is_opened = False

    def writestr(self, name, data):
        self.archive.writestr(name, data)

    def write_pickle(self, name, data):
        data_bytes = pickle.dumps(data)
        self.archive.writestr(name, data_bytes)

    def write_numbered_pickle(self, index, data):
        name = Constants.NUMBERED_PICKLE_FMT.format(index)
        self.write_pickle(name, data)

    def write_array_as_txt(self, name, data, encoding='utf8', fmt='%.18e'):
        with io.BytesIO() as f:
            if LooseVersion(np.__version__) >= LooseVersion('1.14.0'):
                np.savetxt(f, data, encoding=encoding, fmt=fmt)
            else:
                np.savetxt(f, data, fmt=fmt)
            f.seek(0)
            data_bytes = f.read()
            self.archive.writestr(name, data_bytes)

    def write_array_as_png(self, name, data):
        ret, png = cv2.imencode('.png', data)
        assert ret, 'Failed to encode array as png'
        data_bytes = png.tobytes()
        self.archive.writestr(name, data_bytes)

    def write_array_as_jpg(self, name, data):
        ret, jpg = cv2.imencode('.jpg', data)
        assert ret, 'Failed to encode array as jpg'
        data_bytes = jpg.tobytes()
        self.archive.writestr(name, data_bytes)

    def write_array_as_numbered_png(self, index, data):
        name = Constants.NUMBERED_PNG_FMT.format(index)
        return self.write_array_as_png(name, data)

    def write_array_as_numbered_jpg(self, index, data):
        name = Constants.NUMBERED_JPG_FMT.format(index)
        return self.write_array_as_jpg(name, data)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()
