import warnings

class LoggingManager():

    __create_key = object()
    __instance = None

    def __init__(self, create_key):
        assert(create_key == LoggingManager.__create_key), \
            "LoggingManager objects must be created using LoggingMaager.instance"

        self.verbose = False
 
    @classmethod
    def instance(cls):
        if cls.__instance is None:
            cls.__instance = LoggingManager(cls.__create_key)
        return cls.__instance

    def set_verbosity(self, is_verbose):
        self.verbose = is_verbose

    def warning(self, msg):
        if self.verbose is True:
            warnings.warn(msg) 