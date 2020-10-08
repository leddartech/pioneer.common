from pioneer.common.logging_manager import LoggingManager

import numpy as np
import os
import skvideo
import time

from skvideo.io import FFmpegWriter

class Video():

    __create_key = object()

    def __init__(self, create_key, filename, writer):
        assert(create_key == Video.__create_key), \
            "Video objects must be created using Video.create"

        self.filename = filename
        self.writer = writer
    
    @classmethod
    def create(cls, hertz, filename):
        """
        Creates a new instance of currently recording video

        window: the window from which the video is recording
        synchronized: synchronized datasources
        platform: dataset platform
        """

        skvideo.setFFmpegPath('./skvideo/')
        #TODO: Side effect, should try catch or return Result
        writer = FFmpegWriter(filename, inputdict = {'-r': str(hertz)})
        
        print(f"Creating video file {filename} in directory {os.getcwd()}...")
            
        return Video(cls.__create_key, filename, writer)
   

    def update(self, np_image):
        """
        Writes a new video frame with image passed as argument
        np_image: TODO: What is the format of the image? Numpy array?
        """
        if np_image.size > 0:
            self.writer.writeFrame(np_image) 

    def stop(self):
        """
        Stops the recording
        """
        print(f"Closing video file {self.filename} in directory {os.getcwd()}...")
        self.writer.close()
            


class VideoRecorder(object):

    __create_key = object()

    def __init__(self, create_key, hertz, recordable, datasource):
        assert(create_key == VideoRecorder.__create_key), \
            "VideoRecorder objects must be created by using VideoRecorder.create"
        
        self.hertz = hertz
        self.recordable = recordable
        self.current_video = None
        self.datasource = datasource

    @classmethod
    def create(cls, recordable, datasource, platform, synchronized, video_fps=None):
        if video_fps is not None:
            hz = video_fps
        else:
            ts = platform[datasource].timestamps
            hz = 1e6/np.mean(ts[1:] - ts[:-1])

            if hz < 1 or hz > 100:
                hz_ = max(1, min(hz, 100)) #for the case units are wrong
                LoggingManager.instance().warning(f"[For datasource {datasource}] Clipping video framerate to {hz_} fps (was {hz} fps)")
                hz = hz_

        return VideoRecorder(cls.__create_key, hz, recordable, datasource)


    def record(self, is_recording):
        if is_recording:
            if self.current_video is None:
                filename = f"{self.datasource}_{int(time.time())}.mp4"
                self.current_video = Video.create(self.hertz, filename)
                self.recordable.on_video_created()
            else:
                image = self.recordable.get_frame()
                self.current_video.update(image)
                
        else:
            if self.current_video is not None:
                self.current_video.stop()
                self.current_video = None

class RecordableInterface():

    def on_video_created(self):
        pass

    def get_frame(self):
        pass
        