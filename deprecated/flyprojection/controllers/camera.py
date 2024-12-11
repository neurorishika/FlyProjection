import PySpin

import pickle, threading, queue
import numpy as np
import time
import os

import skvideo

skvideo.setFFmpegPath("/usr/bin/")
import skvideo.io

_SYSTEM = None


class CameraError(Exception):
    """
    Exception raised when an error occurs in a camera.
    """

    pass


def list_cameras():
    """
    Return a list of connected Spinnaker cameras. Also initializes the PySpin `System`, if needed.
    """
    global _SYSTEM
    if _SYSTEM is None:
        _SYSTEM = PySpin.System.GetInstance()
    return _SYSTEM.GetCameras()


def video_writer(save_queue, writer, debug=False, fps=10):
    """
    Write images to disk.
    """
    image_counter = 0
    while True:
        image = save_queue.get()
        if debug:
            if image_counter % 10*fps == 0:
                print(f"Writing frame {image_counter}", end="\r")
        if image is None:
            break
        else:
            image_counter += 1
            writer.writeFrame(image)
            save_queue.task_done()
    return


class SpinnakerCamera:
    """
    A class used to encapsulate a Spinnaker camera
    
    Variables:
        cam : Camera object (PySpin.Camera)
        running : True if acquiring images (bool)
        initialized : True if the camera has been initialized (bool)
    
    Methods:
        init : Initializes the camera.  Automatically called if the camera is opened using a `with` clause.
        close : Closes the camera and cleans up.  Automatically called if the camera is opening using a `with` clause.
        start : Start recording images.
        stop : Stop recording images.
        get_raw_image : Get an raw image from the camera.
        get_array : Get an image from the camera, and convert[x_min:x_max, y_min:y_max] it to a numpy/cupy array.
    """

    def __init__(
        self,
        index=0,
        FPS=32,
        CAMERA_FORMAT="Mono8",
        EXPOSURE_TIME=15000,
        GAIN=20,
        GAMMA=1.0,
        record_video=True,
        video_output_path=None,
        video_output_name=None,
        lossless=True,
        debug=False,
    ):
        """
        Initialize the camera object.

        Variables:
            index : camera index (int)
            CAMERA_FORMAT : Format of the camera image. (str)
            EXPOSURE_TIME : Exposure time in microseconds. (int)
            GAIN : Gain in dB. (int)
            GAMMA : Gamma value. (int)
            MAX_FRAME_RATE : Maximum frame rate. (int)
            record_video : If True, record a video. (bool)
            video_output_path : Path to save the video. (str)
            video_output_name : Name of the video. (str)
            show_video : If True, show the video. (bool)
            show_every_n : If not None, show every nth frame. (int)
            ffmpeg_path : Path to ffmpeg. (str)
        """
        
        self.initialized = False

        self.record_video = record_video
        if self.record_video:
            assert video_output_path is not None, "video_output_path must be specified if record_video is True"
            assert video_output_name is not None, "video_output_name must be specified if record_video is True"
            self.video_output_path = video_output_path
            self.video_output_name = video_output_name


        self.CAMERA_FORMAT = CAMERA_FORMAT
        self.EXPOSURE_TIME = EXPOSURE_TIME
        self.GAIN = GAIN
        self.GAMMA = GAMMA

        cam_list = list_cameras()
        if not cam_list.GetSize():
            raise CameraError("No cameras detected.")
        if isinstance(index, int):
            self.cam = cam_list.GetByIndex(index)
        elif isinstance(index, str):
            self.cam = cam_list.GetBySerial(index)
        cam_list.Clear()
        print(f"Camera {self.cam.GetUniqueID()} selected.")

        self.running = False
        self.lossless = lossless
        self.FPS = int(FPS)
        self.debug = debug

    def init(self):
        """
        Initializes the camera setup.
        """
        self.cam.Init()
        print("Camera initialized.")

        # load default attributes
        self.cam.UserSetSelector.SetValue(PySpin.UserSetSelector_Default)
        self.cam.UserSetLoad()
        print("Loaded default attributes.")

        # set stream buffer to newest only
        self.cam.TLStream.StreamBufferHandlingMode.SetValue(PySpin.StreamBufferHandlingMode_NewestOnly)
        print("Stream buffer set to newest only.")

        # set acquisition mode to continuous
        self.cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
        print("Acquisition mode set to continuous.")

        # turn off auto exposure and set exposure time
        self.cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
        self.cam.ExposureTime.SetValue(self.EXPOSURE_TIME)
        print(f"Exposure time set to {self.EXPOSURE_TIME} microseconds.")

        # turn off auto gain and set gain
        self.cam.GainAuto.SetValue(PySpin.GainAuto_Off)
        self.cam.Gain.SetValue(self.GAIN)
        print(f"Gain set to {self.GAIN} dB.")

        # set Gamma value
        self.cam.GammaEnable.SetValue(True)
        self.cam.Gamma.SetValue(self.GAMMA)
        print(f"Gamma set to {self.GAMMA}.")

        # set pixel format
        if self.CAMERA_FORMAT == "Mono8":
            self.cam.PixelFormat.SetValue(PySpin.PixelFormat_Mono8)
        elif self.CAMERA_FORMAT == "Mono16":
            self.cam.PixelFormat.SetValue(PySpin.PixelFormat_Mono16)

        if self.record_video:
            self.timestamps = []
            # save as lossless video at 10 fps
            self.writer = skvideo.io.FFmpegWriter(
                os.path.join(self.video_output_path, self.video_output_name + ".mp4"),
                inputdict={"-r": str(self.FPS)},
                outputdict={"-r": str(self.FPS), "-c:v": "libx264", "-crf": "0"} if self.lossless else {"-r": str(self.FPS), "-c:v": "libx264"},
            )
            self.save_queue = queue.Queue()
            self.save_thread = threading.Thread(target=video_writer, args=(self.save_queue, self.writer, self.debug, self.FPS))

        self.initialized = True
        print("Camera initialized.")

    def __enter__(self):
        """
        Initializes the camera setup using a context manager.
        """
        self.init()
        return self

    def close(self):
        """
        Closes the camera and cleans up.
        """

        self.stop()
        del self.cam
        self.initialized = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Closes the camera and cleans up using a context manager.
        """
        self.close()
        print("Successfully closed camera. Exiting SpinnakerCamera Context.")

    def start(self):
        """
        Start image acquisition.
        """
        if not self.running:
            self.cam.BeginAcquisition()
            if self.record_video:
                self.save_thread.start()
            self.running = True
            self.time_of_last_frame = time.time()

    def stop(self):
        """
        Stop image acquisition.
        """
        if self.running:
            self.cam.EndAcquisition()
            if self.record_video:
                self.save_queue.join()
                self.writer.close()
                with open( os.path.join(self.video_output_path, self.video_output_name + "_timestamps.pkl"), "wb") as f:
                    pickle.dump(self.timestamps, f)
        self.running = False

    def get_raw_image(self, wait=True):
        """
        Get a raw image from the camera.

        Variables:
            wait : If True, wait for an image to be acquired.  Throw an error if no image is available. (bool)
        
        Returns:
            image : Image from the camera (PySpin.Image)
        """
        return self.cam.GetNextImage(PySpin.EVENT_TIMEOUT_INFINITE if wait else PySpin.EVENT_TIMEOUT_NONE)

    def get_array(self, wait=True, get_chunk=False, dont_save=False, crop_bounds=None, mask=None):
        """
        Get an image from the camera, and convert it to a NumPy/CuPy array.

        Variables:
            wait : If True, wait for an image to be acquired.  Throw an error if no image is available. (bool)
            get_chunk : If True, return chunk data (bool)
            dont_save : Skip saving the data
            crop_bounds : crop the image before saving (list of ints: [x_min, x_max, y_min, y_max])

        Returns:
            image : Image from the camera (numpy.ndarray/cupy.ndarray)
            chunk : PySpin chunk data (PySpin)
        """

        img = self.get_raw_image(wait)

        dtype = np.uint8 if self.CAMERA_FORMAT == "Mono8" else np.uint16
        arr = np.array(img.GetData(), dtype=dtype).reshape(img.GetHeight(), img.GetWidth())

        if crop_bounds is not None:
            assert len(crop_bounds) == 4, "crop_bounds must be a list of 4 integers"
            x_min, x_max, y_min, y_max = crop_bounds
            arr = arr[y_min:y_max, x_min:x_max].copy()
        
        if mask is not None:
            assert mask.shape == arr.shape, "mask must have the same shape as the image"
            assert mask.dtype == np.uint8, "mask must be a binary image"
            arr = arr * mask

        if self.record_video and not dont_save:
            self.timestamps.append(img.GetTimeStamp())
            self.save_queue.put(arr)

        if get_chunk:
            return arr, img.GetChunkData()
        else:
            return arr