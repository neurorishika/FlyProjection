from pypylon import pylon

import pickle, threading, queue
import numpy as np
import time
import os
import signal
import torch

import skvideo
import skvideo.io

def video_writer(save_queue, writer, debug=False, fps=10):
    """
    Write images to disk.
    """
    image_counter = 0
    while True:
        image = save_queue.get()
        if image is None:
            break
        else:
            image_counter += 1
            writer.writeFrame(image)
            save_queue.task_done()

        if debug:
            if image_counter % 10*fps == 0:
                print(f"Writing frame {image_counter}", end="\r")
    return

camera_max_fps = {
    "a2A2448-105g5m": 106,
}

def list_basler_cameras():
    """
    Return a list of connected Basler cameras.
    """
    tlf = pylon.TlFactory.GetInstance()
    dev = tlf.EnumerateDevices()
    names = [str(d.GetModelName())+"_"+str(d.GetSerialNumber()) for d in dev]
    for name in names:
        print(name)
    return dev

class BaslerCamera:
    """
    A class used to encapsulate a Basler camera
    
    Variables:
        cam : Camera object (pylon.InstantCamera)
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
        FPS=100,
        EXPOSURE_TIME=9000,
        GAIN=0,
        WIDTH=2048,
        HEIGHT=2048,
        OFFSETX=224,
        OFFSETY=0,
        TRIGGER_MODE="Continuous",
        CAMERA_FORMAT="Mono8",
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
            FPS : Frame rate. (int)
            EXPOSURE_TIME : Exposure time in microseconds. (int)
            GAIN : Gain in dB. (int)
            WIDTH : Width of the image. (int)
            HEIGHT : Height of the image. (int)
            OFFSETX : X offset of the image. (int)
            OFFSETY : Y offset of the image. (int)
            TRIGGER_MODE : Trigger mode. (str: "Continuous" or "Software" or "Hardware")
            CAMERA_FORMAT : Format of the camera image. (str)
            record_video : If True, record a video. (bool)
            video_output_path : Path to save the video. (str)
            video_output_name : Name of the video. (str)
            lossless : If True, save the video as lossless. (bool)
            debug : If True, print debug messages. (bool)
        """

        self.initialized = False

        self.record_video = record_video
        if self.record_video:
            assert video_output_path is not None, "video_output_path must be specified if record_video is True"
            assert video_output_name is not None, "video_output_name must be specified if record_video is True"
            self.video_output_path = video_output_path
            self.video_output_name = video_output_name

        # setup TriggerModes
        self.TRIGGER_MODE = TRIGGER_MODE
        if self.TRIGGER_MODE == "Continuous":
            # Ignore FPS
            print("Trigger mode set to Continuous. FPS will be ignored.")
            self.FPS = 0
            self.EXPOSURE_TIME = EXPOSURE_TIME
        elif self.TRIGGER_MODE == "Software":
            print("Trigger mode set to Software Trigger.")
            # limit FPS to 1e6/EXPOSURE_TIME
            self.FPS = int(FPS)
            assert self.FPS > 0, "FPS must be greater than 0."
            assert self.FPS <= 1e6/EXPOSURE_TIME, "FPS must be less than 1e6/EXPOSURE_TIME"
            self.EXPOSURE_TIME = EXPOSURE_TIME
        elif self.TRIGGER_MODE == "Hardware":
            # not implemented yet
            raise Exception("Hardware trigger mode not implemented yet.")

        self.GAIN = GAIN
        self.WIDTH =  WIDTH
        self.HEIGHT = HEIGHT
        self.OFFSETX = OFFSETX
        self.OFFSETY = OFFSETY
        self.CAMERA_FORMAT = CAMERA_FORMAT

        self.cam = None

        # get camera
        print("Getting camera...")
        cameras = list_basler_cameras()
        if not cameras:
            raise Exception("No cameras detected.")
        else:
            print(f"Found {len(cameras)} cameras.")

        if isinstance(index, int):
            self.cam = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(cameras[index]))
        elif isinstance(index, str):
            # match the serial number
            for cam in cameras:
                if str(cam.GetSerialNumber()) == index:
                    self.cam = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(cam))
                    break
            if self.cam is None:
                raise Exception(f"Camera with serial number {index} not found.")
        else:
            raise Exception("index must be an integer or string.")
        
        print(f"Camera {self.cam.GetDeviceInfo().GetModelName()}_{self.cam.GetDeviceInfo().GetSerialNumber()} selected.")

        self.running = False
        self.lossless = lossless
        self.debug = debug
        signal.signal(signal.SIGINT, self.stop)

    def init(self):
        """
        Initializes the camera setup.
        """

        # assert that its not an USB camera
        assert not self.cam.IsUsb(), "Camera is USB.  Use GigE or CoaXPress cameras."

        if self.TRIGGER_MODE == "Continuous":
            # Register the standard event handler for configuring continuous single frame acquisition.
            self.cam.RegisterConfiguration(pylon.AcquireContinuousConfiguration(), pylon.RegistrationMode_ReplaceAll, pylon.Cleanup_Delete)
            print("Registered continuous acquisition configuration.")
        elif self.TRIGGER_MODE == "Software":
            # Register the standard event handler for configuring software triggered single frame acquisition.
            self.cam.RegisterConfiguration(pylon.SoftwareTriggerConfiguration(), pylon.RegistrationMode_ReplaceAll, pylon.Cleanup_Delete)
            print("Registered software trigger configuration.")

        # Set MaxNumBuffer to 15
        if self.cam.GetDeviceInfo().GetModelName() in camera_max_fps.keys():
            self.cam.MaxNumBuffer = 1000 #int(2*60*60*camera_max_fps[self.cam.GetDeviceInfo().GetModelName()]) # 2 hours of max FPS
            print(f"MaxNumBuffer set to {self.cam.MaxNumBuffer}.")
        else:
            raise Exception(f"Camera model {self.cam.GetDeviceInfo().GetModelName()} not found in camera_max_fps.")

        # Open the camera.
        self.cam.Open()
        print("Camera initialized.")

        # Turn off auto exposure and set exposure time
        self.cam.ExposureAuto.SetValue("Off")
        self.cam.ExposureTime.SetValue(self.EXPOSURE_TIME)
        print(f"Exposure time set to {self.EXPOSURE_TIME} microseconds.")

        # Turn off auto gain and set gain
        self.cam.GainAuto.SetValue("Off")
        self.cam.Gain.SetValue(self.GAIN)
        print(f"Gain set to {self.GAIN} dB.")

        # get viable pixel formats
        pixel_formats = self.cam.PixelFormat.Symbolics
        assert self.CAMERA_FORMAT in pixel_formats, f"Invalid pixel format.  Must be one of {pixel_formats}"
        self.cam.PixelFormat.SetValue(self.CAMERA_FORMAT)

        # set width and height
        self.cam.Width.SetValue(self.WIDTH)
        self.cam.Height.SetValue(self.HEIGHT)
        self.cam.OffsetX.SetValue(self.OFFSETX)
        self.cam.OffsetY.SetValue(self.OFFSETY)
        print(f"Set width to {self.WIDTH}, height to {self.HEIGHT}, offset to ({self.OFFSETX}, {self.OFFSETY}).")

        # get frequency
        self.TSFREQ = self.cam.GevTimestampTickFrequency.GetValue()

        if self.record_video:
            print("Setting up video recording...")
            self.timestamps = []
            # save as lossless video at 10 fps
            self.writer = skvideo.io.FFmpegWriter(
                os.path.join(self.video_output_path, self.video_output_name + ".mp4"),
                # inputdict={"-r": str(self.FPS)},
                #outputdict={"-r": str(self.FPS), "-c:v": "libx264", "-crf": "0"} if self.lossless else {"-r": str(self.FPS), "-c:v": "libx264"},
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
        self.cam.Close()
        del self.cam
        self.initialized = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Closes the camera and cleans up using a context manager.
        """
        self.close()
        print("Successfully closed camera. Exiting BaslerCamera Context.")

    def start(self):
        """
        Start image acquisition.
        """
        if not self.running:
            self.cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            if self.record_video:
                self.save_thread.start()
            self.running = True
            self.time_of_last_frame = time.time()

    def stop(self):
        """
        Stop image acquisition.
        """
        if self.running:
            self.cam.StopGrabbing()
            if self.record_video:
                # stop the thread
                self.save_queue.put(None)
                self.save_queue.join()
                self.save_thread.join()
                self.writer.close()
                with open( os.path.join(self.video_output_path, self.video_output_name + "_timestamps.pkl"), "wb") as f:
                    pickle.dump(self.timestamps, f)
        self.running = False

    def get_raw_image(self, timeout=1000):
        """
        Get a raw image from the camera.

        Variables:
            timeout : Timeout in milliseconds. (int)
        
        Returns:
            image : Image from the camera (pylon.GrabResult)
            timestamp : Timestamp of the image (float)
        """
        if self.TRIGGER_MODE == "Continuous":
            result = self.cam.RetrieveResult(timeout, pylon.TimeoutHandling_ThrowException)
            if result.GrabSucceeded():
                return result, time.time()
            else:
                print("Error: Grab failed due to {0}".format(result.GetErrorDescription()))
                return None, None
        elif self.TRIGGER_MODE == "Software":
            if self.cam.IsGrabbing():
                # send software trigger
                if self.cam.WaitForFrameTriggerReady(timeout, pylon.TimeoutHandling_ThrowException):
                    self.cam.ExecuteSoftwareTrigger()
                    result = self.cam.RetrieveResult(timeout, pylon.TimeoutHandling_ThrowException)
                    if result.GrabSucceeded():
                        return result, time.time()
                    else:
                        print("Error: Grab failed due to {0}".format(result.GetErrorDescription()))
                        return None, None

    def get_array(self, timeout=1000, dont_save=False, crop_bounds=None, mask=None):
        """
        Get an image from the camera, and convert it to a NumPy/CuPy array.

        Variables:
            timeout : Timeout in milliseconds. (int)
            dont_save : Skip saving the data
            crop_bounds : crop the image before saving (list of ints: [x_min, x_max, y_min, y_max])
            mask : mask to apply to the image (numpy.ndarray)

        Returns:
            image : Image from the camera (numpy.ndarray/cupy.ndarray)
        """

        img, time = self.get_raw_image(timeout)

        dtype = np.uint8 if self.CAMERA_FORMAT == "Mono8" else np.uint16
        arr = np.array(img.Array,dtype=dtype)

        if crop_bounds is not None:
            assert len(crop_bounds) == 4, "crop_bounds must be a list of 4 integers"
            x_min, x_max, y_min, y_max = crop_bounds
            arr = arr[y_min:y_max, x_min:x_max]
        
        if mask is not None:
            mask = mask.astype(dtype)[y_min:y_max, x_min:x_max]
            assert mask.shape == arr.shape, "mask must have the same shape as the image"
            assert mask.dtype == np.uint8 if self.CAMERA_FORMAT == "Mono8" else np.uint16, "mask must be a binary image"
            arr = arr * mask

        if self.record_video and not dont_save:
            self.timestamps.append((time, img.TimeStamp/self.TSFREQ))
            self.save_queue.put(arr)

        return arr
    
    def get_tensor(self, timeout=1000, dont_save=False, crop_bounds=None, mask=None):
        """
        Get an image from the camera and convert it to a PyTorch tensor.

        Variables:
            timeout : Timeout in milliseconds. (int)
            dont_save : Skip saving the data
            crop_bounds : Crop the image before saving (list of ints: [x_min, x_max, y_min, y_max])
            mask : Mask to apply to the image (numpy.ndarray)

        Returns:
            image : Image from the camera (torch.Tensor) on GPU if available
        """
        img, time_stamp = self.get_raw_image(timeout)

        dtype = torch.uint8 if self.CAMERA_FORMAT == "Mono8" else torch.uint16
        arr = torch.tensor(img.Array, dtype=dtype)

        if crop_bounds is not None:
            assert len(crop_bounds) == 4, "crop_bounds must be a list of 4 integers"
            x_min, x_max, y_min, y_max = crop_bounds
            arr = arr[y_min:y_max, x_min:x_max]

        if mask is not None:
            mask = torch.tensor(mask.astype(dtype)[y_min:y_max, x_min:x_max], dtype=dtype)
            assert mask.shape == arr.shape, "mask must have the same shape as the image"
            arr = arr * mask

        if self.record_video and not dont_save:
            self.timestamps.append((time_stamp, img.TimeStamp / self.TSFREQ))
            self.save_queue.put(arr.cpu())

        # Normalize to [0, 1]
        arr = arr.float() / 255.0 if self.CAMERA_FORMAT == "Mono8" else arr.float() / 65535.0

        # Add batch and channel dimensions
        arr = arr.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, H, W]

        # Move to GPU if available
        if torch.cuda.is_available():
            arr = arr.cuda()

        return arr