# Assuming BaslerCamera is properly imported and configured
from flyprojection.controllers.basler_camera import list_basler_cameras, BaslerCamera
from flyprojection.behavior.tracker import Tracker, FastTracker
import numpy as np
import time

# Initialize the camera
camera = BaslerCamera(
    index=0,
    FPS=100,
    EXPOSURE_TIME=9000,
    GAIN=0.0,
    WIDTH=2048,
    HEIGHT=2048,
    OFFSETX=248,
    OFFSETY=0,
    TRIGGER_MODE="Continuous",
    CAMERA_FORMAT="Mono8",
    record_video=False,
    video_output_path=None,
    video_output_name=None,
    lossless=True,
    debug=False,
)

# Initialize the tracker with the camera and desired parameters
tracker = FastTracker(
    camera=camera,
    n_targets=5,
    debug=True  # Set to True to see the tracking visualization
)

# Start the camera
camera.start()

times = []
try:
    while True:
        estimates = tracker.process_next_frame()
        # if estimates is not None:
        #     for estimate in estimates:
        #         print(f"ID: {estimate['id']}, Position: {estimate['position']}, "
        #               f"Velocity: {estimate['velocity']}, Angle: {np.rad2deg(estimate['angle'])}°, "
        #               f"Time Since Start: {estimate['time_since_start']:.2f}s")
        times.append(time.time())
except Exception as e:
    print(e)
finally:
    # Release resources
    tracker.release_resources()
    camera.stop()
    print(f"Average FPS: {len(times) / (times[-1] - times[0])}")

