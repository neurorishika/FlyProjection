
import os
import argparse
import datetime
import time
import json
import redis
import cv2
import pickle
import numpy as np
from skg import nsphere_fit

# Initialize PyQtGraph and PySide2
import sys
from PySide2 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

# Setup camera
from flyprojection.controllers.basler_camera import BaslerCamera, list_basler_cameras
from flyprojection.utils import *

# Set up the argument parser
parser = argparse.ArgumentParser(description='Open/Closed Loop Fly Projection System')
parser.add_argument('-r', '--repo_dir', type=str, default='/media/rutalab/424d5920-5085-424f-a8c7-912f65e8393c/Rishika/FlyProjection/', help='Path to the repository directory')
parser.add_argument('-e', '--exp_dir', type=str, default='flyprojection', help='Path to the experiments directory (defaults to last experiment)')
parser.add_argument('-d', '--data_dir', type=str, default='data/', help='Path to the data directory')
parser.add_argument('-c', '--config_dir', type=str, default='configs/', help='Path to the config directory')
parser.add_argument('--process', action='store_true', help='Post process the video immediately')
parser.add_argument('--lossy', action='store_true', help='Record video in lossy format')
parser.add_argument('--nostream', action='store_true', help='Do not stream the video')
parser.add_argument('--nojpg', action='store_true', help='Do not save the images in mjpeg format')
parser.add_argument('--nocrop', action='store_true', help='Do not crop the image to the region of interest')
parser.add_argument('--debug', action='store_true', help='Run in debug mode')

# Parse the arguments
args = parser.parse_args()
exp_dir = args.exp_dir
data_dir = args.data_dir
repo_dir = args.repo_dir
config_dir = args.config_dir
stream = False if args.nostream else True
crop = False if args.nocrop else True
lossless = False if args.lossy else True
jpg = False if (args.nojpg or not lossless) else True
delay_processing = False if args.process else True
debug = True if args.debug else False

# List the cameras
print("Cameras Available:")
cameras = list_basler_cameras()

if len(cameras) == 0:
    print("No cameras found. Exiting.")
    sys.exit()
else:
    print(f"Found {len(cameras)} camera(s).")
    if len(cameras) > 1:
        print("Selecting the first camera. Change the index in main.py to select a different camera.")

if stream:
    # Connect to Redis
    r = redis.Redis(host='localhost', port=6379, db=0)
    print("Streaming the video.")

# If data_dir is not a full path, make it a full path by joining it with repo_dir
if not os.path.isabs(data_dir):
    data_dir = os.path.join(repo_dir, data_dir)
assert os.path.isdir(data_dir), f"Invalid data directory: {data_dir}"

# If exp_dir is not a full path, make it a full path by joining it with repo_dir
if not os.path.isabs(exp_dir):
    exp_dir = os.path.join(repo_dir, exp_dir)

# Make sure the experiment directory exists
assert os.path.isdir(exp_dir), f"Invalid experiment directory: {exp_dir}"

# Remove the trailing slash from the experiment directory
exp_dir = exp_dir.rstrip('/')

# Assert that there is a config.py file and experiment_logic.py file in the experiment directory and load them
assert os.path.isfile(os.path.join(exp_dir, 'config.py')), f"config.py file not found in {exp_dir}"
assert os.path.isfile(os.path.join(exp_dir, 'experiment_logic.py')), f"experiment_logic.py file not found in {exp_dir}"

# Assert that there is a rig_config.json file in the configs directory
assert os.path.isfile(os.path.join(repo_dir, config_dir, 'rig_config.json')), f"rig_config.json file not found in {os.path.join(repo_dir, config_dir)}"

# Load the rig configuration
with open(os.path.join(repo_dir, config_dir, 'rig_config.json'), 'r') as f:
    rig_config = json.load(f)

# Create a new folder for the data
exp_name = os.path.basename(exp_dir)
if not get_boolean_answer(f"Do you want to use the default experiment name '{exp_name}'? [Y/n] ", default=True):
    temp_input = input("Enter the experiment name (add '_' to the beginning to append to the default name): ")
    if temp_input.startswith('_'):
        exp_name += temp_input
    else:
        exp_name = temp_input
exp_data_dir = os.path.join(data_dir, exp_name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
os.makedirs(exp_data_dir, exist_ok=True)
print(f"Data will be saved in {exp_data_dir}")

# Copy the config.py and experiment_logic.py files to the data directory and repo directory + 'flyprojection' folder
os.system(f"cp {os.path.join(exp_dir, 'config.py')} {exp_data_dir}")
os.system(f"cp {os.path.join(exp_dir, 'experiment_logic.py')} {exp_data_dir}")
os.system(f"cp {os.path.join(exp_dir, 'config.py')} {os.path.join(repo_dir, 'flyprojection')}")
os.system(f"cp {os.path.join(exp_dir, 'experiment_logic.py')} {os.path.join(repo_dir, 'flyprojection')}")
print("Copied config.py and experiment_logic.py to data directory and repo directory.")

# Copy the rig_config.json file to the data directory
os.system(f"cp {os.path.join(repo_dir, config_dir, 'rig_config.json')} {exp_data_dir}")

# Securely load config.py and experiment_logic.py
def load_module_from_path(module_name, file_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"Could not load spec for '{module_name}' at '{file_path}'")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load config.py
config = load_module_from_path('config', os.path.join(exp_dir, 'config.py'))

# Load experiment_logic.py
experiment_logic = load_module_from_path('experiment_logic', os.path.join(exp_dir, 'experiment_logic.py'))

# Validate required variables from config.py
for var in ['BOUNDARY_COLOR', 'BOUNDARY_WIDTH', 'BACKGROUND_COLOR', 'ROI_TYPE']:
    assert hasattr(config, var), f"Variable {var} not defined in config.py"

# Validate required functions from experiment_logic.py
for func in ['setup', 'updates']:
    assert hasattr(experiment_logic, func), f"Function {func} not defined in experiment_logic.py"

# Setup display parameters
SCREEN_SIZE = (rig_config['projector_width'], rig_config['projector_height'])

# Process values from config.py
BOUNDARY_COLOR = hex_to_rgb(config.BOUNDARY_COLOR) if isinstance(config.BOUNDARY_COLOR, str) else config.BOUNDARY_COLOR
BACKGROUND_COLOR = hex_to_rgb(config.BACKGROUND_COLOR) if isinstance(config.BACKGROUND_COLOR, str) else config.BACKGROUND_COLOR
ROI_TYPE = config.ROI_TYPE
assert ROI_TYPE in ['rectangle', 'circle'], "Invalid ROI_TYPE. Must be 'rectangle' or 'circle'."

# Calculate boundary dimensions
BOUNDARY_X = (
    int(rig_config['projector_width'] * rig_config['exp_region_x'][0]),
    int(rig_config['projector_width'] * rig_config['exp_region_x'][1])
)
BOUNDARY_Y = (
    int(rig_config['projector_height'] * rig_config['exp_region_y'][0]),
    int(rig_config['projector_height'] * rig_config['exp_region_y'][1])
)
SCALE_FACTOR = (
    (BOUNDARY_X[1] - BOUNDARY_X[0]) / rig_config['physical_x'] +
    (BOUNDARY_Y[1] - BOUNDARY_Y[0]) / rig_config['physical_y']
) / 2

### PYQTGRAPH SETUP ###

app = QtWidgets.QApplication(sys.argv)

# Create the main window
window = QtWidgets.QMainWindow()
window.setWindowTitle("Experiment")

# Remove window decorations (title bar, borders)
window.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.FramelessWindowHint)

# Detect screens and select the second one
screens = app.screens()
if len(screens) > 1:
    screen = screens[1]
    print(f"Using Screen 1: {screen.name()}")
else:
    screen = screens[0]
    print("Warning: Only one screen detected. Using primary screen.")

# Move the window to the desired screen
geometry = screen.geometry()
window.setGeometry(geometry)

# Create a central widget
central_widget = QtWidgets.QWidget()
window.setCentralWidget(central_widget)

# Create a GraphicsLayoutWidget (from PyQtGraph)
graphics_widget = pg.GraphicsLayoutWidget()
graphics_widget.setBackground('k')  # Set background to black
graphics_widget.ci.setContentsMargins(0, 0, 0, 0)  # Remove outer margins

# Create a layout and add the graphics widget
layout = QtWidgets.QVBoxLayout()
layout.addWidget(graphics_widget)
layout.setContentsMargins(0, 0, 0, 0)  # Set layout margins to zero
central_widget.setLayout(layout)

# Create a ViewBox within the graphics widget
view = graphics_widget.addViewBox()
view.setAspectLocked(True)  # Lock aspect ratio
view.setRange(QtCore.QRectF(0, 0, SCREEN_SIZE[0], SCREEN_SIZE[1]))
view.setBackgroundColor('k')  # Set background to black

# Show the window in full screen
window.showFullScreen()

# Setup camera
camera = BaslerCamera(
    index=rig_config['CAM_INDEX'],
    FPS=rig_config['FPS'],
    EXPOSURE_TIME=rig_config['CAM_EXPOSURE_TIME'],
    GAIN=rig_config['CAM_GAIN'],
    WIDTH=rig_config['CAM_WIDTH'],
    HEIGHT=rig_config['CAM_HEIGHT'],
    OFFSETX=rig_config['CAM_OFFSETX'],
    OFFSETY=rig_config['CAM_OFFSETY'],
    CAMERA_FORMAT=rig_config['CAM_FORMAT'],
    TRIGGER_MODE="Continuous",
    record_video=True if not debug else False,
    video_output_path=exp_data_dir,
    video_output_name=exp_name,
    lossless=lossless,
    debug=debug
)

# Initialize and start the camera
camera.init()
camera.start()
print("Camera started.")

# Function to clean up resources on exit
def cleanup():
    try:
        camera.stop()
        camera.close()
        print("\nCamera stopped.")
    except Exception as e:
        print(f"Error during cleanup: {e}")
    
    # After the application exits
    print("\nExperiment completed.")

    # Save the frame metadata as a pickle file
    with open(os.path.join(exp_data_dir, 'frame_metadata.pkl'), 'wb') as f:
        pickle.dump(frame_metadata, f)

    # If stream is True, send a white image to the stream to indicate the end
    if stream:
        white_image = np.ones((rig_config['CAM_HEIGHT'], rig_config['CAM_WIDTH']), dtype=np.uint8) * 255
        _, image_bytes = cv2.imencode('.png', white_image)
        r.set('latest_image', image_bytes.tobytes())

    commands_to_run = []

    # Split the video into phases
    if not 'SPLIT_TIMES' in locals():
        SPLIT_TIMES = []

    if len(SPLIT_TIMES) > 0:
        print("Splitting the video into phases...")
        SPLIT_TIMES = [0] + SPLIT_TIMES + [elapsed_time]
        # add a few frames worth of time to the end of each phase
        SPLIT_TIMES = [time + 2/rig_config['FPS'] for time in SPLIT_TIMES]
        for i in range(len(SPLIT_TIMES)-1):
            print(f"Splitting phase {i+1}...")
            # ffmpeg command to split the video
            if jpg:
                encoding = '-c:v mjpeg -qscale:v 1' if lossless else '-c:v mjpeg -qscale:v 10'
            else:
                encoding = '-c:v libx264 -crf 0' if lossless else '-c:v libx264'
            command = f"ffmpeg -i {os.path.join(exp_data_dir, exp_name)}.mp4 -ss {SPLIT_TIMES[i]} -to {SPLIT_TIMES[i+1]} {encoding+' '}{os.path.join(exp_data_dir, exp_name)}_phase_{i+1}.mp4"
            if delay_processing:
                commands_to_run.append(command)
            else:
                os.system(command)
        print("Video split into phases.")
        split = True
    else:
        split = False

    if delay_processing:
        # create a bash script to run the commands
        with open(os.path.join(exp_data_dir, 'process.sh'), 'w') as f:
            f.write("#!/bin/bash\n")
            for command in commands_to_run:
                # replace the absolute paths with relative paths
                command = command.replace(os.path.join(exp_data_dir, exp_name), f"./{exp_name}")
                f.write(command + '\n')
            # write a command to create a empty file to indicate that the processing is complete
            f.write(f"touch ./processing_complete")
        os.system(f"chmod +x {os.path.join(exp_data_dir, 'process.sh')}")
        print("Commands saved to process.sh. Run this script to process the video.")


# Connect the cleanup function to the application's aboutToQuit signal
app.aboutToQuit.connect(cleanup)

# ROI selection and alignment using PyQtGraph
if not debug:
    # Capture an image for ROI selection
    print("Capturing image for ROI selection...")
    image = camera.get_array(dont_save=True)
    if image is None:
        print("Error: Failed to capture image for ROI selection.")
        sys.exit(1)

    # Create a dialog for ROI selection
    class ROISelDialog(QtWidgets.QDialog):
        def __init__(self, image, roi_type, parent=None):
            super(ROISelDialog, self).__init__(parent)
            self.setWindowTitle("ROI Selection")
            self.image = image
            self.roi_type = roi_type
            self.points = []
            self.num_points = 5 if roi_type == 'circle' else 4

            # Create a layout
            layout = QtWidgets.QVBoxLayout(self)

            # Create a GraphicsLayoutWidget
            self.graphics_widget = pg.GraphicsLayoutWidget()
            layout.addWidget(self.graphics_widget)

            # Create a ViewBox
            self.view = self.graphics_widget.addViewBox()
            self.view.setAspectLocked(True)
            self.view.invertY(False)
            self.view.setMouseEnabled(x=False, y=False)

            # Display the image
            self.image_item = pg.ImageItem()
            self.view.addItem(self.image_item)
            self.image_item.setImage(self.image.T)

            # Fit the image to the view
            self.view.setRange(QtCore.QRectF(0, 0, self.image.shape[1], self.image.shape[0]), padding=0)

            # Instructions label
            instruction_text = "Please click {} points to define the {} ROI.".format(
                self.num_points, 'circular' if roi_type == 'circle' else 'rectangular')
            self.instructions = QtWidgets.QLabel(instruction_text)
            layout.addWidget(self.instructions)

            # Connect the mouse click signal
            self.image_item.scene().sigMouseClicked.connect(self.mouseClicked)

            # Button box with OK and Cancel buttons
            self.button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
            self.button_box.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(False)  # Disabled until enough points are selected
            self.button_box.accepted.connect(self.accept)
            self.button_box.rejected.connect(self.reject)
            layout.addWidget(self.button_box)

            # Resize the dialog
            self.resize(800, 600)

        def mouseClicked(self, event):
            if event.button() == QtCore.Qt.LeftButton:
                pos = event.scenePos()
                mouse_point = self.image_item.mapFromScene(pos)
                x = mouse_point.x()
                y = mouse_point.y()
                if 0 <= x < self.image.shape[1] and 0 <= y < self.image.shape[0]:
                    self.points.append((x, y))
                    # Draw a circle at the clicked point
                    circle = QtWidgets.QGraphicsEllipseItem(x - 5, y - 5, 10, 10)
                    circle.setPen(pg.mkPen('r'))
                    self.view.addItem(circle)
                    # Keep track of the circle items so we can remove them if needed
                    if not hasattr(self, 'circle_items'):
                        self.circle_items = []
                    self.circle_items.append(circle)
                    if len(self.points) >= self.num_points:
                        self.button_box.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(True)
            elif event.button() == QtCore.Qt.RightButton:
                # Right-click to remove the last point
                if self.points:
                    self.points.pop()
                    # Remove the last circle item
                    if self.circle_items:
                        circle = self.circle_items.pop()
                        self.view.removeItem(circle)
                    # Disable OK button if not enough points
                    if len(self.points) < self.num_points:
                        self.button_box.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(False)
                else:
                    # No points to remove
                    pass

    # Create the ROI selection dialog
    roi_dialog = ROISelDialog(image, ROI_TYPE)
    if roi_dialog.exec_() == QtWidgets.QDialog.Accepted:
        points = np.array(roi_dialog.points)
        # Process the selected points
        if ROI_TYPE == 'circle':
            # Fit a circle to the points
            radius, center = nsphere_fit(points, 1)
            x_min = int(center[0] - radius)
            x_max = int(center[0] + radius)
            y_min = int(center[1] - radius)
            y_max = int(center[1] + radius)
            # Create a mask for the circle
            Y, X = np.ogrid[:image.shape[0], :image.shape[1]]
            dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
            mask = dist_from_center <= radius
            # Save the ROI data
            roi_dict = {
                'center': center.tolist(),
                'radius': radius,
                'points': points.tolist(),
                'type': 'circle',
                'x_min': x_min,
                'x_max': x_max,
                'y_min': y_min,
                'y_max': y_max
            }
        else:
            # Rectangle
            x_coords = points[:, 0]
            y_coords = points[:, 1]
            x_min = int(np.min(x_coords))
            x_max = int(np.max(x_coords))
            y_min = int(np.min(y_coords))
            y_max = int(np.max(y_coords))
            # Create a mask for the rectangle
            mask = np.zeros_like(image, dtype=np.uint8)
            mask[y_min:y_max, x_min:x_max] = 1
            # Save the ROI data
            roi_dict = {
                'points': points.tolist(),
                'type': 'rectangle',
                'x_min': x_min,
                'x_max': x_max,
                'y_min': y_min,
                'y_max': y_max
            }
        # Save the ROI to a file
        with open(os.path.join(exp_data_dir, 'roi.json'), 'w') as f:
            json.dump(roi_dict, f)

        print("ROI set successfully.")
        crop_bounds = [x_min, x_max, y_min, y_max]

        # print the size of the image and the mask
        print(f"Image size: {image.shape}")
        print(f"Mask size: {mask.shape}")
    else:
        print("ROI selection canceled.")
        sys.exit(1)
else:
    crop_bounds = None
    mask = None

# Setup a frame metadata list
frame_metadata = []

# Press enter to start the experiment
input("Press Enter to start the experiment.")

# Initialize experiment logic
experiment_logic.setup(view, config, rig_config)

# Variables to keep track of time and frames
start_time = time.time()
n_frames = 0
running = True

# Define update function
def update():
    global running, n_frames, start_time, frame_metadata, elapsed_time, config, experiment_logic, view, rig_config

    # clear the view
    view.clear()

    try:
        n_frames += 1
        elapsed_time = time.time() - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds (FPS: {n_frames/elapsed_time:.2f})", end='\r')

        # Get an image from the camera
        image = camera.get_array(crop_bounds=crop_bounds if crop else None, mask=mask if crop else None)

        if image is None:
            print("Error: Failed to capture image from the camera.")
            return

        if stream:
            # Encode the image and store it in Redis
            _, image_bytes = cv2.imencode('.png', image)
            r.set('latest_image', image_bytes.tobytes())

        # Call experiment logic updates
        experiment_logic.updates(view, rig_config, elapsed_time)

        # Collect frame metadata if needed
        frame_metadata.append({
            'time': elapsed_time,
            # Add other metadata as needed
        })

        # Check if the experiment is still running
        if not experiment_logic.running:
            timer.stop()
            app.quit()

    except Exception as e:
        print(f"Unexpected error during update: {e}")
        timer.stop()
        app.quit()

# Create a timer for the update loop
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(int(1000 / rig_config['FPS']))  # Update rate in milliseconds

# Handle key press events to exit on ESC
def keyPressEvent(event):
    if event.key() == QtCore.Qt.Key_Escape:
        app.quit()
window.keyPressEvent = keyPressEvent

# Start the application event loop
sys.exit(app.exec_())

