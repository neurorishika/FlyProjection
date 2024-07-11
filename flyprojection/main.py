from flyprojection.utils import *
import pygame
import os
import argparse
import datetime
import time
import json
import redis
import cv2
import pickle
import scipy.stats as stats

from controllers.camera import SpinnakerCamera, list_cameras
import matplotlib.pyplot as plt
import numpy as np
from skg import nsphere_fit

# set up the argument parser
parser = argparse.ArgumentParser(description='Open/Closed Loop Fly Projection System')
parser.add_argument('-r','--repo_dir', type=str, default='/home/smellovision/FlyProjection/', help='Path to the repository directory')
parser.add_argument('-e','--exp_dir', type=str, default='flyprojection', help='Path to the experiments directory (defaults to last experiment)')
parser.add_argument('-d','--data_dir', type=str, default='data/', help='Path to the data directory')
parser.add_argument('-c','--config_dir', type=str, default='configs/', help='Path to the config directory')
parser.add_argument('--process', action='store_true', help='Post process the video immediately')
parser.add_argument('--lossy', action='store_true', help='Record video in lossy format')
parser.add_argument('--nostream', action='store_true', help='Do not stream the video')
parser.add_argument('--nojpg', action='store_true', help='Do not save the images in mjpeg format')
parser.add_argument('--nocrop', action='store_true', help='Crop the image to the region of interest')
parser.add_argument('--debug', action='store_true', help='Run in debug mode')

# parse the arguments
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

# list the cameras
cameras = list_cameras()
if len(cameras) == 0:
    print("No cameras found. Exiting.")
    exit()
else:
    print(f"Found {len(cameras)} camera(s).")
    if len(cameras) > 1:
        print("Selecting the first camera. Change the index in main.py to select a different camera.")

if stream:
    # Connect to Redis
    r = redis.Redis(host='localhost', port=6379, db=0)
    print("Streaming the video.")

# if data_dir is not a full path, make it a full path by joining it with repo_dir
if not os.path.isabs(data_dir):
    data_dir = os.path.join(repo_dir, data_dir)
assert os.path.isdir(data_dir), f"Invalid data directory: {data_dir}"

# if exp_dir is not a full path, make it a full path by joining it with repo_dir
if not os.path.isabs(exp_dir):
    exp_dir = os.path.join(repo_dir, exp_dir)

# make sure the experiment directory exists
assert os.path.isdir(exp_dir), f"Invalid experiment directory: {exp_dir}"

# remove the trailing slash from the experiment directory
exp_dir = exp_dir.rstrip('/')

# assert that there is a config.py file and experiment_logic.py file in the experiment directory and load them
assert os.path.isfile(os.path.join(exp_dir, 'config.py')), f"config.py file not found in {exp_dir}"
assert os.path.isfile(os.path.join(exp_dir, 'experiment_logic.py')), f"experiment_logic.py file not found in {exp_dir}"

# assert that there is a rig_config.json file in the configs directory
assert os.path.isfile(os.path.join(repo_dir, config_dir, 'rig_config.json')), f"rig_config.json file not found in {os.path.join(repo_dir, config_dir)}"

# load the rig configuration
with open(os.path.join(repo_dir, config_dir, 'rig_config.json'), 'r') as f:
    rig_config = json.load(f)

# create a new folder for the data
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

# copy the config.py and experiment_logic.py files to the data directory and repo directory + 'flyprojection' folder
os.system(f"cp {os.path.join(exp_dir, 'config.py')} {exp_data_dir}")
os.system(f"cp {os.path.join(exp_dir, 'experiment_logic.py')} {exp_data_dir}")
os.system(f"cp {os.path.join(exp_dir, 'config.py')} {os.path.join(repo_dir, 'flyprojection')}")
os.system(f"cp {os.path.join(exp_dir, 'experiment_logic.py')} {os.path.join(repo_dir, 'flyprojection')}")
print("Copied config.py and experiment_logic.py to data directory and repo directory.")

# copy the rig_config.json file to the data directory
os.system(f"cp {os.path.join(repo_dir, config_dir, 'rig_config.json')} {exp_data_dir}")

# load the config.py file
exec(open(os.path.join(repo_dir, 'flyprojection', 'config.py')).read())

# load the experiment_logic.py file
exec(open(os.path.join(repo_dir, 'flyprojection', 'experiment_logic.py')).read())

# assert that required variables are defined
for var in ['BOUNDARY_COLOR', 'BOUNDARY_WIDTH', 'BACKGROUND_COLOR', 'ROI_TYPE']:
    assert var in locals(), f"Variable {var} not defined in config.py"

# assert that required functions are defined
for func in ['constants', 'updates', 'setup']:
    assert func in locals(), f"Function {func} not defined in experiment_logic.py"

# Setup display and initialize Pygame
os.environ['SDL_VIDEO_WINDOW_POS'] = f"0,0"
pygame.init()

# get screen size (values will be loaded from config.py)
SCREEN_SIZE = (rig_config['projector_width'], rig_config['projector_height'])

# setup display (full screen on monitor 2 with no frame)
screen = pygame.display.set_mode(SCREEN_SIZE, pygame.NOFRAME | pygame.HWSURFACE | pygame.DOUBLEBUF)

# process the values from config.py
BOUNDARY_X = (int(rig_config['projector_width']*rig_config['exp_region_x'][0]), int(rig_config['projector_width']*rig_config['exp_region_x'][1]))
BOUNDARY_Y = (int(rig_config['projector_height']*rig_config['exp_region_y'][0]), int(rig_config['projector_height']*rig_config['exp_region_y'][1]))
SCALE_FACTOR = ((BOUNDARY_X[1]-BOUNDARY_X[0])/rig_config['physical_x'] + (BOUNDARY_Y[1]-BOUNDARY_Y[0])/rig_config['physical_y'])/2

BOUNDARY_COLOR = hex_to_rgb(BOUNDARY_COLOR) if type(BOUNDARY_COLOR) == str else BOUNDARY_COLOR if type(BOUNDARY_COLOR) == tuple else (255, 255, 255)
BACKGROUND_COLOR = hex_to_rgb(BACKGROUND_COLOR) if type(BACKGROUND_COLOR) == str else BACKGROUND_COLOR if type(BACKGROUND_COLOR) == tuple else (0, 0, 0)
assert ROI_TYPE in ['rectangle', 'circle'], "Invalid ROI_TYPE. Must be 'rectangle' or 'circle'."

# Create the display surface
pygame.display.set_caption("Fly Projection")

# Clock for controlling the frame rate
clock = pygame.time.Clock()

# Call the setup function
setup()

# start the camera
with SpinnakerCamera(
    video_output_name=exp_name, 
    video_output_path=exp_data_dir, 
    record_video=True if not debug else False,
    FPS=rig_config['FPS'], 
    lossless=lossless,
    debug=debug) as camera:
    camera.start()

    if not debug:
        # create a blue cross at the center of the working area
        center = relative_to_absolute(0.5, 0.5, rig_config)
        # draw the cross
        pygame.draw.line(screen, (255, 255, 255), (center[0]-10, center[1]), (center[0]+10, center[1]), 5)
        pygame.draw.line(screen, (255, 255, 255), (center[0], center[1]-10), (center[0], center[1]+10), 5)
        pygame.display.flip()

        # ask the user to align the arena with the cross
        input("Please align the arena with the cross on the screen and press Enter to continue.")

        # a loop to make sure the view is correct
        while True:
            # get a single image from the camera and dont save it
            image = camera.get_array(wait=True, dont_save=True)
            # show the image and ask if it is correct
            plt.imshow(image, cmap='gray')
            plt.title("Camera Feed for Arena Alignment")
            plt.show()
            if get_boolean_answer("Please confirm that the view is correct. Is the view correct? [Y/n] ", default=True):
                # save the image
                plt.imsave(os.path.join(exp_data_dir, 'arena_alignment.png'), image, cmap='gray')
                plt.close()
                break
            else:
                plt.close()
        
        # turn off the cross
        screen.fill(BACKGROUND_COLOR)
        pygame.display.flip()

        # get an image from the camera
        image = camera.get_array(wait=True, dont_save=True)

        # get an ROI from the user
        print("Please select the region of interest (ROI) for the experiment.")

        roi_set = False

        while not roi_set:

            # get a ROI from the user on the final image
            if ROI_TYPE == 'circle':
                plt.imshow(image, cmap='gray')
                plt.title("Please select the region of interest (ROI) for the experiment\nby clicking on 5 points on the edge of the ROI.")
                points = plt.ginput(5, timeout=0)
                plt.close()

                # get the center and radius of the circle by fitting a circle to the points
                points = np.array(points)
                radius, center = nsphere_fit(points, 1)

                # get the bounds of the circle
                x_min = int(center[0] - radius)
                x_max = int(center[0] + radius)
                y_min = int(center[1] - radius)
                y_max = int(center[1] + radius)

                # plot the circle and the bounds
                plt.figure()
                plt.imshow(image, cmap='gray')
                plt.plot(center[0], center[1], 'ro')
                circle = plt.Circle(center, radius, color='r', fill=False)
                plt.gca().add_artist(circle)
                plt.plot([x_min, x_max, x_max, x_min, x_min], [y_min, y_min, y_max, y_max, y_min], 'r-')
                plt.title("Selected ROI")
                plt.figure()
                temp_image = image.copy()[y_min:y_max, x_min:x_max]
                plt.imshow(temp_image, cmap='gray')
                plt.title("Cropped Image")
                plt.show()
            else:
                plt.imshow(image, cmap='gray')
                plt.title("Please select the 4 corner points of the rectangle ROI for the experiment.\nClick Order: top-left, top-right, bottom-right, bottom-left.")
                points = plt.ginput(4, timeout=0)
                plt.close()

                # fit a rectangle to the points
                points,params = rect_fit(points, force_rotation=0)

                points = np.array(points)

                # get the bounds of the rectangle
                x_min = int(np.min([point[0] for point in points]))
                x_max = int(np.max([point[0] for point in points]))
                y_min = int(np.min([point[1] for point in points]))
                y_max = int(np.max([point[1] for point in points]))

                # plot the rectangle and the bounds
                plt.figure()
                plt.imshow(image, cmap='gray')
                plt.plot([point[0] for point in points], [point[1] for point in points], 'ro')
                plt.plot([x_min, x_max, x_max, x_min, x_min], [y_min, y_min, y_max, y_max, y_min], 'r-')
                plt.title("Selected ROI")
                plt.figure()
                temp_image = image.copy()[y_min:y_max, x_min:x_max]
                plt.imshow(temp_image, cmap='gray')
                plt.title("Cropped Image")
                plt.show()

            # ask the user if the ROI is correct
            roi_set = get_boolean_answer("Is the ROI correct? [Y/n] ", default=True)

            if roi_set:
                if ROI_TYPE == 'circle':
                    # create a mask for the circle
                    mask = np.zeros_like(image, dtype=np.uint8)
                    for i in range(mask.shape[0]):
                        for j in range(mask.shape[1]):
                            if (i-center[1])**2 + (j-center[0])**2 < radius**2:
                                mask[i,j] = 1
                    mask = mask[y_min:y_max, x_min:x_max]
                    # save the center and radius
                    with open(os.path.join(exp_data_dir, 'roi.json'), 'w') as f:
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
                        json.dump(roi_dict, f)
                else:
                    # create a mask for the rectangle
                    mask = np.ones_like(image[y_min:y_max, x_min:x_max], dtype=np.uint8)
                    # save the points
                    with open(os.path.join(exp_data_dir, 'roi.json'), 'w') as f:
                        roi_dict = {
                            'points': points.tolist(), 
                            'type': 'rectangle',
                            'x_min': x_min,
                            'x_max': x_max,
                            'y_min': y_min,
                            'y_max': y_max
                        }
                        json.dump(roi_dict, f)
            else:
                plt.close()
            
        print("ROI set successfully.")

        crop_bounds = [x_min, x_max, y_min, y_max]
    else:
        crop_bounds = None
        mask = None

    # setup a frame metadata
    frame_metadata = []

    # press enter to start the experiment
    input("Press Enter to start the experiment.")

    # keep track of the time
    start_time = time.time()
    # Main loop
    running = True
    n_frames = 0
    while running:

        n_frames += 1
        # Calculate the time elapsed
        elapsed_time = time.time() - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds (FPS: {n_frames/elapsed_time:.2f})", end='')
        
        # Event handling
        for event in pygame.event.get():
            # Quit on window close
            if event.type == pygame.QUIT:
                running = False
            # Quit on escape
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # Get an image from the camera
        image = camera.get_array(wait=True, crop_bounds=crop_bounds if crop else None, mask=mask if crop else None)

        if stream:
            # encode the image and store it in Redis
            _, image_bytes = cv2.imencode('.png', image)
            r.set('latest_image', image_bytes.tobytes())

        # Clear the screen
        screen.fill(BACKGROUND_COLOR)
        
        # Show boundary (if needed)
        if debug:
            # rectangular boundary in white filled with black
            pygame.draw.rect(screen, BOUNDARY_COLOR, (BOUNDARY_X[0], BOUNDARY_Y[0], BOUNDARY_X[1]-BOUNDARY_X[0], BOUNDARY_Y[1]-BOUNDARY_Y[0]), BOUNDARY_WIDTH)

        # Call the constants function
        constants()

        # Call the updates function
        updates()

        # Update the display
        pygame.display.flip()

        # Control the frame rate
        clock.tick(rig_config['FPS'])

        # clear line
        print("\r", end='')

print("\nExperiment completed.")

# if stream is True, send a white image to the stream
if stream:
    r.set('latest_image', cv2.imencode('.png', np.ones((rig_config['projector_height'], rig_config['projector_width'], 3), dtype=np.uint8)*255)[1].tobytes())

# Quit Pygame
pygame.quit()

# save the frame metadata as a pickle file
with open(os.path.join(exp_data_dir, 'frame_metadata.pkl'), 'wb') as f:
    pickle.dump(frame_metadata, f)

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

# exit the program
exit(0)
