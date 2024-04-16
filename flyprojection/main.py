from flyprojection.utils import hex_to_rgb, relative_to_absolute, get_boolean_answer,rect_fit,get_rectangle_points
import pygame
import os
import argparse
import datetime
import time
import json

from controllers.camera import SpinnakerCamera, list_cameras
import matplotlib.pyplot as plt
import numpy as np
from skg import nsphere_fit

# set up the argument parser
parser = argparse.ArgumentParser(description='Open/Closed Loop Fly Projection System')
parser.add_argument('--repo_dir', type=str, default='/home/smellovision/FlyProjection/', help='Path to the repository directory')
parser.add_argument('--exp_dir', type=str, default='flyprojection', help='Path to the experiments directory (defaults to last experiment)')
parser.add_argument('--data_dir', type=str, default='data/', help='Path to the data directory')
parser.add_argument('--config_dir', type=str, default='configs/', help='Path to the config directory')
parser.add_argument('--debug', action='store_true', help='Run in debug mode')
parser.add_argument('--nocrop', action='store_true', help='Crop the image to the region of interest')
parser.add_argument('--lossy', action='store_true', help='Record video in lossy format')
parser.add_argument('--boundary', action='store_true', help='Show the boundary in the experiment')
parser.add_argument('--ctrax', action='store_true', help='Prepare the data for Ctrax')
parser.add_argument('--compress', action='store_true', help='Compress the video to save space')

# parse the arguments
args = parser.parse_args()
exp_dir = args.exp_dir
data_dir = args.data_dir
repo_dir = args.repo_dir
config_dir = args.config_dir
debug = True if args.debug else False
crop = False if args.nocrop else True
lossless = False if args.lossy else True
show_boundary = True if args.boundary else False
convert_to_avi = True if args.ctrax else False
compress = True if (args.compress and lossless) else False

# list the cameras
cameras = list_cameras()
if len(cameras) == 0:
    print("No cameras found. Exiting.")
    exit()
else:
    print(f"Found {len(cameras)} camera(s).")
    if len(cameras) > 1:
        print("Selecting the first camera. Change the index in main.py to select a different camera.")

# if data_dir is not a full path, make it a full path by joining it with repo_dir
if not os.path.isabs(data_dir):
    data_dir = os.path.join(repo_dir, data_dir)
assert os.path.isdir(data_dir), f"Invalid data directory: {data_dir}"

# if exp_dir is not a full path, make it a full path by joining it with repo_dir
if not os.path.isabs(exp_dir):
    exp_dir = os.path.join(repo_dir, exp_dir)
assert os.path.isdir(exp_dir), f"Invalid experiment directory: {exp_dir}"

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
with SpinnakerCamera(video_output_name=exp_name, video_output_path=exp_data_dir, record_video=True, FPS=rig_config['FPS'], lossless=lossless) as camera:
    camera.start()

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

    
    # press enter to start the experiment
    input("Press Enter to start the experiment.")

    # keep track of the time
    start_time = time.time()
    # Main loop
    running = True
    while running:

        # Calculate the time elapsed
        elapsed_time = time.time() - start_time
        if debug:
            print(f"Elapsed time: {elapsed_time:.2f} seconds", end='\r')
        
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
        image = camera.get_array(wait=True, crop_bounds=[x_min, x_max, y_min, y_max] if crop else None)

        # Clear the screen
        screen.fill(BACKGROUND_COLOR)
        
        # Show boundary (if needed)
        if show_boundary:
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

print("\nExperiment completed.")

# Quit Pygame
pygame.quit()

# Split the video into phases
if not 'SPLIT_TIMES' in locals():
    SPLIT_TIMES = []

if len(SPLIT_TIMES) > 0:
    print("Splitting the video into phases...")
    SPLIT_TIMES = [0] + SPLIT_TIMES + [elapsed_time]
    # add a few frames worth of time to the end of each phase
    SPLIT_TIMES = [time + 2/rig_config['FPS'] for time in SPLIT_TIMES]
    for i in range(len(SPLIT_TIMES)-1):
        print(f"\n\n\nSplitting phase {i+1}...\n\n\n")
        # ffmpeg command to split the video
        command = f"ffmpeg -i {os.path.join(exp_data_dir, exp_name)}.mp4 -ss {SPLIT_TIMES[i]} -to {SPLIT_TIMES[i+1]} -c:v libx264{' -crf 0' if lossless else ''} {os.path.join(exp_data_dir, exp_name)}_phase_{i+1}.mp4" 
        os.system(command)
    print("Video split into phases.")
    split = True
else:
    split = False

# convert the video to uncompressed AVI
if convert_to_avi:
    print("Converting the video to uncompressed AVI...")
    # create a subfolder for the avi files
    os.makedirs(os.path.join(exp_data_dir, 'avi'), exist_ok=True)
    if split:
        for i in range(len(SPLIT_TIMES)-1):
            command = f"mencoder {os.path.join(exp_data_dir, exp_name)}_phase_{i+1}.mp4 -o {os.path.join(exp_data_dir, 'avi', exp_name)}_phase_{i+1}.avi -vf format=rgb24 -ovc raw -nosound"
            os.system(command)
    else:
        command = "mencoder {} -o {}.avi -vf format=rgb24 -ovc raw -nosound".format(os.path.join(exp_data_dir, exp_name + '.mp4'), os.path.join(exp_data_dir, 'avi', exp_name + '_processed'))
        os.system(command)

# compress the video to save space
if compress:
    print("Compressing the video to save space...")
    # create a subfolder for the compressed files
    os.makedirs(os.path.join(exp_data_dir, 'compressed'), exist_ok=True)
    if split:
        for i in range(len(SPLIT_TIMES)-1):
            command = f"ffmpeg -i {os.path.join(exp_data_dir, exp_name)}_phase_{i+1}.mp4 -c:v libx264 {os.path.join(exp_data_dir, 'compressed', exp_name)}_phase_{i+1}.mp4"
            os.system(command)
    else:
        command = f"ffmpeg -i {os.path.join(exp_data_dir, exp_name)}.mp4 -c:v libx264 {os.path.join(exp_data_dir, 'compressed', exp_name)}.mp4"
        os.system(command)

# exit the program
exit(0)
