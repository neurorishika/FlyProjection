from flyprojection.utils import hex_to_rgb, relative_to_absolute, get_boolean_answer, rect_fit
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
parser = argparse.ArgumentParser(description='Open/Closed Loop Fly Projection System Rig Configuration')
parser.add_argument('--repo_dir', type=str, default='/home/smellovision/FlyProjection/', help='Path to the repository directory')


# parse the arguments
args = parser.parse_args()
repo_dir = args.repo_dir

# assert that there is a configs directory in the repository directory with a archived_configs subdirectory
assert os.path.isdir(os.path.join(repo_dir, 'configs')), f"Invalid configs directory: {os.path.join(repo_dir, 'configs')}"
assert os.path.isdir(os.path.join(repo_dir, 'configs', 'archived_configs')), f"Invalid configs directory: {os.path.join(repo_dir, 'configs', 'archived_configs')}"
# assert that there is a rig_config.json file in the configs directory
assert os.path.isfile(os.path.join(repo_dir, 'configs', 'rig_config.json')), f"rig_config.json file not found in {os.path.join(repo_dir, 'configs')}"

# load the rig configuration
with open(os.path.join(repo_dir, 'configs', 'rig_config.json'), 'r') as f:
    rig_config = json.load(f)

# ask if the width and height of the projector should be changed
if get_boolean_answer(f"Do you want to use the current projector width and height ({rig_config['projector_width']}x{rig_config['projector_height']})? [Y/n] ", default=True):
    projector_width = int(rig_config['projector_width'])
    projector_height = int(rig_config['projector_height'])
else:
    projector_width = int(input("Enter the projector width: "))
    projector_height = int(input("Enter the projector height: "))

rig_config['projector_width'] = projector_width
rig_config['projector_height'] = projector_height

# list the cameras
cameras = list_cameras()
if len(cameras) == 0:
    print("No cameras found. Exiting.")
    exit()
else:
    print(f"Found {len(cameras)} camera(s).")
    if len(cameras) > 1:
        # ask the user to select a camera
        if get_boolean_answer(f"Do you want to use the current camera index ({rig_config['camera_index']})? [Y/n] ", default=True):
            camera_index = int(rig_config['camera_index'])
        else:
            camera_index = int(input("Enter the index of the camera you would like to use (0-indexed and must be less than the number of cameras): "))
        assert camera_index < len(cameras), f"Invalid camera index: {camera_index}"
    else:
        camera_index = 0
rig_config['camera_index'] = camera_index

# setup FPS
if get_boolean_answer(f"Do you want to use the current FPS ({rig_config['FPS']})? [Y/n] ", default=True):
    FPS = int(rig_config['FPS'])
else:
    FPS = int(input("Enter the FPS: "))
rig_config['FPS'] = FPS

# Setup display and initialize Pygame
os.environ['SDL_VIDEO_WINDOW_POS'] = f"0,0"
pygame.init()
screen = pygame.display.set_mode((projector_width, projector_height), pygame.NOFRAME | pygame.HWSURFACE | pygame.DOUBLEBUF)

# ask the user if working area and projection margins should be reconfigured
if get_boolean_answer("Do you want to reconfigure the working area and projection margins? [Y/n] ", default=True):

    # ask user to place a white sheet of paper in front of the camera on the working area
    input("Place a white sheet of paper in front of the camera on the working area, turn off the backlight, and press Enter.")

    # take a picture
    with SpinnakerCamera(index=camera_index, FPS=FPS, record_video=False) as cam:
        cam.start()

        while True:

            # fill the screen with red
            screen.fill((255, 0, 0))
            pygame.display.flip()

            # wait for the camera to adjust
            time.sleep(2)

            image = cam.get_array(wait=True,dont_save=True)

            # ask user mark the four corners of the projected image using matplotlib ginput
            plt.imshow(image)
            plt.title("Mark the four corners of the projected image\nTop left -> Top right -> Bottom right -> Bottom left")
            plt.axis('off')
            projection_margins = np.array(plt.ginput(4, timeout=0))
            plt.close()
            
            # fit a rectangle to the marked points
            projection_margins, params_outer = rect_fit(projection_margins)

            # ask the user to turn on the backlight
            input("Turn on the backlight and press Enter.")

            # fill the screen with black
            screen.fill((0, 0, 0))
            pygame.display.flip()

            # wait for the camera to adjust
            time.sleep(2)

            # get an image from the camera
            image = cam.get_array(wait=True, dont_save=True)

            # ask user to mark the four corners of the working area using matplotlib ginput
            plt.imshow(image)
            plt.title("Mark the four corners of the working area\nTop left -> Top right -> Bottom right -> Bottom left")
            plt.axis('off')
            working_area = np.array(plt.ginput(4, timeout=0))
            plt.close()

            # fit a rectangle to the marked points forcing the rotation to be the same
            working_area, params_inner = rect_fit(working_area, force_rotation=params_outer[4])

            # show the marked rectangles
            plt.imshow(image)
            for i in [1,2,3,0]:
                plt.plot([projection_margins[i-1][0], projection_margins[i][0]], [projection_margins[i-1][1], projection_margins[i][1]], 'r')
                plt.plot([working_area[i-1][0], working_area[i][0]], [working_area[i-1][1], working_area[i][1]], 'g')
            plt.show()

            print("Projecting working area, press Enter to continue.")
        
            # convert to relative coordinates
            rel_width = params_inner[2]/params_outer[2]
            rel_height = params_inner[3]/params_outer[3]

            top_corner_x = ((params_inner[0]-params_inner[2]/2) - (params_outer[0]-params_outer[2]/2))/params_outer[2]
            top_corner_y = ((params_inner[1]-params_inner[3]/2) - (params_outer[1]-params_outer[3]/2))/params_outer[3]

            # save the projection margins and working area
            rig_config['projection_margins'] = projection_margins
            rig_config['working_area'] = working_area
            rig_config['projection_params'] = list(params_outer)
            rig_config['working_params'] = list(params_inner)
            rig_config['exp_region_x'] = np.clip([top_corner_x, top_corner_x + rel_width], 0, 1).tolist()
            rig_config['exp_region_y'] = np.clip([top_corner_y, top_corner_y + rel_height], 0, 1).tolist()

            print("Experiment region:")
            print(f"X: {rig_config['exp_region_x']}")
            print(f"Y: {rig_config['exp_region_y']}")

            # show the working area
            BOUNDARY_X = (int(rig_config['exp_region_x'][0]*projector_width), int(rig_config['exp_region_x'][1]*projector_width))
            BOUNDARY_Y = (int(rig_config['exp_region_y'][0]*projector_height), int(rig_config['exp_region_y'][1]*projector_height))

            screen.fill((0, 0, 0))
            pygame.draw.rect(screen, (255, 255, 255), (BOUNDARY_X[0], BOUNDARY_Y[0], BOUNDARY_X[1]-BOUNDARY_X[0], BOUNDARY_Y[1]-BOUNDARY_Y[0]), 0)
            pygame.display.flip()

            # ask the user if the working area is correct
            if get_boolean_answer("Are the working area correct? [Y/n] ", default=True):
                break
        
        # measure the physical dimensions of the working area
        input("Measure the physical dimensions of the working area and press Enter.")

        # enter the physical dimensions of the working area
        PHYSICAL_X = float(input("Enter the physical width of the working area in inches: "))
        PHYSICAL_Y = float(input("Enter the physical height of the working area in inches: "))
        rig_config['physical_x'] = PHYSICAL_X
        rig_config['physical_y'] = PHYSICAL_Y
    
        # stop the camera
        cam.stop()
else:
    print("Using the current working area and projection margins.")
    print(f"Experiment region:")
    print(f"X: {rig_config['exp_region_x']}")
    print(f"Y: {rig_config['exp_region_y']}")
    print(f"Physical dimensions:")
    print(f"X: {rig_config['physical_x']}")
    print(f"Y: {rig_config['physical_y']}")

timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
rig_config['timestamp'] = timestamp

# save the rig configuration
with open(os.path.join(repo_dir, 'configs', 'rig_config.json'), 'w') as f:
    json.dump(rig_config, f, indent=4)

# save a copy of the rig configuration with the current timestamp
with open(os.path.join(repo_dir, 'configs', 'archived_configs', f'rig_config_{timestamp}.json'), 'w') as f:
    json.dump(rig_config, f, indent=4)






    

    




