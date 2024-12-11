from flyprojection.utils import hex_to_rgb, relative_to_absolute, get_boolean_answer,rect_fit,get_rectangle_points
import os
import argparse
import datetime
import time
import json
import cv2
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
from skg import nsphere_fit

# set up the argument parser
parser = argparse.ArgumentParser(description='Open/Closed Loop Fly Projection System')
parser.add_argument('--data_dir', type=str, default='data/exp_name/timestamp', help='Path to the data directory')
parser.add_argument('--nocrop', action='store_true', help='Crop the image to the region of interest')
parser.add_argument('--ctrax', action='store_true', help='Prepare the data for Ctrax')
parser.add_argument('--compress', action='store_true', help='Compress the video file')
parser.add_argument('--lossy', action='store_true', help='Compress the video file using lossy compression')

# parse the arguments
args = parser.parse_args()
data_dir = args.data_dir
crop = False if args.nocrop else True
convert_to_avi = args.ctrax
compress = args.compress
lossless = False if args.lossy else True


# check if the data directory exists
assert os.path.isdir(data_dir), f"Invalid data directory: {data_dir}"

# in the data directory, there should be a config.py file and experiment_logic.py file
assert os.path.isfile(os.path.join(data_dir, 'config.py')), f"config.py file not found in {data_dir}"

# get experiment name and timestamp
timestamp = os.path.basename(os.path.dirname(data_dir))
exp_name = os.path.basename(os.path.dirname(os.path.dirname(data_dir)))
print(f"Experiment Name: {exp_name}")
print(f"Timestamp: {timestamp}")

#  make sure there is a video file in the data directory with the experiment name
assert os.path.isfile(os.path.join(data_dir, f"{exp_name}.mp4")), f"Video file not found in {data_dir}"

# make sure there is a rig_config.json file in the data directory
assert os.path.isfile(os.path.join(data_dir, 'rig_config.json')), f"rig_config.json file not found in {data_dir}"

# load the rig_config.json file
with open(os.path.join(data_dir, 'rig_config.json'), 'r') as f:
    rig_config = json.load(f)

# load the config.py file
exec(open(os.path.join(data_dir, 'config.py')).read())

# assert that required variables are defined
for var in ['ROI_TYPE']:
    assert var in locals(), f"Variable {var} not defined in config.py"

# get screen size (values will be loaded from config.py)
SCREEN_SIZE = (rig_config['projector_width'], rig_config['projector_height'])

# process the values from config.py
BOUNDARY_X = (int(rig_config['projector_width']*rig_config['exp_region_x'][0]), int(rig_config['projector_width']*rig_config['exp_region_x'][1]))
BOUNDARY_Y = (int(rig_config['projector_height']*rig_config['exp_region_y'][0]), int(rig_config['projector_height']*rig_config['exp_region_y'][1]))
SCALE_FACTOR = ((BOUNDARY_X[1]-BOUNDARY_X[0])/rig_config['physical_x'] + (BOUNDARY_Y[1]-BOUNDARY_Y[0])/rig_config['physical_y'])/2

assert ROI_TYPE in ['rectangle', 'circle'], "Invalid ROI_TYPE. Must be 'rectangle' or 'circle'."

# check if arena_alignment.png exists
if os.path.isfile(os.path.join(data_dir, 'arena_alignment.png')):
    # load the image
    image = cv2.imread(os.path.join(data_dir, 'arena_alignment.png'), cv2.IMREAD_GRAYSCALE)
else:
    # open the video file and get the average of 10 random frames
    cap = cv2.VideoCapture(os.path.join(data_dir, f"{exp_name}.mp4"))
    frames = []
    for i in tqdm(range(10)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, np.random.randint(0, cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        ret, frame = cap.read()
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()
    image = np.mean(frames, axis=0).astype(np.uint8)

# show the image
plt.imshow(image, cmap='gray')
plt.title("Arena Alignment Image")
# save the image
plt.savefig(os.path.join(data_dir, 'arena_alignment.png'))
plt.show()

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
            with open(os.path.join(data_dir, 'roi_processed.json'), 'w') as f:
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
            with open(os.path.join(data_dir, 'roi_processed.json'), 'w') as f:
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

# get the video file
video_file = os.path.join(data_dir, f"{exp_name}.mp4")

# use ffmpeg to crop the video
if crop:
    # press enter to crop the video
    input("Press Enter to crop the video...")

    print("Cropping the video...")
    command = f"ffmpeg -i {video_file} -vf crop={x_max-x_min}:{y_max-y_min}:{x_min}:{y_min} -c:v libx264{' -crf 0' if lossless else ''} {os.path.join(data_dir, exp_name)}_cropped.mp4" 
    os.system(command)
    print("Video cropped.")
    video_file = os.path.join(data_dir, f"{exp_name}_cropped.mp4")

# get duration of the video
cap = cv2.VideoCapture(video_file)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
elapsed_time = frame_count/fps
cap.release()

# Split the video into phases
if not 'SPLIT_TIMES' in locals():
    SPLIT_TIMES = []

if len(SPLIT_TIMES) > 0:
    print("Splitting the video into phases...")
    SPLIT_TIMES = [0] + SPLIT_TIMES + [elapsed_time]
    # add two frame worth of time to the end of each phase
    SPLIT_TIMES = [time + 2/rig_config['FPS'] for time in SPLIT_TIMES]
    for i in range(len(SPLIT_TIMES)-1):
        print(f"Splitting phase {i+1}...")
        # ffmpeg command to split the video
        command = f"ffmpeg -i {video_file} -ss {SPLIT_TIMES[i]} -to {SPLIT_TIMES[i+1]} -c:v libx264{' -crf 0' if lossless else ''} {os.path.join(data_dir, exp_name)}_phase_{i+1}.mp4"
        os.system(command)
    print("Video split into phases.")
    split = True
else:
    split = False

# convert the video to uncompressed AVI
if convert_to_avi:
    print("Converting the video to uncompressed AVI...")
    # create a subfolder for the avi files
    os.makedirs(os.path.join(data_dir, 'avi'), exist_ok=True)
    if split:
        for i in range(len(SPLIT_TIMES)-1):
            command = f"mencoder {os.path.join(data_dir, exp_name)}_phase_{i+1}.mp4 -o {os.path.join(data_dir, 'avi', exp_name)}_phase_{i+1}.avi -vf format=rgb24 -ovc raw -nosound"
            os.system(command)
    else:
        command = "mencoder " + video_file + " -o " + os.path.join(data_dir, 'avi', exp_name) + ".avi -vf format=rgb24 -ovc raw -nosound"
        os.system(command)

# compress the video
if compress:
    print("Compressing the video to save space...")
    # create a subfolder for the compressed files
    os.makedirs(os.path.join(data_dir, 'compressed'), exist_ok=True)
    if split:
        for i in range(len(SPLIT_TIMES)-1):
            command = f"ffmpeg -i {os.path.join(data_dir, exp_name)}_phase_{i+1}.mp4 -c:v libx264 {os.path.join(data_dir, 'compressed', exp_name)}_phase_{i+1}.mp4"
            os.system(command)
    else:
        command = f"ffmpeg -i {video_file} -c:v libx264 {os.path.join(data_dir, 'compressed', exp_name)}.mp4"
        os.system(command)

# exit the program
exit(0)
