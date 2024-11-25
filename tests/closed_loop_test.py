import numpy as np
import time
import pygame  # Import pygame for event handling
from flyprojection.controllers.basler_camera import BaslerCamera
from flyprojection.projection.artist import Artist, Drawing
from flyprojection.behavior.tracker import FastTracker
from flyprojection.utils import setup_async_logger, signal_handler
import json
import os
import shutil
import datetime
import signal
import argparse

# Define the pinwheel drawing function
def add_pinwheel(drawing, center, radius, num_spokes=20, color=(1, 0, 0), start_offset=0):
    for i in range(num_spokes):
        start_angle = (i * (2 * np.pi / num_spokes) + start_offset) % (2 * np.pi)
        end_angle = ((i + 1) * (2 * np.pi / num_spokes) + start_offset) % (2 * np.pi)
        drawing.add_arc(
            center,
            radius,
            start_angle,
            end_angle,
            color=color if i % 2 == 0 else (0, 0, 0),
            line_width=0,
            fill=True,
        )



if __name__ == "__main__":
    # set up the argument parser
    parser = argparse.ArgumentParser(description='Open/Closed Loop Fly Projection System Rig Configuration')
    parser.add_argument('--repo_dir', type=str, default='/mnt/sda1/Rishika/FlyProjection/', help='Path to the repository directory')
    parser.add_argument('--pinwheel_dimension', type=int, default=100, help='Dimension of the pinwheel')

    # parse the arguments
    args = parser.parse_args()
    repo_dir = args.repo_dir

    # assert that there is a configs directory in the repository directory with a archived_configs subdirectory
    assert os.path.isdir(os.path.join(repo_dir, 'configs')), f"Invalid configs directory: {os.path.join(repo_dir, 'configs')}"
    assert os.path.isdir(os.path.join(repo_dir, 'configs', 'archived_configs')), f"Invalid configs directory: {os.path.join(repo_dir, 'configs', 'archived_configs')}"
    # assert that there is a rig_config.json file in the configs directory
    assert os.path.isfile(os.path.join(repo_dir, 'configs', 'rig_config.json')), f"rig_config.json file not found in {os.path.join(repo_dir, 'configs')}"

    # backup the rig_config.json file
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.copy(os.path.join(repo_dir, 'configs', 'rig_config.json'), os.path.join(repo_dir, 'configs', 'archived_configs', f'rig_config_until_{current_time}.json'))

    # load the rig configuration
    with open(os.path.join(repo_dir, 'configs', 'rig_config.json'), 'r') as f:
        rig_config = json.load(f)


    # Initialize the camera
    camera = BaslerCamera(
        index=rig_config['camera_index'],
        EXPOSURE_TIME=rig_config['experiment_exposure_time'],
        GAIN=0.0,
        WIDTH=rig_config['camera_width'],
        HEIGHT=rig_config['camera_height'],
        OFFSETX=rig_config['camera_offset_x'],
        OFFSETY=rig_config['camera_offset_y'],
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
        debug=False  # Set to True to see the tracking visualization
    )

    # Start the camera
    camera.start()

    # Initialize the Artist
    artist = Artist(camera, rig_config, method="map")

    # Setup the asynchronous logger
    logger = setup_async_logger(os.path.join(repo_dir, 'tests', 'closed_loop_test.log'))

    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    pinwheel_dimension = min(artist.camera_width, artist.camera_height)//10
    pinwheel_blueprint = Drawing()
    add_pinwheel(pinwheel_blueprint, (pinwheel_dimension//2, pinwheel_dimension//2), pinwheel_dimension//2, num_spokes=20, color=(0, 0, 1), start_offset=0)

    # Get or create the sprite
    pinwheel_sprite = pinwheel_blueprint.create_sprite(
        pinwheel_dimension,
        pinwheel_dimension,
        pinwheel_blueprint.get_drawing_function(),
    )

    # Initialize variables for rotation
    angle = 0  # Starting angle in radians
    angular_speed = np.pi / 4  # Rotation speed in radians per second (e.g., 45 degrees per second)
    last_time = time.time()

    times = []

    max_time = 60*10 # Run the loop for 10 minutes


    try:
        while True:
            current_time = time.time()
            delta_time = current_time - last_time
            last_time = current_time
            times.append(delta_time)

            estimates = tracker.process_next_frame()

            # Update the rotation angle
            angle += angular_speed * delta_time  # Update angle based on time elapsed

            start_time = time.time()

            # Create a new Drawing instance
            drawing = Drawing()

            if estimates is not None:
                for estimate in estimates:
                    # log the estimates
                    logger.info(f"ID: {estimate['id']}, Position: {estimate['position']}, "
                          f"Velocity: {estimate['velocity']}, Angle: {np.rad2deg(estimate['angle'])}°, "
                          f"Time Since Start: {estimate['time_since_start']:.2f}s")

                    if not np.isnan(estimate['position']).any():
                        # draw a pinwheel sprite at the estimated position
                        drawing.add_sprite(
                            pinwheel_sprite,
                            center=np.int32(estimate['position']).tolist(),
                            rotation=angle,
                        )

            # # Add pinwheel sprite to the drawing
            # drawing.add_sprite(
            #     pinwheel_sprite,
            #     center=(artist.camera_width / 2, artist.camera_height / 2),
            #     rotation=angle,
            # )

            # Get the drawing function
            draw_fn = drawing.get_drawing_function()

            # Draw the geometry using Artist
            artist.draw_geometry(draw_fn, debug=False)

            if current_time - start_time > max_time:
                break

    except Exception as e:
        print("Exiting due to exception: ", e)
    finally:
        tracker.release_resources()
        artist.close()
        camera.stop()
        print(f"Average frame rate: {1 / np.mean(times)} Hz")
