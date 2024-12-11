import numpy as np
import time
import pygame
from flyprojection.controllers.basler_camera import BaslerCamera
from flyprojection.projection.artist import Artist, Drawing
from flyprojection.behavior.tracker import FastTracker
from flyprojection.controllers.led_server import LEDPanelController
from flyprojection.utils.logging import AsyncLogger, AsyncHDF5Saver
from flyprojection.utils.geometry import bounding_boxes_intersect, cartesian_to_polar, polar_to_cartesian
from flyprojection.utils.input import get_directory_path_qt, get_string_answer_qt
from flyprojection.utils.utils import signal_handler
import json
import os
import shutil
import datetime
import signal
import argparse
from numba import njit
import random
import traceback
import sys

@njit
def dynamics(state, t, c_r, k_r, r0, w, c_theta):
    r, theta, r_dot, theta_dot = state
    drdt = r_dot
    dr_dotdt = -c_r * r_dot - k_r * (r - r0)
    dthetadt = theta_dot
    dtheta_dotdt = -c_theta * (theta_dot - w)
    return np.array([drdt, dthetadt, dr_dotdt, dtheta_dotdt])

def check_initial_conditions(initial_conditions_polar, r0):
    r = initial_conditions_polar[0]
    r_dot = initial_conditions_polar[2]  
    return not ((r < r0 and r_dot < 0) or (r > r0 and r_dot > 0))

def integrate_with_stopping(f, initial_state, t_max, dt, c_r, k_r, r0, w, c_theta, epsilon, max_steps):
    trajectory = np.zeros((max_steps, len(initial_state)))
    trajectory[0] = initial_state
    t = 0.0
    state = initial_state
    step = 0

    while t < t_max and abs(state[0] - r0) >= epsilon and step < max_steps - 1:
        dx = dynamics(state, t, c_r, k_r, r0, w, c_theta)
        state += dx * dt
        t += dt
        step += 1
        trajectory[step] = state

    return np.arange(step + 1) * dt, trajectory[:step + 1]

def attempt_engagement(estimate, center, radius, engagement, engaged_paths, c_r, k_r, w, c_theta, epsilon, t_max, dt, max_steps, min_v_threshold, max_angular_velocity, thickness):
    # Check if conditions for a new engagement are met
    if engagement[estimate['id']] or np.isnan(estimate['position']).any() or np.isnan(estimate['direction']) or np.isnan(estimate['velocity']) or np.isnan(estimate['angular_velocity_smooth']):
        return None

    if estimate['velocity'] < min_v_threshold:
        return None

    if abs(estimate['angular_velocity_smooth']) > max_angular_velocity:
        return None

    # Distance from the center
    r_dist = np.sqrt((estimate['position'][0] - center[0])**2 + (estimate['position'][1] - center[1])**2)
    scale_factor = np.sin(estimate['direction'] - np.arctan2(estimate['position'][1] - center[1], estimate['position'][0] - center[0]))
    magnitude = abs(radius - r_dist)

    # Construct the initial state
    state = (
        estimate['position'][0],
        estimate['position'][1],
        magnitude * np.cos(estimate['direction']),
        magnitude * np.sin(estimate['direction'])
    )
    state_polar = cartesian_to_polar(state, center)

    if not check_initial_conditions(state_polar, radius):
        return None

    w_ = w * np.sign(state_polar[3])

    # Integrate
    _, trajectory = integrate_with_stopping(
        dynamics, state_polar, t_max, dt, c_r, k_r, radius, w_, c_theta, epsilon, max_steps
    )

    # Convert trajectory to Cartesian
    cartesian_trajectory = np.array([polar_to_cartesian(st, center) for st in trajectory])

    # Check for intersection with existing paths
    for path in engaged_paths.values():
        if bounding_boxes_intersect(cartesian_trajectory, path):
            return None

    # If we reach here, we succeeded in generating a valid path
    return cartesian_trajectory

if __name__ == "__main__":

    max_time = 60  # 1 minute

    # Path parameters
    k_r = 1.0
    c_r = 2 * np.sqrt(k_r)  # critical damping
    w = 0.2
    c_theta = 1.0
    epsilon = 0.01
    t_max = 1000
    dt = 0.05
    max_steps = int(t_max / dt)
    r_ratio = 0.7
    thickness = 20
    min_v_threshold = 150
    max_angular_velocity = np.pi/4
    max_time_outside = 5

    # parse arguments
    parser = argparse.ArgumentParser(description='Open/Closed Loop Fly Projection System Rig Configuration')
    parser.add_argument('--repo_dir', type=str, default='/mnt/sda1/Rishika/FlyProjection/', help='Path to the repository directory')
    parser.add_argument('--sphere_of_infuence', type=int, default=100, help='Dimension of the sphere of influence')
    parser.add_argument('--n_targets', type=int, default=5, help='Number of targets to track')
    parser.add_argument('--display', type=bool, default=False, help='Display the tracking visualization')
    args = parser.parse_args()

    repo_dir = args.repo_dir

    assert os.path.isdir(os.path.join(repo_dir, 'configs')), f"Invalid configs directory: {os.path.join(repo_dir, 'configs')}"
    assert os.path.isdir(os.path.join(repo_dir, 'configs', 'archived_configs')), f"Invalid configs directory: {os.path.join(repo_dir, 'configs', 'archived_configs')}"
    assert os.path.isfile(os.path.join(repo_dir, 'configs', 'rig_config.json')), f"rig_config.json file not found in {os.path.join(repo_dir, 'configs')}"

    with open(os.path.join(repo_dir, 'configs', 'rig_config.json'), 'r') as f:
        rig_config = json.load(f)

    center = rig_config['camera_space_arena_center']
    radius = rig_config['camera_space_arena_radius'] * r_ratio

    # Setup configuration
    global_metadata = {"experiment_id": "cleaned_trajectory_test"}
    metadata_config = {
        "dtype": np.dtype([("timestamp", np.float64)])
    }
    data_dtype = np.dtype([
        ('id', np.int32),
        ('position_x', np.float64),
        ('position_y', np.float64),
        ('velocity', np.float64),
        ('velocity_smooth', np.float64),
        ('angle', np.float64),
        ('angle_smooth', np.float64),
        ('direction', np.float64),
        ('direction_smooth', np.float64),
        ('angular_velocity_smooth', np.float64),
        ('time_since_start', np.float64),
        ('engagement', np.bool_),
        ('red_at_position', np.float64),
        ('blue_at_position', np.float64),
        ('green_at_position', np.float64),
    ])
    datasets_config = {
        "camera_feed": {
            "shape": (rig_config['camera_height'], rig_config['camera_width']),
            "dtype": np.uint8,
            "compression": "gzip",
            "compression_opts": 9,
            "difference_coding": False
        },
        "stimulus_feed": {
            "shape": (rig_config['camera_height'], rig_config['camera_width'], 3),
            "dtype": np.uint8,
            "compression": "gzip",
            "compression_opts": 9,
            "difference_coding": False
        },
        "data": {
        "shape": (args.n_targets,),  # One row per frame, with one entry per target
        "dtype": data_dtype,
        # For estimate data, difference coding and background don't make sense, so omit them:
        "difference_coding": False
        }
    }

    # get save directory from user
    save_dir = get_directory_path_qt("Select save directory", default=os.path.join(repo_dir, 'data'))

    # get experiment name from user
    experiment_name = get_string_answer_qt("Enter experiment name: ")

    # create experiment directory by combining save directory and experiment name
    experiment_dir = os.path.join(save_dir, experiment_name + datetime.datetime.now().strftime("_%Y-%m-%d_%H-%M"))

    # create experiment directory if it does not exist
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
        

    engagement = np.array([False]*args.n_targets)
    time_since_last_encounter = np.zeros(args.n_targets)
    engaged_paths = {}
    times = []

    # Use camera as a context manager
    with BaslerCamera(
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
        ) as camera, \
        LEDPanelController(
            host='flyprojection-server', 
            port=65432
        ) as compass, \
        AsyncLogger(
            log_file=os.path.join(experiment_dir, "flyprojection.log"),
            logger_name="flyprojection",
        ) as logger, \
        AsyncHDF5Saver(
            h5_filename=os.path.join(experiment_dir, "data.h5"),
            datasets_config=datasets_config,
            metadata_config=metadata_config,
            global_metadata=global_metadata
        ) as saver:

        ## SETUP LED PANEL ##
        arena_surround = compass.create_stripes(
        stripe_width=2,
        color1=(0, 0, 50),
        color2=(0, 0, 0),
        orientation="vertical",
        )
        compass.send_image(arena_surround)

        with FastTracker(
            camera=camera,
            n_targets=args.n_targets,
            debug=args.display,
            smoothing_alpha=0.1
        ) as tracker, \
        Artist(
            camera=camera,
            rig_config=rig_config,
            method="map"
        ) as artist:

            signal.signal(signal.SIGINT, signal_handler)
            
            ## START CAMERA ##
            camera.start()

            started = False

            start_time = time.time()
            last_time = start_time

            try:
                while True:
                    current_time = time.time()
                    if started:
                        if last_time is not None:
                            delta_time = current_time - last_time
                            times.append(delta_time)
                        else:
                            delta_time = 0
                        last_time = current_time
                    else:
                        delta_time = 0

                    estimates, camera_image = tracker.process_next_frame(return_camera_image=True)
                    background = Drawing()

                    # Draw a circle in the center
                    background.add_circle(center, radius, color=(1, 0, 0), fill=False, line_width=thickness)

                    
                    if estimates is not None:
                        # Create data
                        data = np.zeros(args.n_targets, dtype=data_dtype)

                        # Shuffle estimates to avoid engaging the same fly every time
                        random.shuffle(estimates)

                        for estimate in estimates:

                            # get color at position
                            if not np.isnan(estimate['position']).any():
                                color_at_position = artist.get_values_at_camera_coords([estimate['position']])[0]
                            else:
                                color_at_position = [0.0, 0.0, 0.0]
                            
                            if color_at_position[0] > 0.0:
                                time_since_last_encounter[estimate['id']] = 0
                            else:
                                time_since_last_encounter[estimate['id']] += delta_time

                            logger.info(
                                f"ID: {estimate['id']}, Position: {estimate['position'][0]:.0f}, {estimate['position'][1]:.0f}, "
                                f"Velocity: {estimate['velocity_smooth']:.2f}, Angle: {np.rad2deg(estimate['angle_smooth']):.2f}°, "
                                f"Time Since Start: {estimate['time_since_start']:.2f}s, Angular Velocity: {estimate['angular_velocity_smooth']:.2f}"
                            )

                            if not engagement[estimate['id']]:
                                path = attempt_engagement(
                                    estimate, center, radius, engagement, engaged_paths, c_r, k_r, w, c_theta, epsilon,
                                    t_max, dt, max_steps, min_v_threshold, max_angular_velocity, thickness
                                )
                                if path is not None:
                                    # Add path and mark engagement
                                    background.add_path(list(zip(path[:, 0], path[:, 1])), color=(1, 0, 0), line_width=thickness)
                                    engagement[estimate['id']] = True
                                    engaged_paths[estimate['id']] = path

                            else:
                                # Already engaged, check if the fly is outside for too long
                                if time_since_last_encounter[estimate['id']] > max_time_outside:
                                    engagement[estimate['id']] = False
                                    engaged_paths.pop(estimate['id'], None)
                                    time_since_last_encounter[estimate['id']] = 0
                                else:
                                    # Redraw existing path
                                    path = engaged_paths[estimate['id']]
                                    background.add_path(list(zip(path[:, 0], path[:, 1])), color=(1, 0, 0), line_width=thickness)
                            
                            # Update frame data
                            data[estimate['id']] = (
                                estimate['id'],
                                estimate['position'][0],
                                estimate['position'][1],
                                estimate['velocity'],
                                estimate['velocity_smooth'],
                                estimate['angle'],
                                estimate['angle_smooth'],
                                estimate['direction'],
                                estimate['direction_smooth'],
                                estimate['angular_velocity_smooth'],
                                estimate['time_since_start'],
                                engagement[estimate['id']],
                                color_at_position[0],
                                color_at_position[1],
                                color_at_position[2]
                            )

                    logger.info(f"Engagement: {engagement}")

                    draw_fn = background.get_drawing_function()
                    stimulus_image = artist.draw_geometry(draw_fn, debug=False, return_camera_image=True)

                    if camera_image is not None and stimulus_image is not None and estimates is not None:

                        # start time if not started
                        if not started:
                            start_time = time.time()
                            last_time = None
                            started = True

                        metadata = {
                            "timestamp": current_time,
                            "frame_number": tracker.frame_count
                        }
                        # Save data
                        saver.save_frame_async(
                            metadata=metadata,
                            camera_feed=camera_image,
                            stimulus_feed=stimulus_image,
                            data=data
                        )

                    if started and current_time - start_time > max_time:
                        break

            except Exception as e:
                print("Exiting due to exception:", e)
                
                traceback.print_exc()
            finally:
                camera.stop()
                print(f"Average frame rate: {1 / np.mean(times)} Hz")
