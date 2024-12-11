import numpy as np
import time
import pygame  # Import pygame for event handling
from flyprojection.controllers.basler_camera import BaslerCamera
from flyprojection.projection.artist import Artist, Drawing
from flyprojection.behavior.tracker import FastTracker
from flyprojection.utils.logging_archived import setup_async_logger, signal_handler
from flyprojection.utils.geometry import bounding_boxes_intersect, cartesian_to_polar, polar_to_cartesian
import json
import os
import shutil
import datetime
import signal
import argparse
from numba import njit
import random


@njit
def derivative(state, t, c_r, k_r, r0, w, c_theta):
    """
    Computes derivatives for the damped oscillator in polar coordinates.
    State order: (r, theta, r_dot, theta_dot)
    """
    r, theta, r_dot, theta_dot = state

    # Derivatives
    drdt = r_dot
    dr_dotdt = -c_r * r_dot - k_r * (r - r0)
    dthetadt = theta_dot
    dtheta_dotdt = -c_theta * (theta_dot - w)

    # Return in the order (dr/dt, dtheta/dt, dr_dot/dt, dtheta_dot/dt)
    return np.array([drdt, dthetadt, dr_dotdt, dtheta_dotdt])


def check_initial_conditions(initial_conditions_polar, r0):
    """
    Validates initial conditions to ensure motion is directed outwards or tangentially.
    initial_conditions_polar: (r, theta, r_dot, theta_dot)
    """
    r = initial_conditions_polar[0]
    r_dot = initial_conditions_polar[2]  # Note the change since r_dot is now at index 2
    return not ((r < r0 and r_dot < 0) or (r > r0 and r_dot > 0))


def integrate_with_stopping(f, initial_state, t_max, dt, c_r, k_r, r0, w, c_theta, epsilon, max_steps):
    """
    Integrates the system until early stopping criteria are met.
    initial_state: (r, theta, r_dot, theta_dot)
    """
    trajectory = np.zeros((max_steps, len(initial_state)))
    trajectory[0] = initial_state
    t = 0.0
    state = initial_state
    step = 0

    # Stop when |r - r0| < epsilon or we reach t_max or max_steps
    while t < t_max and abs(state[0] - r0) >= epsilon and step < max_steps - 1:
        dx = derivative(state, t, c_r, k_r, r0, w, c_theta)
        state += dx * dt
        t += dt
        step += 1
        trajectory[step] = state

    return np.arange(step + 1) * dt, trajectory[:step + 1]


# Path parameters
k_r = 1.0
c_r = 2 * np.sqrt(k_r)  # critical damping
w = 0.2  # angular velocity
c_theta = 1.0  # damping coefficient for angular velocity
epsilon = 0.01  # stopping threshold for r
t_max = 1000 # maximum time
dt = 0.05 # time step
max_steps = int(t_max / dt)

# Task parameters
r_ratio = 0.7  # ratio of radius to arena radius
thickness = 20  # thickness of the trajectory
min_v_threshold = 150 # velocity threshold for engagement
max_angular_velocity = np.pi/2 # maximum angular velocity
max_time_outside = 5 # maximum time outside the odor zone




if __name__ == "__main__":
    # set up the argument parser
    parser = argparse.ArgumentParser(description='Open/Closed Loop Fly Projection System Rig Configuration')
    parser.add_argument('--repo_dir', type=str, default='/mnt/sda1/Rishika/FlyProjection/', help='Path to the repository directory')
    parser.add_argument('--sphere_of_infuence', type=int, default=100, help='Dimension of the sphere of influence')
    parser.add_argument('--n_targets', type=int, default=5, help='Number of targets to track')
    parser.add_argument('--display', type=bool, default=False, help='Display the tracking visualization')

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

    center = rig_config['camera_space_arena_center']
    radius = rig_config['camera_space_arena_radius']*r_ratio

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
    camera.init()

    # Initialize the tracker with the camera and desired parameters
    tracker = FastTracker(
        camera=camera,
        n_targets=args.n_targets,
        debug=args.display,  # Set to True to see the tracking visualization
        smoothing_alpha=0.1
    )

    # Start the camera
    camera.start()

    # Initialize the Artist
    artist = Artist(camera, rig_config, method="map")

    # Setup the asynchronous logger
    logger = setup_async_logger(os.path.join(repo_dir, 'tests', 'closed_loop_test.log'))

    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    last_time = time.time()

    times = []

    max_time = 60*10 # Run the loop for 10 minutes

    engagement = np.array([False]*args.n_targets)
    time_since_last_encounter = np.array([0.]*args.n_targets)

    engaged_paths = {}

    try:
        start_time = time.time()

        while True:
            current_time = time.time()
            delta_time = current_time - last_time
            last_time = current_time
            times.append(delta_time)

            # setup the background
            background = Drawing()

            # draw a circle around the arena
            background.add_circle(center, radius, color=(1, 0, 0), fill=False, line_width=thickness)

            estimates = tracker.process_next_frame()

            if estimates is not None:
                # shuffle the estimates
                random.shuffle(estimates)

                for estimate in estimates:
                    # log the estimates
                    logger.info(f"ID: {estimate['id']}, Position: {estimate['position'][0]}, {estimate['position'][1]}, "
                          f"Velocity: {estimate['velocity_smooth']}, Angle: {np.rad2deg(estimate['angle_smooth'])}°, "
                          f"Time Since Start: {estimate['time_since_start']:.2f}s, Angular Velocity: {estimate['angular_velocity_smooth']}")
                    
                    # get color at the position
                    if not np.isnan(estimate['position']).any():
                        red = artist.get_values_at_camera_coords([estimate['position']])[0][0]
                        logger.info(f"Color at position: {artist.get_values_at_camera_coords([estimate['position']])}")
                        if red > 0:
                            # reset time since last encounter
                            time_since_last_encounter[estimate['id']] = 0
                        else:
                            # increment time since last encounter
                            time_since_last_encounter[estimate['id']] += delta_time

                    # # plot a line in the direction
                    # if not np.isnan(estimate['position']).any() and not np.isnan(estimate['direction_smooth']) and not np.isnan(estimate['velocity_smooth']):
                    #     x = [estimate['position'][0], estimate['position'][0] + estimate['velocity_smooth']*np.cos(estimate['direction_smooth'])]
                    #     y = [estimate['position'][1], estimate['position'][1] + estimate['velocity_smooth']*np.sin(estimate['direction_smooth'])]
                    #     background.add_path(list(zip(x, y)), color=(1, 0, 0), line_width=5)

                    if not np.isnan(estimate['position']).any() and not np.isnan(estimate['direction']) and engagement[estimate['id']] == False and not np.isnan(estimate['velocity']) and not np.isnan(estimate['angular_velocity_smooth']):
                        # reject if the velocity is less than the threshold
                        if estimate['velocity'] < min_v_threshold:
                            continue
                        # reject if the angular velocity is greater than the maximum angular velocity
                        if np.abs(estimate['angular_velocity_smooth']) > max_angular_velocity:
                            continue
                        
                        # distance from the center
                        r_dist = np.sqrt((estimate['position'][0] - center[0])**2 + (estimate['position'][1] - center[1])**2)
                        scale_factor = np.sin(estimate['direction'] - np.arctan2(estimate['position'][1] - center[1], estimate['position'][0] - center[0]))
                        magnitude = np.abs(radius-r_dist) #+ np.abs(scale_factor)**2*(radius-np.abs(radius-r_dist))
                        
                        # calculate the state
                        state = (estimate['position'][0], estimate['position'][1], magnitude*np.cos(estimate['direction']), magnitude*np.sin(estimate['direction']))

                        state_polar = cartesian_to_polar(state, center)

                        # Reject invalid initial conditions
                        if not check_initial_conditions(state_polar, radius):
                            continue

                        # Scale angular velocity to match the initial angular velocity direction
                        w_ = w * np.sign(state_polar[3])

                        # Integrate the system
                        _, trajectory = integrate_with_stopping(
                            f, state_polar, t_max, dt, c_r, k_r, radius, w_, c_theta, epsilon, max_steps
                        )

                        # Convert trajectory to Cartesian for plotting
                        cartesian_trajectory = np.array([polar_to_cartesian(state, center) for state in trajectory])

                        x = cartesian_trajectory[:, 0]
                        y = cartesian_trajectory[:, 1]

                        # plot as a path
                        if len(x) > 1 and len(y) > 1 and len(x) == len(y):

                            # make sure the new path does not intersect with any existing paths
                            failed = False
                            for path in engaged_paths.values():
                                if bounding_boxes_intersect(cartesian_trajectory, path):
                                    failed = True
                                    break
                            if failed:
                                continue

                            background.add_path(list(zip(x, y)), color=(1, 0, 0), line_width=thickness)
                            # # set engagement to True
                            engagement[estimate['id']] = True
                            # store the path
                            engaged_paths[estimate['id']] = cartesian_trajectory
                    
                    if engagement[estimate['id']] == True:
                        # reset the engagement if the fly is outside the zone for too long
                        if time_since_last_encounter[estimate['id']] > max_time_outside:
                            engagement[estimate['id']] = False
                            engaged_paths.pop(estimate['id'], None)
                            time_since_last_encounter[estimate['id']] = 0
                        else:
                            # get the path
                            path = engaged_paths[estimate['id']]
                            x = path[:, 0]
                            y = path[:, 1]
                            background.add_path(list(zip(x, y)), color=(1, 0, 0), line_width=thickness)
            
            logger.info(f"Engagement: {engagement}")


            # Get the drawing function
            draw_fn = background.get_drawing_function()

            # Draw the geometry using Artist
            artist.draw_geometry(draw_fn, debug=False)

            if current_time - start_time > max_time:
                break

    except Exception as e:
        print("Exiting due to exception: ", e)
        # print the stack trace
        import traceback
        traceback.print_exc()
    finally:
        tracker.release_resources()
        artist.close()
        camera.stop()
        print(f"Average frame rate: {1 / np.mean(times)} Hz")
