import numpy as np
import time
import matplotlib.pyplot as plt
from flyprojection.controllers.basler_camera import BaslerCamera
from flyprojection.projection.artist import Artist, Drawing
from flyprojection.behavior.tracker import FastTracker
from flyprojection.controllers.led_server import WS2812B_LEDController, KasaPowerController
from flyprojection.utils.logging import AsyncLogger, AsyncHDF5Saver, MultiStreamManager
from flyprojection.utils.geometry import cartesian_to_polar
from flyprojection.utils.input import get_directory_path_qt, get_string_answer_qt, get_boolean_answer_qt
import json
import os
import datetime
import signal
import argparse
import random
import traceback
import sys
import threading

# Setup signal handler for graceful shutdown
stop_event = threading.Event()
def signal_handler(signum, frame):
    print("\nKeyboard interrupt received. Stopping gracefully...")
    stop_event.set()

# --- NEW: Engagement function using a chord ---
def attempt_engagement_chord(estimate, center, radius, path_steps, max_engagement_angle, min_v_threshold, max_angular_velocity):
    """
    If the animal is pointed toward the center (within max_engagement_angle),
    and meets velocity and angular velocity criteria, compute and return a chord
    across the circle defined by the given center and radius.
    """
    # Validate basic conditions
    if estimate['velocity'] < min_v_threshold or abs(estimate['angular_velocity_smooth']) > max_angular_velocity:
        return None

    # Compute desired heading: from animal's position to the arena center
    pos = np.array(estimate['position'])
    center_arr = np.array(center)
    vector_to_center = center_arr - pos
    desired_angle = np.arctan2(vector_to_center[1], vector_to_center[0])
    # Compute angular difference (wrap-around correctly)
    angle_diff = abs(np.arctan2(np.sin(estimate['direction'] - desired_angle),
                                np.cos(estimate['direction'] - desired_angle)))
    if angle_diff > max_engagement_angle:
        return None

    # The engagement chord is defined as the straight line (across the circle)
    # with the same orientation as the animal's heading.
    v = np.array([np.cos(estimate['direction']), np.sin(estimate['direction'])])
    endpoint1 = np.array(center) + radius * v
    endpoint2 = np.array(center) - radius * v
    chord = np.linspace(endpoint1, endpoint2, path_steps)
    return chord

if __name__ == "__main__":
    # Define and parse command line arguments
    parser = argparse.ArgumentParser(description='Fly Behavior Recording with Chord Engagement')
    parser.add_argument('--repo_dir', type=str, default='/mnt/sda1/Rishika/FlyProjection/', help='Path to the repository directory')
    parser.add_argument('--n_targets', type=int, default=5, help='Number of targets to track')
    parser.add_argument('--display', type=bool, default=False, help='Display the tracking visualization')
    parser.add_argument('--duration', type=float, default=1, help='Total duration of the experiment in minutes')
    args = parser.parse_args()

    # Convert duration from minutes to seconds
    total_duration = args.duration * 60

    ### DEFINE FIXED EXPERIMENT PARAMETERS ###
    # Parameters for the (imaginary) engagement circle.
    actual_radius = 50      # mm; physical radius of the (imaginary) engagement circle
    actual_thickness = 5    # mm; physical thickness (used in drawing if needed)

    # Engagement thresholds
    max_engagement_angle = np.deg2rad(15)  # threshold in radians (e.g., 15°)
    min_v_threshold = 150                # minimum velocity threshold
    max_angular_velocity = np.pi/4         # maximum angular velocity allowed

    # Colors for drawing (for LED feedback and stimulus window)
    background_color = (0, 0, 1)  # blue background
    # We no longer draw the curved circle stimulus; engagement is now based on chord generation

    # LED panel parameters (for visual feedback on the rig)
    compass_center = (32, 3)
    compass_radius = 30
    compass_ring_width = 2
    compass_color1 = (0, 0, 0)
    compass_color2 = (0, 0, 0)

    ### END FIXED EXPERIMENT PARAMETERS ###

    # Set up experiment directory and load rig configuration
    repo_dir = args.repo_dir
    assert os.path.isdir(os.path.join(repo_dir, 'configs')), f"Invalid configs directory: {os.path.join(repo_dir, 'configs')}"
    assert os.path.isdir(os.path.join(repo_dir, 'configs', 'archived_configs')), f"Invalid configs directory: {os.path.join(repo_dir, 'configs', 'archived_configs')}"
    rig_config_path = os.path.join(repo_dir, 'configs', 'rig_config.json')
    assert os.path.isfile(rig_config_path), f"rig_config.json file not found in {os.path.join(repo_dir, 'configs')}"
    with open(rig_config_path, 'r') as f:
        rig_config = json.load(f)

    # Process rig parameters (using physical_to_camera_scaling)
    center = rig_config['camera_space_arena_center']
    arena_radius = rig_config['camera_space_arena_radius']
    radius = rig_config['physical_to_camera_scaling'] * actual_radius

    ### SETUP EXPERIMENT DIRECTORY ###
    save_dir = get_directory_path_qt("Select save directory", default=os.path.join(repo_dir, 'data'))
    experiment_name = get_string_answer_qt("Enter experiment name: ")
    experiment_dir = os.path.join(save_dir, experiment_name + datetime.datetime.now().strftime("_%Y-%m-%d_%H-%M"))
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    # Save a copy of this script to the experiment directory
    os.system(f'cp {os.path.abspath(__file__)} {experiment_dir}')

    ### SETUP EXPERIMENT DATA CONFIGURATION ###
    metadata_config = {
        "dtype": np.dtype([("timestamp", np.float64), ("frame_number", np.int32)]),
    }
    # Include a 'phase' field (if you wish to add phase later) and the engagement_path.
    # engagement_path is an array of shape (path_steps, 2) of type uint16.
    path_steps = 30  # number of points in the chord
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
        ('intask', np.bool_),
        ('distance_from_center', np.float64),
        ('red_at_position', np.float64),
        ('blue_at_position', np.float64),
        ('green_at_position', np.float64),
        ('phase', 'S10'),
        ('engagement_path', np.uint16, (path_steps, 2))
    ])
    datasets_config = {
        "data": {
            "shape": (args.n_targets,),  # One row per target per frame
            "dtype": data_dtype,
        }
    }
    global_metadata = {
        "experiment": os.path.basename(__file__).replace(".py", ""),
        "experiment_name": experiment_name,
        "experiment_dir": experiment_dir,
    }

    manager = MultiStreamManager(stop_event=stop_event)
    manager.add_stream(
        output_file=os.path.join(experiment_dir, "camera.mp4"),
        width=2048,
        height=2048,
        fps=20,
        codec='h264_nvenc',
        pix_fmt_in='gray',
        preset='p1',
        log_file=os.path.join(experiment_dir, "camera.log"),
    )
    manager.add_stream(
        output_file=os.path.join(experiment_dir, "stimulus.mp4"),
        width=2048,
        height=2048,
        fps=20,
        codec='h264_nvenc',
        pix_fmt_in='rgb24',
        preset='p1',
        log_file=os.path.join(experiment_dir, "stimulus.log"),
    )
    ### END SETUP EXPERIMENT DATA CONFIGURATION ###

    # Ask user if they want to start the experiment
    start_experiment = get_boolean_answer_qt("Start experiment?")
    if not start_experiment:
        sys.exit()

    # Initialize tracking arrays for each target
    in_task = np.array([False] * args.n_targets)
    # 'engagement' flags indicate if a chord has been triggered
    engagement = np.array([False] * args.n_targets)
    engaged_paths = {}
    time_since_last_encounter = np.zeros(args.n_targets)
    times = []

    with BaslerCamera(
            index=rig_config['camera_index'],
            EXPOSURE_TIME=rig_config['experiment_exposure_time'],
            GAIN=rig_config['camera_gain'],
            WIDTH=rig_config['camera_width'],
            HEIGHT=rig_config['camera_height'],
            OFFSETX=rig_config['camera_offset_x'],
            OFFSETY=rig_config['camera_offset_y'],
            TRIGGER_MODE="Continuous",
            CAMERA_FORMAT="Mono8",
            record_video=False,
            debug=False,
        ) as camera, \
        WS2812B_LEDController(
            host=rig_config['visual_led_panel_hostname'],
            port=rig_config['visual_led_panel_port'],
        ) as compass, \
        KasaPowerController(
            ip=rig_config['IR_LED_IP'],
            default_state="on"
        ) as ir_led_controller, \
        AsyncLogger(
            log_file=os.path.join(experiment_dir, "terminal.log"),
            logger_name=experiment_name,
            level="INFO"
        ) as logger, \
        AsyncHDF5Saver(
            h5_filename=os.path.join(experiment_dir, "data.h5"),
            datasets_config=datasets_config,
            metadata_config=metadata_config,
            global_metadata=global_metadata
        ) as saver, \
        manager as manager, \
        FastTracker(
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

        # Setup LED panel (send a radial pattern as in the original rig)
        arena_surround = compass.create_radial(
            center=compass_center,
            ring_width=compass_ring_width,
            color1=compass_color1,
            color2=compass_color2,
        )
        compass.send_image(arena_surround)

        signal.signal(signal.SIGINT, signal_handler)
        camera.start()
        frame_no = 0
        start_time = time.time()
        last_time = start_time

        try:
            while True:
                current_time = time.time()
                elapsed_time = current_time - start_time
                if elapsed_time > total_duration:
                    break

                delta_time = current_time - last_time if last_time is not None else 0
                last_time = current_time
                if delta_time <= 0:
                    delta_time = 1e-6

                estimates, camera_image = tracker.process_next_frame(return_camera_image=True)

                # Prepare the drawing: we always draw the arena background.
                # In this version, we do not draw a stimulus circle.
                background = Drawing()
                background.add_circle(center, arena_radius, color=background_color, fill=True)

                if estimates is not None:
                    data = np.zeros(args.n_targets, dtype=data_dtype)
                    for estimate in estimates:
                        # Get color at the animal's current position (if available)
                        if not np.isnan(estimate['position']).any():
                            color_at_position = artist.get_values_at_camera_coords([estimate['position']])[0]
                            # Also compute the distance from the arena center (for logging)
                            pos = np.array(estimate['position'])
                            dist_from_center = np.linalg.norm(pos - np.array(center))
                        else:
                            color_at_position = [0.0, 0.0, 0.0]
                            dist_from_center = np.inf

                        # Compute vector from animal to center and determine desired angle.
                        vector_to_center = np.array(center) - np.array(estimate['position'])
                        desired_angle = np.arctan2(vector_to_center[1], vector_to_center[0])
                        angle_diff = abs(np.arctan2(np.sin(estimate['direction'] - desired_angle),
                                                    np.cos(estimate['direction'] - desired_angle)))
                        # Mark the target as 'in task' if its heading is nearly toward the center.
                        if angle_diff < max_engagement_angle:
                            in_task[estimate['id']] = True
                        else:
                            in_task[estimate['id']] = False

                        logger.info(
                            f"Frame: {frame_no}, ID: {estimate['id']}, Pos: {estimate['position'][0]:.0f},{estimate['position'][1]:.0f}, "
                            f"Angle diff: {np.rad2deg(angle_diff):.2f}°, In Task: {in_task[estimate['id']]}"
                        )

                        # Trigger engagement (chord) if not already engaged and conditions are met.
                        if in_task[estimate['id']] and not engagement[estimate['id']]:
                            chord = attempt_engagement_chord(
                                estimate, center, radius, path_steps,
                                max_engagement_angle, min_v_threshold, max_angular_velocity
                            )
                            if chord is not None:
                                engagement[estimate['id']] = True
                                engaged_paths[estimate['id']] = chord
                                # Reset the timer for disengagement.
                                time_since_last_encounter[estimate['id']] = 0
                                logger.info(f"Engagement triggered for ID {estimate['id']}")
                        else:
                            # If already engaged, update the time since last encounter.
                            time_since_last_encounter[estimate['id']] += delta_time
                            # If too much time passes, reset engagement.
                            if time_since_last_encounter[estimate['id']] > 3:  # e.g., 3 seconds timeout
                                engagement[estimate['id']] = False
                                engaged_paths.pop(estimate['id'], None)
                                time_since_last_encounter[estimate['id']] = 0

                        # Draw the engagement chord if one exists for this target.
                        if engagement[estimate['id']]:
                            chord = engaged_paths[estimate['id']]
                            background.add_path(list(zip(chord[:, 0], chord[:, 1])), color=(1, 0, 0), line_width=100)

                        # Prepare engagement_path data for saving.
                        if engagement[estimate['id']]:
                            # Convert to uint16 as required (rounding here)
                            eng_path = np.array(engaged_paths[estimate['id']]).round().astype(np.uint16)
                        else:
                            eng_path = np.ones((path_steps, 2), dtype=np.uint16) * np.nan

                        # Save frame data for this target.
                        phase_str = b"engage"  # you might change this to reflect different phases
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
                            in_task[estimate['id']],
                            dist_from_center,
                            color_at_position[0],
                            color_at_position[1],
                            color_at_position[2],
                            phase_str,
                            eng_path
                        )

                draw_fn = background.get_drawing_function()
                stimulus_image = artist.draw_geometry(draw_fn, debug=False, return_camera_image=True)

                if camera_image is not None and stimulus_image is not None and estimates is not None:
                    metadata = {
                        "timestamp": current_time,
                        "frame_number": frame_no
                    }
                    saver.save_frame_async(metadata=metadata, data=data)
                    frame_no += 1
                    manager.write_frame_to_all([camera_image, stimulus_image])
            
                if stop_event.is_set():
                    print("Exiting main loop due to KeyboardInterrupt...")
                    # tell the user how many frames were collected
                    print(f"Collected {frame_no} frames. Make sure to verify that the data was saved correctly in the log files.")
                    break

        except Exception as e:
            print("Exiting due to exception:", e)
            traceback.print_exc()
        finally:
            camera.stop()
            if len(times) > 0:
                fps = 1 / np.mean(times)
                print(f"Average frame rate: {1 / np.mean(times[-int(30 * fps):]):.2f} Hz")
            else:
                print("No frames recorded.")
            with open(os.path.join(experiment_dir, 'times.txt'), 'w') as f:
                f.write('\n'.join(map(str, times)))
            fig, ax = plt.subplots(figsize=(30, 3))
            ax.plot(np.nan_to_num(1 / np.array(times)), color='black')
            plt.box(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            plt.savefig(os.path.join(experiment_dir, 'times.png'))
            plt.close(fig)

    # Overlay the videos using ffmpeg
    os.system(
        f'cd {experiment_dir} && '
        f'ffmpeg -i camera.mp4 -i stimulus.mp4 -filter_complex "[0:v]format=rgb24,scale=w=640:h=480[camera];'
        f'[1:v]scale=w=640:h=480[stimulus];[camera][stimulus]blend=all_mode=\'overlay\':all_opacity=0.5[out]" '
        f'-map "[out]" -preset fast overlay_output.mp4 && '
        f'ffmpeg -itsscale 0.05 -i overlay_output.mp4 -c copy speedup.mp4'
    )
    print("Experiment complete. Exiting...")
    sys.exit()