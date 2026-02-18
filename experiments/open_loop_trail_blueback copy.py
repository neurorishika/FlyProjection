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

if __name__ == "__main__":
    # Define and parse command line arguments
    parser = argparse.ArgumentParser(description='Simple Fly Behavior Recording System with LED/IR Controls')
    parser.add_argument('--repo_dir', type=str, default='/mnt/sda1/Rishika/FlyProjection/', help='Path to the repository directory')
    parser.add_argument('--n_targets', type=int, default=5, help='Number of targets to track')
    parser.add_argument('--display', type=bool, default=False, help='Display the tracking visualization')
    parser.add_argument('--duration_pre', type=float, default=15, help='Duration of pre-stimulus phase in minutes')
    parser.add_argument('--duration_stim', type=float, default=30, help='Duration of stimulus phase in minutes')
    parser.add_argument('--duration_post', type=float, default=15, help='Duration of post-stimulus phase in minutes')
    args = parser.parse_args()

    # Convert durations (minutes to seconds)
    duration_pre = args.duration_pre * 60
    duration_stim = args.duration_stim * 60
    duration_post = args.duration_post * 60
    total_duration = duration_pre + duration_stim + duration_post

    ### DEFINE FIXED EXPERIMENT PARAMETERS ###
    # Parameters for the stimulus circle (central circle)
    actual_radius = 42.5      # mm; physical radius of the stimulus circle
    actual_thickness = 5    # mm; physical thickness of the stimulus circle

    # Disengagement threshold (in seconds)
    max_time_outside_intask = 10  # maximum allowed time outside the stimulus zone

    # Colors for drawing
    background_color = (0, 0, 1)  # blue background
    trail_color = (1, 0, 1)       # color for the central stimulus circle

    # LED panel parameters (for visual feedback on the rig)
    compass_center = (32, 3)
    compass_radius = 30
    compass_ring_width = 2
    compass_color1 = (0, 0, 0)
    compass_color2 = (0, 0, 0)

    ### END FIXED EXPERIMENT PARAMETERS ###

    # Set up the experiment directory and load rig configuration
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
    task_boundary = rig_config['physical_to_camera_scaling'] * 5  # e.g. 5 mm boundary
    thickness = rig_config['physical_to_camera_scaling'] * actual_thickness
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
    # Added a 'phase' field to store the experimental phase for each frame.
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
        ('distance_from_stimulus', np.float64),
        ('red_at_position', np.float64),
        ('blue_at_position', np.float64),
        ('green_at_position', np.float64),
        ('phase', 'S10')
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
            level="WARNING"
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

        # Flag to indicate when the experiment clock starts (i.e. after valid estimates are detected)
        experiment_started = False

        # Placeholders for timing
        start_time = None
        last_time = None

        try:
            while True:
                current_time = time.time()
                estimates, camera_image = tracker.process_next_frame(return_camera_image=True)

                # Wait until we receive a valid frame before starting the timer
                if not experiment_started:
                    if estimates is None:
                        continue
                    else:
                        experiment_started = True
                        start_time = current_time
                        last_time = current_time
                        delta_time = 0
                        logger.info("Experiment started. Start Time: {}".format(datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')))
                else:
                    elapsed_time = current_time - start_time
                    if elapsed_time > total_duration:
                        break
                    delta_time = current_time - last_time
                    last_time = current_time
                    if delta_time <= 0:
                        delta_time = 1e-6

                # Determine current phase based on elapsed experiment time
                elapsed_time = current_time - start_time
                if elapsed_time < duration_pre:
                    phase = 'pre'
                elif elapsed_time < duration_pre + duration_stim:
                    phase = 'stim'
                else:
                    phase = 'post'

                # Prepare the drawing: always draw the arena background; add the central circle only during stimulus phase.
                background = Drawing()
                background.add_circle(center, arena_radius, color=background_color, fill=True)
                if phase == 'stim':
                    background.add_circle(center, radius, color=trail_color, fill=False, line_width=thickness)

                if estimates is not None:
                    data = np.zeros(args.n_targets, dtype=data_dtype)
                    for estimate in estimates:
                        # Get color at the animal's current position (if available)
                        if not np.isnan(estimate['position']).any():
                            color_at_position = artist.get_values_at_camera_coords([estimate['position']])[0]
                            polar = cartesian_to_polar((estimate['position'][0], estimate['position'][1], 0, 0), center)
                            distance_from_stimulus = abs(polar[0] - radius)
                        else:
                            color_at_position = [0.0, 0.0, 0.0]
                            distance_from_stimulus = np.inf

                        # Update in_task status based on proximity to where the stimulus circle would be.
                        if distance_from_stimulus < thickness/2:
                            in_task[estimate['id']] = True
                            time_since_last_encounter[estimate['id']] = 0
                        elif in_task[estimate['id']]:
                            time_since_last_encounter[estimate['id']] += delta_time
                            if time_since_last_encounter[estimate['id']] > max_time_outside_intask:
                                in_task[estimate['id']] = False
                                time_since_last_encounter[estimate['id']] = 0

                        logger.info(
                            f"Frame: {frame_no}, Phase: {phase}, ID: {estimate['id']}, "
                            f"Position: {estimate['position'][0]:.0f}, {estimate['position'][1]:.0f}, "
                            f"Distance: {distance_from_stimulus:.2f}, In Task: {in_task[estimate['id']]}"
                        )

                        # Save phase information as a byte string (S10 type)
                        phase_bytes = phase.encode('utf-8')
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
                            distance_from_stimulus,
                            color_at_position[0],
                            color_at_position[1],
                            color_at_position[2],
                            phase_bytes
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
