#!/usr/bin/env python3
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import json
import datetime
import signal
import argparse
import sys
import threading
import traceback

# Import custom modules from the flyprojection package
from flyprojection.controllers.basler_camera import BaslerCamera
from flyprojection.projection.artist import Artist, Drawing
from flyprojection.behavior.tracker import FastTracker
from flyprojection.controllers.led_server import WS2812B_LEDController, KasaPowerController
from flyprojection.utils.logging import AsyncLogger, AsyncHDF5Saver, MultiStreamManager
from flyprojection.utils.geometry import cartesian_to_polar
from flyprojection.utils.input import get_directory_path_qt, get_string_answer_qt, get_boolean_answer_qt

# Setup signal handler for graceful shutdown
stop_event = threading.Event()
def signal_handler(signum, frame):
    print("\nKeyboard interrupt received. Stopping gracefully...")
    stop_event.set()

# Main experiment framework
def main():
    ### COMMAND LINE ARGUMENTS ###
    parser = argparse.ArgumentParser(description='Fly Projection Experiment Framework')
    parser.add_argument('--repo_dir', type=str, default='/mnt/sda1/Rishika/FlyProjection/', help='Path to the repository directory')
    parser.add_argument('--n_targets', type=int, default=5, help='Number of targets to track')
    parser.add_argument('--display', type=bool, default=False, help='Display the tracking visualization')
    parser.add_argument('--duration_pre', type=float, default=15, help='Duration of pre-stimulus phase in minutes')
    parser.add_argument('--duration_stim', type=float, default=30, help='Duration of stimulus phase in minutes')
    parser.add_argument('--duration_post', type=float, default=15, help='Duration of post-stimulus phase in minutes')
    # New parameter: radius for the red circle drawn on animals within the trail region
    parser.add_argument('--red_circle_radius', type=float, default=10, help='Radius of the red circle for animals in the trail region (in mm)')
    args = parser.parse_args()

    # Convert durations from minutes to seconds and compute total experiment duration
    duration_pre = args.duration_pre * 60
    duration_stim = args.duration_stim * 60
    duration_post = args.duration_post * 60
    total_duration = duration_pre + duration_stim + duration_post

    ### FIXED EXPERIMENT PARAMETERS ###
    # Physical parameters (in mm) for stimulus, colors, and LED configurations
    actual_radius = 42.5         # Stimulus radius in mm
    actual_max_radius = 110         # Stimulus max_radius in mm
    max_time_outside = 10        # Maximum allowed time outside stimulus zone (in seconds)

    # Colors for drawing (using normalized RGB)
    background_color = (0, 0, 0)   # Blue background
    trail_color = (1, 0, 0)        # Magenta for the stimulus/trail

    # LED panel (compass) parameters
    compass_center = (32, 3)
    compass_radius = 30
    compass_ring_width = 2
    compass_color1 = (0, 0, 0)
    compass_color2 = (0, 0, 0)
    ### END FIXED EXPERIMENT PARAMETERS ###

    ### RIG CONFIGURATION ###
    repo_dir = args.repo_dir
    configs_dir = os.path.join(repo_dir, 'configs')
    archived_configs_dir = os.path.join(configs_dir, 'archived_configs')
    rig_config_path = os.path.join(configs_dir, 'rig_config.json')
    assert os.path.isdir(configs_dir), f"Invalid configs directory: {configs_dir}"
    assert os.path.isdir(archived_configs_dir), f"Invalid archived configs directory: {archived_configs_dir}"
    assert os.path.isfile(rig_config_path), f"rig_config.json not found in {configs_dir}"

    with open(rig_config_path, 'r') as f:
        rig_config = json.load(f)

    # Process rig parameters
    center = rig_config['camera_space_arena_center']
    arena_radius = rig_config['camera_space_arena_radius']
    scaling = rig_config['physical_to_camera_scaling']
    task_boundary = scaling * 5            # Example conversion: 5 mm boundary in camera coordinates
    max_radius = scaling * actual_max_radius   # Convert physical max_radius to camera units
    radius = scaling * actual_radius         # Convert physical radius to camera units

    # convert red_circle_radius from mm to camera units
    red_circle_radius = scaling * args.red_circle_radius # Convert physical radius to camera units

    ### SETUP EXPERIMENT DIRECTORY ###
    save_dir = get_directory_path_qt("Select save directory", default=os.path.join(repo_dir, 'data'))
    experiment_name = get_string_answer_qt("Enter experiment name: ")
    experiment_dir = os.path.join(save_dir, experiment_name + datetime.datetime.now().strftime("_%Y-%m-%d_%H-%M"))
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    # Save a copy of this script for reproducibility
    os.system(f'cp {os.path.abspath(__file__)} {experiment_dir}')

    ### EXPERIMENT DATA CONFIGURATION ###
    metadata_config = {
        "dtype": np.dtype([("timestamp", np.float64), ("frame_number", np.int32)]),
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
        # Placeholder for additional experiment-specific fields (e.g., engagement, in_task, phase, etc.)
        ('phase', 'S10')
    ])
    datasets_config = {
        "data": {
            "shape": (args.n_targets,),  # One record per target per frame
            "dtype": data_dtype,
        }
    }
    global_metadata = {
        "experiment": os.path.basename(__file__).replace(".py", ""),
        "experiment_name": experiment_name,
        "experiment_dir": experiment_dir,
    }

    ### INITIALIZE VIDEO STREAMS AND OTHER MANAGERS ###
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

    # Confirm start of experiment
    start_experiment = get_boolean_answer_qt("Start experiment?")
    if not start_experiment:
        sys.exit()

    # Initialize common runtime variables
    times = []
    frame_no = 0

    # Setup the signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)

    ### MAIN EXPERIMENT RUN ###
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
            level="INFO",
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

        # Setup LED panel with a radial pattern
        arena_surround = compass.create_radial(
            center=compass_center,
            ring_width=compass_ring_width,
            color1=compass_color1,
            color2=compass_color2,
        )
        compass.send_image(arena_surround)

        # Start camera and initialize experiment timing
        camera.start()
        experiment_started = False
        start_time = None
        last_time = None

        try:
            while True:
                current_time = time.time()
                estimates, camera_image = tracker.process_next_frame(return_camera_image=True)

                # Wait for the first valid frame before starting the experiment clock
                if not experiment_started:
                    if estimates is None:
                        continue
                    else:
                        experiment_started = True
                        start_time = current_time
                        last_time = current_time
                        delta_time = 0
                        logger.warning("Experiment started. Current timestamp: {}".format(datetime.datetime.now()))
                else:
                    elapsed_time = current_time - start_time
                    if elapsed_time > total_duration:
                        break
                    delta_time = current_time - last_time
                    last_time = current_time
                    if delta_time <= 0:
                        delta_time = 1e-6

                # Determine experiment phase (pre, stim, post)
                elapsed_time = current_time - start_time
                if elapsed_time < duration_pre:
                    phase = 'pre'
                elif elapsed_time < duration_pre + duration_stim:
                    phase = 'stim'
                else:
                    phase = 'post'

                # Build drawing for current frame
                background = Drawing()
                background.add_circle(center, arena_radius, color=background_color, fill=True)

                # Process each estimate and, in stim phase, draw a red circle if the animal is within the trail region
                if estimates is not None:
                    data = np.zeros(args.n_targets, dtype=data_dtype)
                    for estimate in estimates:
                        # Example: extract tracking information and compute additional metrics if needed
                        if not np.isnan(estimate['position']).any():
                            color_at_position = artist.get_values_at_camera_coords([estimate['position']])[0]
                            # Compute the radial distance from the arena center
                            dx = estimate['future_position'][0] - center[0]
                            dy = estimate['future_position'][1] - center[1]
                            animal_r = np.sqrt(dx**2 + dy**2)
                            # In stim phase, if the animal is within the trail region, draw a red circle
                            if phase == 'stim' and animal_r < max_radius:
                                background.add_circle(estimate['future_position'], red_circle_radius, color=(1, 0, 0), fill=False, line_width=5)
                                log_message = f"Animal {estimate['id']} is within the trail region."
                                logger.info(log_message)
                        else:
                            color_at_position = [0.0, 0.0, 0.0]
                        # Populate data record (modify as needed)
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
                            phase_bytes
                        )

                # Generate stimulus image using the drawing
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
                    if experiment_started:
                        times.append(current_time - last_time)

                if stop_event.is_set():
                    print("Exiting main loop due to KeyboardInterrupt...")
                    print(f"Collected {frame_no} frames. Verify that data was saved correctly.")
                    break

        except Exception as e:
            print("Exiting due to exception:", e)
            traceback.print_exc()
        finally:
            camera.stop()
            if times:
                fps = 1 / np.mean(times)
                print(f"Average frame rate: {fps:.2f} Hz")
            else:
                print("No frames recorded.")
            # Save timing information and generate a frame rate plot
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

    # Overlay the camera and stimulus videos using ffmpeg for a final composite output
    os.system(
        f'cd {experiment_dir} && '
        f'ffmpeg -i camera.mp4 -i stimulus.mp4 -filter_complex "[0:v]format=rgb24,scale=w=640:h=480[camera];'
        f'[1:v]scale=w=640:h=480[stimulus];[camera][stimulus]blend=all_mode=\'overlay\':all_opacity=0.5[out]" '
        f'-map "[out]" -preset fast overlay_output.mp4 && '
        f'ffmpeg -itsscale 0.05 -i overlay_output.mp4 -c copy speedup.mp4'
    )
    print("Experiment complete. Exiting...")

if __name__ == "__main__":
    main()
