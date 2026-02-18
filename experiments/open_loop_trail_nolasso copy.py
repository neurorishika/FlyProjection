from flyprojection.controllers.basler_camera import BaslerCamera
from flyprojection.projection.artist import Artist, Drawing
from flyprojection.behavior.tracker import FastTracker
from flyprojection.controllers.led_server import (
    WS2812B_LEDController,
    KasaPowerController,
)
from flyprojection.utils.logging import AsyncLogger, AsyncHDF5Saver, MultiStreamManager
from flyprojection.utils.geometry import cartesian_to_polar
from flyprojection.utils.input import (
    get_directory_path_qt,
    get_string_answer_qt,
    get_boolean_answer_qt,
)
import numpy as np
import time
import matplotlib.pyplot as plt
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
    stop_event.set()  # Signal threads to stop


### DEFINE FIXED EXPERIMENT PARAMETERS ###

# Experiment parameters
burn_in_time = 60  # seconds
max_time = burn_in_time + 15 * 60  # seconds

# Task Specific Parameters
actual_radius = 50  # mm; radius of the trail
actual_thickness = 5  # mm; thickness of the trail
max_time_outside_intask = 10  # seconds; time after which a fly is considered disengaged

# compass parameters
compass_center = (32, 3)
compass_radius = 30
compass_ring_width = 2
compass_color1 = (0, 0, 0)
compass_color2 = (0, 0, 0)

# trail parameters
trail_color = (1, 0, 1)
background_color = (0, 0, 1)

arena_diameter_mm = 75
strip_width_mm = 20

arena_radius = (arena_diameter_mm / 2) * rig_config["physical_to_camera_scaling"]
strip_width = strip_width_mm * rig_config["physical_to_camera_scaling"]

strip_color = (1, 0, 0)  # optogenetic light
background_color = (0, 0, 1)

### END FIXED EXPERIMENT PARAMETERS ###


if __name__ == "__main__":

    # Package all defined variables into a dictionary
    variables = dict(locals())

    # Remove variables that should not be saved
    for var in [
        "__name__",
        "__doc__",
        "__package__",
        "__loader__",
        "__spec__",
        "__annotations__",
        "variables",
    ]:
        if var in variables:
            del variables[var]

    ### DEFINE CUSTOM PARAMETERS ###

    # define and parse command line arguments
    parser = argparse.ArgumentParser(
        description="Open Loop Fly Projection System - Trail Only (No Lasso)"
    )
    parser.add_argument(
        "--repo_dir",
        type=str,
        default="/mnt/sda1/Rishika/FlyProjection/",
        help="Path to the repository directory",
    )
    parser.add_argument(
        "--display", type=bool, default=False, help="Display the tracking visualization"
    )
    args = parser.parse_args()

    ### END CUSTOM PARAMETERS ###

    # set the repository directory
    repo_dir = args.repo_dir

    # get the rig configuration

    assert os.path.isdir(
        os.path.join(repo_dir, "configs")
    ), f"Invalid configs directory: {os.path.join(repo_dir, 'configs')}"
    assert os.path.isdir(
        os.path.join(repo_dir, "configs", "archived_configs")
    ), f"Invalid configs directory: {os.path.join(repo_dir, 'configs', 'archived_configs')}"
    assert os.path.isfile(
        os.path.join(repo_dir, "configs", "rig_config.json")
    ), f"rig_config.json file not found in {os.path.join(repo_dir, 'configs')}"

    with open(os.path.join(repo_dir, "configs", "rig_config.json"), "r") as f:
        rig_config = json.load(f)

    ### PROCESS RIG PARAMETERS ###

    # Set up the real parameters
    center = rig_config["camera_space_arena_center"]
    arena_radius = rig_config["camera_space_arena_radius"]

    thickness = (
        actual_thickness * rig_config["physical_to_camera_scaling"]
    )  # mm to pixels
    radius = actual_radius * rig_config["physical_to_camera_scaling"]  # mm to pixels

    ### END PROCESS RIG PARAMETERS ###

    # SETUP EXPERIMENT DIRECTORY

    # get save directory from user
    save_dir = get_directory_path_qt(
        "Select save directory", default=os.path.join(repo_dir, "data")
    )

    # get experiment name from user
    experiment_name = get_string_answer_qt("Enter experiment name: ")

    # create experiment directory by combining save directory and experiment name
    experiment_dir = os.path.join(
        save_dir, experiment_name + datetime.datetime.now().strftime("_%Y-%m-%d_%H-%M")
    )

    # create experiment directory if it does not exist
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    ## copy this file to the experiment directory
    os.system(f"cp {os.path.abspath(__file__)} {experiment_dir}")

    # Ask user for number of targets
    n_targets = int(
        get_string_answer_qt("Enter number of targets to track:", default="5")
    )

    ### SETUP EXPERIMENT DATA CONFIGURATION ###

    # Setup configuration
    global_metadata = {
        "experiment": os.path.basename(__file__).replace(".py", ""),
        "experiment_name": experiment_name,
        "experiment_dir": experiment_dir,
    }

    metadata_config = {
        "dtype": np.dtype([("timestamp", np.float64), ("frame_number", np.int32)]),
    }
    data_dtype = np.dtype(
        [
            ("id", np.int32),
            ("position_x", np.float64),
            ("position_y", np.float64),
            ("velocity", np.float64),
            ("velocity_smooth", np.float64),
            ("angle", np.float64),
            ("angle_smooth", np.float64),
            ("direction", np.float64),
            ("direction_smooth", np.float64),
            ("angular_velocity_smooth", np.float64),
            ("time_since_start", np.float64),
            ("intask", np.bool_),
            ("distance_from_strip", np.float64),
            ("red_at_position", np.float64),
            ("green_at_position", np.float64),
            ("blue_at_position", np.float64),
        ]
    )
    datasets_config = {
        "data": {
            "shape": (n_targets,),  # One row per frame, with one entry per target
            "dtype": data_dtype,
        }
    }

    manager = MultiStreamManager(stop_event=stop_event)
    manager.add_stream(
        output_file=os.path.join(experiment_dir, "camera.mp4"),
        width=2048,
        height=2048,
        fps=20,
        codec="h264_nvenc",
        pix_fmt_in="gray",
        preset="p1",
        log_file=os.path.join(experiment_dir, "camera.log"),
    )
    manager.add_stream(
        output_file=os.path.join(experiment_dir, "stimulus.mp4"),
        width=2048,
        height=2048,
        fps=20,
        codec="h264_nvenc",
        pix_fmt_in="rgb24",
        preset="p1",
        log_file=os.path.join(experiment_dir, "stimulus.log"),
    )

    ### END SETUP EXPERIMENT DATA CONFIGURATION ###

    # START EXPERIMENT

    # ask user if they want to start the experiment
    start_experiment = get_boolean_answer_qt("Start experiment?")
    if not start_experiment:
        sys.exit()

    in_task = np.array([False] * n_targets)
    time_since_last_encounter = np.zeros(n_targets)

    times = []

    # Use camera as a context manager
    with BaslerCamera(
        index=rig_config["camera_index"],
        EXPOSURE_TIME=rig_config["experiment_exposure_time"],
        GAIN=rig_config["camera_gain"],
        WIDTH=rig_config["camera_width"],
        HEIGHT=rig_config["camera_height"],
        OFFSETX=rig_config["camera_offset_x"],
        OFFSETY=rig_config["camera_offset_y"],
        TRIGGER_MODE="Continuous",
        CAMERA_FORMAT="Mono8",
        record_video=False,
        debug=False,
    ) as camera, WS2812B_LEDController(
        host=rig_config["visual_led_panel_hostname"],
        port=rig_config["visual_led_panel_port"],
    ) as compass, KasaPowerController(
        ip=rig_config["IR_LED_IP"], default_state="on"
    ) as ir_led_controller, AsyncLogger(
        log_file=os.path.join(experiment_dir, "terminal.log"),
        logger_name=experiment_name,
        level="INFO",
    ) as logger, AsyncHDF5Saver(
        h5_filename=os.path.join(experiment_dir, "data.h5"),
        datasets_config=datasets_config,
        metadata_config=metadata_config,
        global_metadata=global_metadata,
    ) as saver, manager as manager:

        ## SETUP LED PANEL ##
        arena_surround = compass.create_radial(
            center=compass_center,
            ring_width=compass_ring_width,
            color1=compass_color1,
            color2=compass_color2,
        )
        compass.send_image(arena_surround)

        with FastTracker(
            camera=camera,
            n_targets=n_targets,
            debug=args.display,
            smoothing_alpha=0.1,
        ) as tracker, Artist(
            camera=camera, rig_config=rig_config, method="map"
        ) as artist:

            signal.signal(signal.SIGINT, signal_handler)

            ## START CAMERA ##
            camera.start()

            started = False
            frame_no = 0

            start_time = time.time()
            last_time = start_time

            try:
                while not stop_event.is_set():
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

                    estimates, camera_image = tracker.process_next_frame(
                        return_camera_image=True
                    )

                    if started and current_time - start_time < burn_in_time:
                        logger.info(f"Burn-in time: {current_time - start_time:.2f}s")
                        continue

                    # Create drawing
                    background = Drawing()

                    if started:
                        # Draw background
                        background.add_circle(
                            center, arena_radius, color=background_color, fill=True
                        )
                       # Define vertical strip boundaries (X-axis restriction)
                        strip_left = center[0] - strip_width / 2
                        strip_right = center[0] + strip_width / 2

                        # Draw vertical strip rectangle
                        background.add_rectangle(
                            top_left=(strip_left, center[1] - arena_radius),
                            width=strip_width,
                            height=2 * arena_radius,
                            color=strip_color,
                            fill=True
                        )

                    if estimates is not None:
                        # Create data
                        data = np.zeros(n_targets, dtype=data_dtype)

                        for estimate in estimates:

                            # get important statistics
                            if not np.isnan(estimate["position"]).any():
                                x_pos = estimate["position"][0]

                                if strip_left <= x_pos <= strip_right:
                                    in_strip = True
                                    distance_from_strip = 0
                                else:
                                    in_strip = False
                                    if x_pos < strip_left:
                                        distance_from_strip = strip_left - x_pos
                                    else:
                                        distance_from_strip = x_pos - strip_right

                            else:
                                in_strip = False
                                distance_from_strip = np.inf
                            # Get color at position for data recording
                            if not np.isnan(estimate["position"]).any():
                                color_at_position = artist.get_values_at_camera_coords(
                                    [estimate["position"]]
                                )[0]
                            else:
                                color_at_position = [0.0, 0.0, 0.0]

                            # check task status based on strip position
                            if (
                                not in_task[estimate["id"]]
                                and distance_from_strip < thickness / 2
                            ):
                                in_task[estimate["id"]] = True
                                time_since_last_encounter[estimate["id"]] = 0
                                logger.info(f"Fly {estimate['id']} entered task")

                            # Track time since last encounter with strip
                            if in_task[estimate["id"]]:
                                if distance_from_strip < thickness / 2:
                                    time_since_last_encounter[estimate["id"]] = 0
                                else:
                                    time_since_last_encounter[
                                        estimate["id"]
                                    ] += delta_time
                                    if (
                                        time_since_last_encounter[estimate["id"]]
                                        > max_time_outside_intask
                                    ):
                                        in_task[estimate["id"]] = False
                                       time_since_last_encounter[estimate["id"]] = 0
                                        logger.info(f"Fly {estimate['id']} exited task")
                            else:
                                time_since_last_encounter[estimate["id"]] = 0

                            # Save data
                            data[estimate["id"]] = (
                                estimate["id"],
                                estimate["position"][0],
                                estimate["position"][1],
                                estimate["velocity"],
                                estimate["velocity_smooth"],
                                estimate["angle"],
                                estimate["angle_smooth"],
                                estimate["direction"],
                                estimate["direction_smooth"],
                                estimate["angular_velocity_smooth"],
                                estimate["time_since_start"],
                                in_task[estimate["id"]],
                                distance_from_strip,
                                color_at_position[0],
                                color_at_position[1],
                                color_at_position[2],
                            )

                    logger.info(f"In Task: {in_task}")

                    draw_fn = background.get_drawing_function()
                    stimulus_image = artist.draw_geometry(
                        draw_fn, debug=False, return_camera_image=True
                    )

                    if (
                        camera_image is not None
                        and stimulus_image is not None
                        and estimates is not None
                    ):

                        # start time if not started
                        if not started:
                            start_time = current_time
                            last_time = current_time
                            started = True
                            logger.warning(
                                "Experiment started. Start Time: {}".format(
                                    datetime.datetime.fromtimestamp(
                                        start_time
                                    ).strftime("%Y-%m-%d %H:%M:%S")
                                )
                            )
                            end_time = start_time + max_time + burn_in_time
                            end_time_readable = datetime.datetime.fromtimestamp(
                                end_time
                            ).strftime("%Y-%m-%d %H:%M:%S")
                            logger.warning(
                                f"Experiment will end at {end_time_readable}"
                            )

                        metadata = {"timestamp": current_time, "frame_number": frame_no}
                        # Save data
                        saver.save_frame_async(metadata=metadata, data=data)
                        # Increment frame number
                        frame_no += 1

                        manager.write_frame_to_all([camera_image, stimulus_image])

                    if started and current_time - start_time > max_time + burn_in_time:
                        break

                    if stop_event.is_set():
                        print("Exiting main loop due to KeyboardInterrupt...")
                        # tell the user how many frames were collected
                        print(
                            f"Collected {frame_no} frames. Make sure to verify that the data was saved correctly in the log files."
                        )
                        break

            except Exception as e:
                print("Exiting due to exception:", e)

                traceback.print_exc()
            finally:
                camera.stop()
                fps = 1 / np.mean(times)
                print(
                    f"Average frame rate: {1 / np.mean(times[-int(30 * fps):]):.2f} Hz"
                )
                # save the fps to a file
                with open(os.path.join(experiment_dir, "times.txt"), "w") as f:
                    f.write("\n".join(map(str, times)))
                # plot the fps and save the plot
                fig, ax = plt.subplots(figsize=(30, 3))
                ax.plot(np.nan_to_num(1 / np.array(times)), color="black")
                # hide everything except the y-axis
                plt.box(False)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["bottom"].set_visible(False)
                plt.savefig(os.path.join(experiment_dir, "times.png"))
                plt.close(fig)

    # cd to experiment directory and run the ffmpeg command to overlay the videos:
    os.system(
        f'cd {experiment_dir} && ffmpeg -i camera.mp4 -i stimulus.mp4 -filter_complex "[0:v]format=rgb24,scale=w=640:h=480[camera];[1:v]scale=w=640:h=480[stimulus];[camera][stimulus]blend=all_mode=\'overlay\':all_opacity=0.5[out]" -map "[out]" -preset fast overlay_output.mp4 && ffmpeg -itsscale 0.05 -i overlay_output.mp4 -c copy speedup.mp4'
    )
    print("Experiment complete. Exiting...")
