import numpy as np
import time
import matplotlib.pyplot as plt
from flyprojection.controllers.basler_camera import BaslerCamera
from flyprojection.projection.artist import Artist, Drawing
from flyprojection.behavior.tracker import FastTracker
from flyprojection.controllers.led_server import (
    WS2812B_LEDController,
    KasaPowerController,
)
from flyprojection.utils.logging import AsyncLogger, AsyncHDF5Saver, MultiStreamManager
from flyprojection.utils.geometry import (
    cartesian_to_polar,
    polar_to_cartesian,
    resample_path,
    paths_intersect,
)
from flyprojection.utils.input import (
    get_directory_path_qt,
    get_string_answer_qt,
    get_boolean_answer_qt,
)
import json
import os
import datetime
import signal
import argparse
from numba import njit
import random
import traceback
import sys

# Setup signal handler for graceful shutdown
import threading

stop_event = threading.Event()


def signal_handler(signum, frame):
    print("\nKeyboard interrupt received. Stopping gracefully...")
    stop_event.set()  # Signal threads to stop


### DEFINE CUSTOM FUNCTIONS ###


@njit
def dynamics(state, t, c_r, k_r, r0, w, c_theta):
    r, theta, r_dot, theta_dot = state
    drdt = r_dot
    dr_dotdt = -c_r * r_dot - k_r * (r - r0)
    dthetadt = theta_dot
    dtheta_dotdt = -c_theta * (theta_dot - w)
    return np.array([drdt, dthetadt, dr_dotdt, dtheta_dotdt])


def check_orientation(initial_conditions_polar, r0):
    r = initial_conditions_polar[0]
    r_dot = initial_conditions_polar[2]
    return not ((r < r0 and r_dot < 0) or (r > r0 and r_dot > 0))


def integrate_with_stopping(
    f, initial_state, t_max, dt, c_r, k_r, r0, w, c_theta, epsilon, max_steps
):
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

    return np.arange(step + 1) * dt, trajectory[: step + 1]


def attempt_engagement(
    estimate,
    center,
    radius,
    engagement,
    engaged_paths,
    c_r,
    k_r,
    w,
    c_theta,
    epsilon,
    t_max,
    dt,
    max_steps,
    path_steps,
    min_v_threshold,
    max_angular_velocity,
    thickness,
    task_boundary,
):
    if (
        engagement[estimate["id"]]
        or np.isnan(estimate["position"]).any()
        or np.isnan(estimate["direction"])
        or np.isnan(estimate["velocity"])
        or np.isnan(estimate["angular_velocity_smooth"])
    ):
        return None

    if (
        estimate["velocity"] < min_v_threshold
        or abs(estimate["angular_velocity_smooth"]) > max_angular_velocity
    ):
        return None

    # Precompute values for efficiency
    delta_x = estimate["position"][0] - center[0]
    delta_y = estimate["position"][1] - center[1]
    r_dist = np.sqrt(delta_x**2 + delta_y**2)
    scale_factor = np.sin(estimate["direction"] - np.arctan2(delta_y, delta_x))
    magnitude = abs(radius - r_dist)

    if magnitude < thickness / 2 + task_boundary:
        return None

    # Initial state in polar coordinates
    state_polar = cartesian_to_polar(
        (
            estimate["position"][0],
            estimate["position"][1],
            magnitude * np.cos(estimate["direction"]),
            magnitude * np.sin(estimate["direction"]),
        ),
        center,
    )

    if not check_orientation(state_polar, radius):
        return None

    w_ = w * np.sign(state_polar[3])

    # Integrate
    _, trajectory = integrate_with_stopping(
        dynamics,
        state_polar,
        t_max,
        dt,
        c_r,
        k_r,
        radius,
        w_,
        c_theta,
        epsilon,
        max_steps,
    )

    # Convert trajectory to Cartesian and check intersections
    cartesian_trajectory = np.array(
        [polar_to_cartesian(st, center) for st in trajectory]
    )
    if any(
        paths_intersect(cartesian_trajectory, path) for path in engaged_paths.values()
    ):
        return None

    # resample to path_steps
    cartesian_trajectory = resample_path(cartesian_trajectory, path_steps)

    return cartesian_trajectory


### END CUSTOM FUNCTIONS ###


### DEFINE FIXED EXPERIMENT PARAMETERS ###

# Experiment parameters
burn_in_time = 30  # seconds
max_time = 330  # seconds

# Path parameters
k_r = 1.0
c_r = 2 * np.sqrt(k_r)  # critical damping
w = 0.2
c_theta = 1.0
epsilon = 0.01
t_max = 1000
dt = 0.05
max_steps = int(t_max / dt)
path_steps = 30

min_v_threshold = 150
max_angular_velocity = np.pi / 4

# Task Specific Parameters
actual_radius = 50  # mm; radius of the trail
actual_thickness = 5  # mm; thickness of the trail
max_time_outside_pretask = 3  # seconds; time after which a fly is considered disengaged
max_time_outside_intask = 10  # seconds; time after which a fly is considered disengaged
actual_task_boundary = (
    5  # mm; boundary around when left results countdown to disengagement
)

# compass parameters
compass_center = (32, 3)
compass_radius = 30
compass_ring_width = 2
compass_color1 = (0, 0, 0)
compass_color2 = (0, 0, 0)

# trail parameters
trail_color = (1, 0, 1)
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
        description="Open/Closed Loop Fly Projection System Rig Configuration"
    )
    parser.add_argument(
        "--repo_dir",
        type=str,
        default="/mnt/sda1/Rishika/FlyProjection/",
        help="Path to the repository directory",
    )
    parser.add_argument(
        "--sphere_of_infuence",
        type=int,
        default=100,
        help="Dimension of the sphere of influence",
    )
    parser.add_argument(
        "--n_targets", type=int, default=5, help="Number of targets to track"
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

    task_boundary = (
        actual_task_boundary * rig_config["physical_to_camera_scaling"]
    )  # mm to pixels
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

    ### SETUP EXPERIMENT DATA CONFIGURATION ###

    # Setup configuration
    global_metadata = {
        "experiment": os.path.basename(__file__).replace(".py", ""),
        "experiment_name": experiment_name,
        "experiment_dir": experiment_dir,
        # "variables": variables,
        # "arguments": args,
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
            ("engagement", np.bool_),
            ("intask", np.bool_),
            ("distance_from_trail", np.float64),
            ("red_at_position", np.float64),
            ("blue_at_position", np.float64),
            ("green_at_position", np.float64),
            ("engagement_path", np.uint16, (path_steps, 2)),
        ]
    )
    datasets_config = {
        "data": {
            "shape": (args.n_targets,),  # One row per frame, with one entry per target
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

    engagement = np.array([False] * args.n_targets)
    in_task = np.array([False] * args.n_targets)
    time_since_last_encounter = np.zeros(args.n_targets)

    engaged_paths = {}
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
        level="WARNING",
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
            n_targets=args.n_targets,
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

                    background = Drawing()

                    if started:
                        # Draw background
                        background.add_circle(
                            center, arena_radius, color=background_color, fill=True
                        )
                        # Draw a circle in the center
                        background.add_circle(
                            center,
                            radius,
                            color=trail_color,
                            fill=False,
                            line_width=thickness,
                        )

                    if estimates is not None:
                        # Create data
                        data = np.zeros(args.n_targets, dtype=data_dtype)

                        # Shuffle estimates to avoid engaging the same fly every time
                        random.shuffle(estimates)

                        for estimate in estimates:

                            path = None

                            # get important_statistics
                            if not np.isnan(estimate["position"]).any():
                                color_at_position = artist.get_values_at_camera_coords(
                                    [estimate["position"]]
                                )[0]
                                polar = cartesian_to_polar(
                                    (
                                        estimate["position"][0],
                                        estimate["position"][1],
                                        0,
                                        0,
                                    ),
                                    center,
                                )
                                distance_from_trail = np.abs(polar[0] - radius)
                            else:
                                color_at_position = [0.0, 0.0, 0.0]
                                distance_from_trail = np.inf

                            # check task status
                            if (
                                not in_task[estimate["id"]]
                                and distance_from_trail < thickness / 2
                            ):
                                in_task[estimate["id"]] = True
                                engagement[estimate["id"]] = False
                                engaged_paths.pop(estimate["id"], None)
                                time_since_last_encounter[estimate["id"]] = 0

                            if in_task[estimate["id"]]:
                                if distance_from_trail < thickness / 2:
                                    time_since_last_encounter[estimate["id"]] = 0
                                else:
                                    time_since_last_encounter[
                                        estimate["id"]
                                    ] += delta_time
                            else:
                                if color_at_position[0] > 0.0:
                                    time_since_last_encounter[estimate["id"]] = 0
                                else:
                                    time_since_last_encounter[
                                        estimate["id"]
                                    ] += delta_time

                            logger.info(
                                f"Frame: {frame_no}, Framerate: {1 / delta_time if delta_time > 0 else 0:.2f} Hz, "
                                f"ID: {estimate['id']}, Position: {estimate['position'][0]:.0f}, {estimate['position'][1]:.0f}, "
                                f"Velocity: {estimate['velocity_smooth']:.2f}, Angle: {np.rad2deg(estimate['angle_smooth']):.2f}°, "
                                f"Time Since Start: {estimate['time_since_start']:.2f}s, Angular Velocity: {estimate['angular_velocity_smooth']:.2f}"
                            )

                            if not engagement[estimate["id"]]:
                                if in_task[estimate["id"]]:
                                    if (
                                        time_since_last_encounter[estimate["id"]]
                                        > max_time_outside_intask
                                        or distance_from_trail
                                        > thickness / 2 + task_boundary
                                    ):
                                        in_task[estimate["id"]] = False
                                        path = attempt_engagement(
                                            estimate,
                                            center,
                                            radius,
                                            engagement,
                                            engaged_paths,
                                            c_r,
                                            k_r,
                                            w,
                                            c_theta,
                                            epsilon,
                                            t_max,
                                            dt,
                                            max_steps,
                                            path_steps,
                                            min_v_threshold,
                                            max_angular_velocity,
                                            thickness,
                                            task_boundary,
                                        )
                                    else:
                                        path = None
                                else:
                                    # check if fly is in odor
                                    if color_at_position[0] > 0.0:
                                        # start task
                                        in_task[estimate["id"]] = True
                                    # else, do standard engagement
                                    else:
                                        path = attempt_engagement(
                                            estimate,
                                            center,
                                            radius,
                                            engagement,
                                            engaged_paths,
                                            c_r,
                                            k_r,
                                            w,
                                            c_theta,
                                            epsilon,
                                            t_max,
                                            dt,
                                            max_steps,
                                            path_steps,
                                            min_v_threshold,
                                            max_angular_velocity,
                                            thickness,
                                            task_boundary,
                                        )

                                if path is not None:
                                    # Add path and mark engagement
                                    background.add_path(
                                        list(zip(path[:, 0], path[:, 1])),
                                        color=trail_color,
                                        line_width=thickness,
                                    )
                                    engagement[estimate["id"]] = True
                                    engaged_paths[estimate["id"]] = path

                            else:

                                # Already engaged, check if the fly is outside for too long or reaches the central circle
                                if (
                                    time_since_last_encounter[estimate["id"]]
                                    > max_time_outside_pretask
                                ):
                                    in_task[estimate["id"]] = False
                                    engagement[estimate["id"]] = False
                                    engaged_paths.pop(estimate["id"], None)
                                    time_since_last_encounter[estimate["id"]] = 0
                                else:
                                    # Redraw existing path
                                    path = engaged_paths[estimate["id"]]
                                    background.add_path(
                                        list(zip(path[:, 0], path[:, 1])),
                                        color=trail_color,
                                        line_width=thickness,
                                    )

                            # turn path into numpy array of uint16
                            if path is not None:
                                path = np.array(path, dtype=np.uint16)

                            # Update frame data
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
                                engagement[estimate["id"]],
                                in_task[estimate["id"]],
                                distance_from_trail,
                                color_at_position[0],
                                color_at_position[1],
                                color_at_position[2],
                                (
                                    np.array(path)
                                    if path is not None
                                    else np.ones((path_steps, 2), dtype=np.uint16)
                                    * np.nan
                                ),
                            )

                    logger.info(f"Engagement: {engagement} In Task: {in_task}")

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
                            start_time = time.time()
                            last_time = None
                            started = True
                            logger.warning(f"Started at {start_time}")

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
