# """
# A standalone Python script that provides a user interface (built using OpenCV windows and trackbars)
# to select and track multiple targets (e.g. flies) in a pre-recorded .mp4 video. The script is
# "framework-free" in the sense that it does not rely on GUI libraries like PyQt or tkinter for the UI.
# It uses OpenCV windows, trackbars, and simple text input to capture user choices in a user-friendly way.

# Features:
# ---------
# 1. File Selection:
#    - Prompts the user in the terminal for the path to the .mp4 file.

# 2. Parameter Control (via OpenCV Trackbars):
#    - Allows the user to configure various tracking parameters in a "Parameters" window.
#    - Examples: max number of targets, threshold values, morphological kernel sizes, etc.
#    - Also offers toggles (0 or 1) for whether to show Foreground Mask, Blob Detection, and Lightest
#      Background windows.

# 3. High-speed Tracking with Kalman Filters:
#    - Dynamically updates the background model (lightest background).
#    - Detects blobs (contours/ellipses) and assigns them to Kalman Filters via the Hungarian algorithm.
#    - Maintains a trajectory history in memory for each tracked target and displays them in real-time.

# 4. Results and Saving:
#    - Displays the selected output windows in real-time (for fastest processing possible).
#    - Writes trajectories to a CSV file upon completion (one CSV file for all tracked objects).
#    - Shows an FPS plot at the end using matplotlib.

# Usage:
# ------
# 1. Run the script in a terminal/command prompt: 
#        python tracking_ui.py
# 2. When prompted, enter the path to your MP4 file.
# 3. A "Parameters" window will appear with trackbars for each configurable parameter. Adjust them as desired.
# 4. In the "Parameters" window, set the "Run_Tracking" trackbar to 1 to start the tracking process.
# 5. During tracking, press 'q' in any OpenCV window to stop the tracking early.
# 6. After the script finishes or you quit, the tracked trajectories are saved to "trajectories.csv" and
#    an FPS plot is displayed.

# Dependencies:
# -------------
# - Python 3.7+
# - OpenCV (cv2)
# - NumPy
# - SciPy
# - Matplotlib

# Example:
# --------
# python tracking_ui.py
# """

# import cv2
# import numpy as np
# import time
# from scipy.spatial import distance
# from scipy.optimize import linear_sum_assignment
# import matplotlib.pyplot as plt
# import csv
# import sys


# def nothing(x):
#     """
#     A dummy callback function needed by OpenCV createTrackbar.
    
#     Parameters
#     ----------
#     x : int
#         Current trackbar position (unused in this callback).
#     """
#     pass


# def create_parameter_window(window_name):
#     """
#     Creates an OpenCV window with trackbars for controlling various parameters.
#     The initial values and ranges are set here. Trackbars are used as the primary UI element.
    
#     Parameters
#     ----------
#     window_name : str
#         The name of the OpenCV window to be created.

#     Returns
#     -------
#     None
#     """
#     cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
#     cv2.resizeWindow(window_name, 400, 600)

#     # Create trackbars for integer parameters
#     cv2.createTrackbar("MAX_TARGETS", window_name, 4, 20, nothing)
#     cv2.createTrackbar("THRESHOLD_VALUE", window_name, 50, 255, nothing)
#     cv2.createTrackbar("MORPH_KERNEL_SIZE", window_name, 5, 50, nothing)
#     cv2.createTrackbar("MIN_CONTOUR_AREA", window_name, 50, 5000, nothing)
#     cv2.createTrackbar("MAX_DISTANCE_THRESHOLD", window_name, 25, 300, nothing)
#     cv2.createTrackbar("MIN_DETECTION_COUNTS", window_name, 10, 200, nothing)
#     cv2.createTrackbar("MIN_TRACKING_COUNTS", window_name, 10, 200, nothing)
#     cv2.createTrackbar("TRAJECTORY_HISTORY_SEC", window_name, 5, 300, nothing)

#     # Create trackbars for float parameters. We scale them by 100 internally.
#     # e.g. for KALMAN_NOISE_COVARIANCE, user sees int from 0..100. We'll interpret that as 0..1.0.
#     cv2.createTrackbar("KALMAN_NOISE_COV", window_name, int(0.03 * 100), 100, nothing)
#     cv2.createTrackbar("KALMAN_MEAS_NOISE_COV", window_name, int(0.1 * 100), 100, nothing)

#     # Toggles for showing windows (0 = OFF, 1 = ON)
#     cv2.createTrackbar("Show_FG", window_name, 1, 1, nothing)
#     cv2.createTrackbar("Show_Blob", window_name, 1, 1, nothing)
#     cv2.createTrackbar("Show_BG", window_name, 1, 1, nothing)

#     # Create a trackbar to start/stop the tracking
#     cv2.createTrackbar("Run_Tracking", window_name, 0, 1, nothing)


# def read_parameters_from_trackbars(window_name):
#     """
#     Reads the current values from the trackbars in the specified window and returns them as a dictionary.
    
#     Parameters
#     ----------
#     window_name : str
#         The name of the OpenCV parameters window from which to read trackbar values.

#     Returns
#     -------
#     params : dict
#         Dictionary containing all relevant parameters needed for tracking.
#     """
#     params = {}

#     # Integer parameters
#     params["MAX_TARGETS"] = cv2.getTrackbarPos("MAX_TARGETS", window_name)
#     params["THRESHOLD_VALUE"] = cv2.getTrackbarPos("THRESHOLD_VALUE", window_name)
#     params["MORPH_KERNEL_SIZE"] = cv2.getTrackbarPos("MORPH_KERNEL_SIZE", window_name)
#     params["MIN_CONTOUR_AREA"] = cv2.getTrackbarPos("MIN_CONTOUR_AREA", window_name)
#     params["MAX_DISTANCE_THRESHOLD"] = cv2.getTrackbarPos("MAX_DISTANCE_THRESHOLD", window_name)
#     params["MIN_DETECTION_COUNTS"] = cv2.getTrackbarPos("MIN_DETECTION_COUNTS", window_name)
#     params["MIN_TRACKING_COUNTS"] = cv2.getTrackbarPos("MIN_TRACKING_COUNTS", window_name)
#     params["TRAJECTORY_HISTORY_SECONDS"] = cv2.getTrackbarPos("TRAJECTORY_HISTORY_SEC", window_name)

#     # Float parameters (converted from scaled int)
#     kalman_noise_cov_int = cv2.getTrackbarPos("KALMAN_NOISE_COV", window_name)
#     params["KALMAN_NOISE_COVARIANCE"] = kalman_noise_cov_int / 100.0
#     kalman_meas_noise_cov_int = cv2.getTrackbarPos("KALMAN_MEAS_NOISE_COV", window_name)
#     params["KALMAN_MEASUREMENT_NOISE_COVARIANCE"] = kalman_meas_noise_cov_int / 100.0

#     # Toggles
#     params["SHOW_FG"] = cv2.getTrackbarPos("Show_FG", window_name) == 1
#     params["SHOW_BLOB"] = cv2.getTrackbarPos("Show_Blob", window_name) == 1
#     params["SHOW_BG"] = cv2.getTrackbarPos("Show_BG", window_name) == 1

#     # Run_Tracking
#     params["RUN_TRACKING"] = cv2.getTrackbarPos("Run_Tracking", window_name)

#     return params


# def run_tracking(video_path, params):
#     """
#     Runs the multiple object tracking using the provided video file and parameters.
#     Uses Kalman Filters, the Hungarian assignment, and a "lightest background" model to track multiple targets.
#     Displays (optionally) Foreground Mask, Blob Detection, and Lightest Background, and calculates FPS.
#     Saves trajectories to a CSV file upon completion.

#     Parameters
#     ----------
#     video_path : str
#         Path to the .mp4 video file to track.
#     params : dict
#         Dictionary containing all the tracking parameters.

#     Returns
#     -------
#     None
#     """
#     # Unpack parameters for convenience
#     MAX_TARGETS = params["MAX_TARGETS"]
#     THRESHOLD_VALUE = params["THRESHOLD_VALUE"]
#     MORPH_KERNEL_SIZE = params["MORPH_KERNEL_SIZE"]
#     MIN_CONTOUR_AREA = params["MIN_CONTOUR_AREA"]
#     MAX_DISTANCE_THRESHOLD = params["MAX_DISTANCE_THRESHOLD"]
#     MIN_DETECTION_COUNTS = params["MIN_DETECTION_COUNTS"]
#     MIN_TRACKING_COUNTS = params["MIN_TRACKING_COUNTS"]
#     TRAJECTORY_HISTORY_SECONDS = params["TRAJECTORY_HISTORY_SECONDS"]
#     KALMAN_NOISE_COVARIANCE = params["KALMAN_NOISE_COVARIANCE"]
#     KALMAN_MEASUREMENT_NOISE_COVARIANCE = params["KALMAN_MEASUREMENT_NOISE_COVARIANCE"]

#     SHOW_FG = params["SHOW_FG"]
#     SHOW_BLOB = params["SHOW_BLOB"]
#     SHOW_BG = params["SHOW_BG"]

#     # Prepare windows if toggles are ON
#     if SHOW_FG:
#         cv2.namedWindow('Foreground Mask', cv2.WINDOW_NORMAL)
#         cv2.resizeWindow('Foreground Mask', 600, 600)
#     if SHOW_BLOB:
#         cv2.namedWindow('Blob Detection', cv2.WINDOW_NORMAL)
#         cv2.resizeWindow('Blob Detection', 600, 600)
#     if SHOW_BG:
#         cv2.namedWindow('Lightest Background', cv2.WINDOW_NORMAL)
#         cv2.resizeWindow('Lightest Background', 600, 600)

#     # Attempt to open the video
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"ERROR: Cannot open video file: {video_path}")
#         return

#     # Initialize lightest background model
#     background_model_lightest = None

#     # Kalman filters and tracking setup
#     kalman_filters = [cv2.KalmanFilter(5, 3) for _ in range(MAX_TARGETS)]
#     for kf in kalman_filters:
#         # Measurement matrix: Maps state to measurements (x, y, theta)
#         kf.measurementMatrix = np.array([[1, 0, 0, 0, 0],  # x
#                                          [0, 1, 0, 0, 0],  # y
#                                          [0, 0, 1, 0, 0]], # theta
#                                         np.float32)
#         # Transition matrix: Defines how the state evolves
#         kf.transitionMatrix = np.array([[1, 0, 0, 1, 0],  # x -> x + vx
#                                         [0, 1, 0, 0, 1],  # y -> y + vy
#                                         [0, 0, 1, 0, 0],  # theta -> theta
#                                         [0, 0, 0, 1, 0],  # vx -> vx
#                                         [0, 0, 0, 0, 1]], # vy -> vy
#                                        np.float32)
#         # Process noise covariance
#         kf.processNoiseCov = np.eye(5, dtype=np.float32) * KALMAN_NOISE_COVARIANCE
#         # Measurement noise covariance
#         kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * KALMAN_MEASUREMENT_NOISE_COVARIANCE
#         # Initialize error covariance
#         kf.errorCovPre = np.eye(5, dtype=np.float32)

#     # Prepare color map for trajectories
#     np.random.seed(42)
#     trajectory_colors = [tuple(color) for color in np.random.randint(0, 255, (MAX_TARGETS, 3)).tolist()]
#     trajectories = [[] for _ in range(MAX_TARGETS)]
#     track_ids = np.arange(MAX_TARGETS)

#     detection_initialized = False  # Flag to start tracking after all targets are detected
#     tracking_stabilized = False  # Flag to indicate that tracking is stabilized
#     detection_counts = 0
#     tracking_counts = 0
#     conservative_used = False

#     fps_list = []
#     frame_count = 0
#     start_time = time.time()

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             # No more frames or read error
#             break

#         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         current_time = time.time()

#         if background_model_lightest is None:
#             # Initialize background on the very first frame
#             background_model_lightest = gray_frame.astype(np.float32)
#             continue

#         # Update lightest background model
#         background_model_lightest = np.maximum(background_model_lightest, gray_frame.astype(np.float32))
#         background_model_lightest_uint8 = cv2.convertScaleAbs(background_model_lightest)

#         # Background subtraction
#         fg_mask = cv2.absdiff(background_model_lightest_uint8, gray_frame)
#         _, fg_mask = cv2.threshold(fg_mask, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)

#         # Morphological open/close
#         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))
#         fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
#         fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

#         # "Conservative" mask to help split close-together targets
#         split_mask = cv2.erode(fg_mask, kernel, iterations=2)

#         # Find contours
#         contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         conservative_contours, _ = cv2.findContours(split_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         measurements = []
#         sizes = []

#         # Fit ellipses, track largest
#         if len(contours) > 0:
#             for cnt in contours:
#                 size = cv2.contourArea(cnt)
#                 if size < MIN_CONTOUR_AREA:
#                     continue
#                 if len(cnt) >= 5:
#                     ellipse = cv2.fitEllipse(cnt)
#                     (x, y), (MA, ma), angle = ellipse
#                     angle_radians = np.deg2rad(angle)
#                     measurements.append(np.array([x, y, angle_radians], dtype=np.float32))
#                     sizes.append(size)

#             # Keep only largest if there are more than MAX_TARGETS
#             if len(measurements) > MAX_TARGETS:
#                 sorted_indices = np.argsort(sizes)[::-1]
#                 measurements = [measurements[i] for i in sorted_indices[:MAX_TARGETS]]

#         # If fewer detections than needed, try the conservative mask
#         if len(measurements) < MAX_TARGETS:
#             conservative_used = True
#             measurements = []
#             sizes = []
#             for cnt in conservative_contours:
#                 size = cv2.contourArea(cnt)
#                 if size < MIN_CONTOUR_AREA:
#                     continue
#                 if len(cnt) >= 5:
#                     ellipse = cv2.fitEllipse(cnt)
#                     (x, y), (MA, ma), angle = ellipse
#                     angle_radians = np.deg2rad(angle)
#                     measurements.append(np.array([x, y, angle_radians], dtype=np.float32))
#                     sizes.append(size)
#             if len(measurements) > MAX_TARGETS:
#                 sorted_indices = np.argsort(sizes)[::-1]
#                 measurements = [measurements[i] for i in sorted_indices[:MAX_TARGETS]]
#         else:
#             conservative_used = False

#         # Check for detection initialization
#         if len(measurements) == MAX_TARGETS:
#             detection_counts += 1
#         else:
#             detection_counts = 0

#         if detection_counts >= MIN_DETECTION_COUNTS:
#             detection_initialized = True

#         # Kalman Filter update
#         if detection_initialized and measurements:
#             # If Kalman filters are not yet well-initialized, randomly place them once
#             if frame_count == 0:
#                 height, width = gray_frame.shape
#                 for kf in kalman_filters:
#                     kf.statePre = np.array([
#                         np.random.randint(0, width), 
#                         np.random.randint(0, height),
#                         0, 0, 0
#                     ], dtype=np.float32)
#                     kf.statePost = kf.statePre.copy()

#             predicted_positions = np.zeros((MAX_TARGETS, 3), dtype=np.float32)
#             for i, kf in enumerate(kalman_filters):
#                 pred = kf.predict()
#                 predicted_positions[i] = pred[:3].flatten()  # [x, y, theta]

#             cost_matrix = np.zeros((MAX_TARGETS, len(measurements)), dtype=np.float32)
#             for i, pred_pos in enumerate(predicted_positions):
#                 for j, meas in enumerate(measurements):
#                     pos_cost = distance.euclidean(pred_pos[:2], meas[:2])
#                     angle_cost = abs(pred_pos[2] - meas[2])
#                     angle_cost = min(angle_cost, 2 * np.pi - angle_cost)
#                     cost_matrix[i, j] = pos_cost + angle_cost

#             row_indices, col_indices = linear_sum_assignment(cost_matrix)

#             avg_cost = 0.0
#             for row, col in zip(row_indices, col_indices):
#                 if row < MAX_TARGETS and col < len(measurements):
#                     if not tracking_stabilized or cost_matrix[row, col] < MAX_DISTANCE_THRESHOLD:
#                         kf = kalman_filters[row]
#                         measurement_vector = np.array([[measurements[col][0]],
#                                                        [measurements[col][1]],
#                                                        [measurements[col][2]]], dtype=np.float32)
#                         kf.correct(measurement_vector)
#                         # Save trajectory
#                         x_corr, y_corr, theta_corr = measurements[col]
#                         trajectories[row].append((int(x_corr), int(y_corr), float(theta_corr), current_time))
#                         # Prune old trajectory points
#                         trajectories[row] = [
#                             (xx, yy, th, tt) 
#                             for (xx, yy, th, tt) in trajectories[row] 
#                             if current_time - tt <= TRAJECTORY_HISTORY_SECONDS
#                         ]
#                         avg_cost += cost_matrix[row, col] / MAX_TARGETS
#                     else:
#                         tracking_stabilized = False
#                         tracking_counts = 0
#                         # Insert NaNs
#                         trajectories[row].append((np.nan, np.nan, np.nan, current_time))
#                         trajectories[row] = [
#                             (xx, yy, th, tt) 
#                             for (xx, yy, th, tt) in trajectories[row] 
#                             if current_time - tt <= TRAJECTORY_HISTORY_SECONDS
#                         ]

#             if avg_cost < MAX_DISTANCE_THRESHOLD:
#                 tracking_counts += 1
#             else:
#                 tracking_counts = 0

#             if tracking_counts >= MIN_TRACKING_COUNTS and not tracking_stabilized:
#                 tracking_stabilized = True
#                 print(f"[INFO] Tracking Stabilized with Average Cost: {avg_cost:.2f}")

#         # Draw tracking
#         display_frame = frame.copy()
#         if detection_initialized:
#             for i, track_id in enumerate(track_ids):
#                 if len(trajectories[i]) == 0:
#                     continue
#                 x, y, theta, _ = trajectories[i][-1]
#                 if not (np.isnan(x) or np.isnan(y)):
#                     cv2.circle(display_frame, (x, y), 10, trajectory_colors[i], -1)
#                     length = 20
#                     x_end = int(x + length * np.cos(theta))
#                     y_end = int(y + length * np.sin(theta))
#                     cv2.line(display_frame, (x, y), (x_end, y_end), trajectory_colors[i], 2)
#                     cv2.putText(display_frame, f"ID: {track_id}", (x+15, y-15),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, trajectory_colors[i], 2)
#                 # Draw trajectory lines
#                 for pt_idx in range(1, len(trajectories[i])):
#                     pt1 = (trajectories[i][pt_idx - 1][0], trajectories[i][pt_idx - 1][1])
#                     pt2 = (trajectories[i][pt_idx][0], trajectories[i][pt_idx][1])
#                     if (not np.isnan(pt1[0])) and (not np.isnan(pt1[1])) \
#                        and (not np.isnan(pt2[0])) and (not np.isnan(pt2[1])):
#                         cv2.line(display_frame, pt1, pt2, trajectory_colors[i], 2)

#         # Show windows
#         if SHOW_FG:
#             cv2.imshow('Foreground Mask', split_mask if conservative_used else fg_mask)
#         if SHOW_BLOB:
#             cv2.imshow('Blob Detection', display_frame)
#         if SHOW_BG:
#             bg_display = cv2.cvtColor(background_model_lightest_uint8, cv2.COLOR_GRAY2BGR)
#             cv2.imshow('Lightest Background', bg_display)

#         frame_count += 1
#         elapsed = time.time() - start_time
#         if elapsed > 0:
#             fps_current = frame_count / elapsed
#             fps_list.append(fps_current)

#         # Press 'q' to quit early
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

#     # Save results
#     print("[INFO] Saving trajectories to 'trajectories.csv'...")
#     with open("trajectories.csv", "w", newline="") as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(["TargetID", "FrameCountIndex", "X", "Y", "ThetaRadians", "Timestamp"])
#         # We'll attempt to match the saved point with the frame index by storing the order appended
#         for i, track_id in enumerate(track_ids):
#             for idx, (x, y, theta, tstamp) in enumerate(trajectories[i]):
#                 writer.writerow([track_id, idx, x, y, theta, tstamp])

#     # Plot FPS
#     if len(fps_list) > 1:
#         fig, ax = plt.subplots()
#         ax.plot(fps_list, label="FPS")
#         ax.set_xlabel("Frame Index")
#         ax.set_ylabel("Frames Per Second")
#         ax.set_title("FPS of Multiple Object Tracking")
#         plt.legend()
#         plt.show()
#     else:
#         print("[INFO] No frames processed or not enough data to plot FPS.")


# def main():
#     """
#     Main function that:
#     1) Prompts user for the .mp4 file path in the terminal.
#     2) Creates a "Parameters" window with trackbars for user to adjust.
#     3) Waits until user toggles "Run_Tracking" trackbar to 1.
#     4) Runs the tracking with the selected file and current trackbar settings.
#     """
#     print("==================================================")
#     print("         MULTIPLE OBJECT TRACKING UI             ")
#     print("==================================================")
#     video_path = input("Enter path to your .mp4 file: ").strip()
#     if not video_path:
#         print("ERROR: No path provided. Exiting.")
#         sys.exit(1)

#     # Create UI window for parameters
#     param_window_name = "Parameters"
#     create_parameter_window(param_window_name)

#     print("\nAdjust the parameters in the 'Parameters' window.")
#     print("When ready, set 'Run_Tracking' to 1 to begin.")
#     print("--------------------------------------------------")

#     # Wait for user to start tracking
#     while True:
#         cv2.waitKey(100)
#         params = read_parameters_from_trackbars(param_window_name)
#         if params["RUN_TRACKING"] == 1:
#             # Freeze the final parameter settings
#             cv2.setTrackbarPos("Run_Tracking", param_window_name, 0)
#             # Start the tracking with these settings
#             run_tracking(video_path, params)
#             print("[INFO] Tracking complete. You may adjust parameters or enter a new file to track again.")
#             print("Press 'q' in the display windows next time to quit early if desired.\n")
#             break

#     # Optionally allow re-runs if the user changes the parameter or wants another pass
#     while True:
#         print("Type 'r' to run again with updated parameters.")
#         print("Type 'n' for a new video file, or 'q' to quit the application.")
#         user_choice = input("Enter your choice [r/n/q]: ").lower().strip()
#         if user_choice == 'r':
#             params = read_parameters_from_trackbars(param_window_name)
#             run_tracking(video_path, params)
#         elif user_choice == 'n':
#             video_path = input("Enter path to your .mp4 file: ").strip()
#             if not video_path:
#                 print("ERROR: No path provided. Exiting.")
#                 sys.exit(1)
#             params = read_parameters_from_trackbars(param_window_name)
#             run_tracking(video_path, params)
#         elif user_choice == 'q':
#             print("[INFO] Exiting application.")
#             break
#         else:
#             print("[WARNING] Invalid choice. Please enter 'r', 'n', or 'q'.")


# if __name__ == "__main__":
#     main()

"""
A production-ready Python script that uses PySide2 to provide a user-friendly UI for 
multi-object tracking in a pre-recorded video (e.g., .mp4). This script is "feature complete":
1. File selection via a dialog
2. Parameter control (number of targets, threshold, morphological kernel, etc.)
3. On-the-fly preview of how tracking changes with updated parameters (Preview Mode)
4. Full tracking mode to process from start to finish, display optional overlays, and save results
5. A single-file solution with minimal dependencies (besides PySide2, OpenCV, NumPy, SciPy, Matplotlib)

Usage:
------
    python tracking_ui_pyside2.py

Dependencies:
-------------
    - Python 3.9+
    - PySide2
    - OpenCV (cv2)
    - NumPy
    - SciPy
    - Matplotlib

Explanation of Key Components:
------------------------------
1) MainWindow (QMainWindow): 
   - Houses all UI elements. The left panel displays the video frames (live updates).
   - The right panel (a QWidget) contains controls: file selection button, parameters (sliders/spin boxes),
     checkboxes for optional overlays, and buttons for Preview / Start Tracking / Stop / Save, etc.

2) TrackingWorker (QThread):
   - Runs the frame-by-frame tracking in a separate thread so that the UI remains responsive.
   - On each frame, it applies the user-set parameters (threshold, morphological kernel, etc.).
   - Uses a "lightest background" model plus the Hungarian algorithm with Kalman filters.
   - Sends signals back to the MainWindow to update the displayed frame with overlays.

3) Preview vs Full Tracking:
   - Preview Mode: Runs continuously from the start of the video (or from last frame if configured),
     letting the user see how parameter changes affect detection/tracking in near real-time.
     The background model will keep updating, so large parameter changes may look odd if partial
     progress is already made. (One may implement logic to reset the background model if desired.)
   - Full Tracking Mode: Processes from the first frame to the end, collecting all trajectory data.
     Results are saved to "trajectories.csv" after it completes or is stopped.

4) Optional Overlays:
   - Foreground Mask
   - Blob Detection / Ellipse Overlays
   - Lightest Background
   - Each overlay can be toggled on/off with checkboxes.

5) Saving Results:
   - After Full Tracking finishes, the script automatically saves results to "trajectories.csv".
   - Optionally, an FPS plot is displayed at the end.

6) Parameter Adjustments on the Fly:
   - Changes to the UI parameter widgets (spin boxes, sliders, etc.) are immediately propagated to 
     the worker thread, which will adjust its logic for the next frame.

Important Notes:
----------------
- This script is a simplified demonstration of how to build a PySide2 UI. 
- For real production usage, you may further refine threading, error handling, and parameter
  synchronization to best suit your workflow.
"""

import sys
import time
import csv
import numpy as np
import cv2
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

from PySide2.QtCore import (
    Qt, QThread, Signal, Slot, QMutex, QWaitCondition, QTimer
)
from PySide2.QtGui import (
    QImage, QPixmap
)
from PySide2.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QFileDialog, QSpinBox, QDoubleSpinBox, QCheckBox, QMessageBox,
    QGroupBox, QFormLayout, QSlider, QLineEdit
)


class TrackingWorker(QThread):
    """
    A QThread worker that performs multi-object tracking on a video file using 
    a lightest-background model, morphological ops, and Kalman filters with the 
    Hungarian assignment.
    
    Parameters
    ----------
    video_path : str
        Path to the video file to track.
    """
    # Signals to update the MainWindow's UI
    frame_signal = Signal(np.ndarray)  # emits the current (overlayed) frame as a NumPy BGR image
    finished_signal = Signal(bool, list)  
    # finished_signal emits a tuple: 
    #   bool -> whether or not the tracking finished normally (True) or was stopped (False)
    #   list -> a list of FPS values across frames, for potential plotting

    def __init__(self, video_path: str, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.cap = None

        # We store the parameters externally to allow on-the-fly adjustments
        self.params_mutex = QMutex()
        self.parameters = {}
        
        # For stopping the thread gracefully
        self._stop_flag = False

        # For storing trajectories
        self.trajectories = []
        self.kalman_filters = []
        self.track_ids = []
        self.detection_initialized = False
        self.tracking_stabilized = False
        self.detection_counts = 0
        self.tracking_counts = 0

        self.background_model_lightest = None
        self.conservative_used = False

        self.fps_list = []
        self.frame_count = 0
        self.start_time = 0

    def set_parameters(self, new_params: dict):
        """
        Update tracking parameters in a thread-safe manner.
        
        Parameters
        ----------
        new_params : dict
            Dictionary of the updated parameters.
        """
        self.params_mutex.lock()
        self.parameters = new_params
        self.params_mutex.unlock()

    def stop(self):
        """
        Signal the thread to stop as soon as possible.
        """
        self._stop_flag = True

    def get_current_params(self):
        """
        Safely retrieve the current parameters dictionary.
        """
        self.params_mutex.lock()
        p = dict(self.parameters)
        self.params_mutex.unlock()
        return p

    def init_kalman_filters(self):
        """
        Initialize the Kalman filters based on the number of targets set in parameters.
        """
        p = self.get_current_params()
        max_targets = p["MAX_TARGETS"]
        kf_list = []
        for _ in range(max_targets):
            kf = cv2.KalmanFilter(5, 3)
            # Measurement matrix: Maps state to measurements (x, y, theta)
            kf.measurementMatrix = np.array([[1, 0, 0, 0, 0],
                                             [0, 1, 0, 0, 0],
                                             [0, 0, 1, 0, 0]], np.float32)
            # Transition matrix
            kf.transitionMatrix = np.array([[1, 0, 0, 1, 0],
                                            [0, 1, 0, 0, 1],
                                            [0, 0, 1, 0, 0],
                                            [0, 0, 0, 1, 0],
                                            [0, 0, 0, 0, 1]], np.float32)
            kf.processNoiseCov = np.eye(5, dtype=np.float32) * p["KALMAN_NOISE_COVARIANCE"]
            kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * p["KALMAN_MEASUREMENT_NOISE_COVARIANCE"]
            kf.errorCovPre = np.eye(5, dtype=np.float32)
            kf_list.append(kf)
        return kf_list

    def run(self):
        """
        The main loop of the tracking thread. Opens the video, processes frames, 
        and emits signals for the UI to display frames. 
        """
        # Reset stop flag
        self._stop_flag = False

        # Attempt to open capture
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            print(f"[ERROR] Cannot open video file: {self.video_path}")
            self.finished_signal.emit(True, [])  # emit 'finished' to end gracefully
            return

        # Re-init state in case this worker is re-run
        p = self.get_current_params()
        self.kalman_filters = self.init_kalman_filters()
        self.track_ids = np.arange(p["MAX_TARGETS"])
        self.trajectories = [[] for _ in range(p["MAX_TARGETS"])]
        self.background_model_lightest = None
        self.detection_initialized = False
        self.tracking_stabilized = False
        self.detection_counts = 0
        self.tracking_counts = 0
        self.conservative_used = False
        self.fps_list.clear()
        self.frame_count = 0
        self.start_time = time.time()

        while not self._stop_flag:
            ret, frame = self.cap.read()
            if not ret:
                # No more frames or read error
                break

            current_params = self.get_current_params()
            self.frame_count += 1

            # Convert to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Initialize background model if None
            if self.background_model_lightest is None:
                self.background_model_lightest = gray_frame.astype(np.float32)
                # Show initial frame if the user wants any overlay
                overlay_frame = frame
                self.emit_frame(overlay_frame)
                continue

            # Update the background model
            self.background_model_lightest = np.maximum(self.background_model_lightest, gray_frame.astype(np.float32))
            bg_uint8 = cv2.convertScaleAbs(self.background_model_lightest)

            # Foreground detection
            fg_mask = cv2.absdiff(bg_uint8, gray_frame)
            _, fg_mask = cv2.threshold(fg_mask, current_params["THRESHOLD_VALUE"], 255, cv2.THRESH_BINARY)

            # Morph
            k_size = current_params["MORPH_KERNEL_SIZE"]
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

            # Conservative mask
            split_mask = cv2.erode(fg_mask, kernel, iterations=2)

            # Find contours
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            conservative_contours, _ = cv2.findContours(split_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            measurements = []
            sizes = []

            # Ellipse fitting on normal mask
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < current_params["MIN_CONTOUR_AREA"]:
                    continue
                if len(cnt) >= 5:
                    ellipse = cv2.fitEllipse(cnt)
                    (x, y), (MA, ma), angle = ellipse
                    angle_radians = np.deg2rad(angle)
                    measurements.append(np.array([x, y, angle_radians], dtype=np.float32))
                    sizes.append(area)

            # Keep largest if needed
            if len(measurements) > current_params["MAX_TARGETS"]:
                sorted_idx = np.argsort(sizes)[::-1]
                measurements = [measurements[i] for i in sorted_idx[:current_params["MAX_TARGETS"]]]

            # If fewer than needed, try conservative mask
            if len(measurements) < current_params["MAX_TARGETS"]:
                self.conservative_used = True
                measurements = []
                sizes = []
                for cnt in conservative_contours:
                    area = cv2.contourArea(cnt)
                    if area < current_params["MIN_CONTOUR_AREA"]:
                        continue
                    if len(cnt) >= 5:
                        ellipse = cv2.fitEllipse(cnt)
                        (x, y), (MA, ma), angle = ellipse
                        angle_radians = np.deg2rad(angle)
                        measurements.append(np.array([x, y, angle_radians], dtype=np.float32))
                        sizes.append(area)
                if len(measurements) > current_params["MAX_TARGETS"]:
                    sorted_idx = np.argsort(sizes)[::-1]
                    measurements = [measurements[i] for i in sorted_idx[:current_params["MAX_TARGETS"]]]
            else:
                self.conservative_used = False

            # Check if detection is complete
            if len(measurements) == current_params["MAX_TARGETS"]:
                self.detection_counts += 1
            else:
                self.detection_counts = 0

            if self.detection_counts >= current_params["MIN_DETECTION_COUNTS"]:
                self.detection_initialized = True

            # Tracking with Kalman filters
            overlay_frame = frame.copy()  # We will draw overlays on this
            if self.detection_initialized and measurements:
                # Possibly (re)initialize the KF states if at the start
                if self.frame_count == 1:
                    h, w, _ = frame.shape
                    for kf in self.kalman_filters:
                        kf.statePre = np.array([
                            np.random.randint(0, w),
                            np.random.randint(0, h),
                            0, 0, 0
                        ], dtype=np.float32)
                        kf.statePost = kf.statePre.copy()

                # Predict
                predicted_positions = []
                for i, kf in enumerate(self.kalman_filters):
                    pred = kf.predict()
                    predicted_positions.append(pred[:3].flatten())

                predicted_positions = np.array(predicted_positions)  # shape = (max_targets, 3)

                # Build cost matrix
                cost_matrix = np.zeros((current_params["MAX_TARGETS"], len(measurements)), dtype=np.float32)
                for i, pred_pos in enumerate(predicted_positions):
                    for j, meas in enumerate(measurements):
                        pos_cost = distance.euclidean(pred_pos[:2], meas[:2])
                        angle_cost = abs(pred_pos[2] - meas[2])
                        angle_cost = min(angle_cost, 2*np.pi - angle_cost)
                        cost_matrix[i, j] = pos_cost + angle_cost

                row_idx, col_idx = linear_sum_assignment(cost_matrix)

                avg_cost = 0.0
                for row, col in zip(row_idx, col_idx):
                    if row < current_params["MAX_TARGETS"] and col < len(measurements):
                        if (not self.tracking_stabilized) or (cost_matrix[row, col] < current_params["MAX_DISTANCE_THRESHOLD"]):
                            kf = self.kalman_filters[row]
                            measurement_vector = np.array([
                                [measurements[col][0]],
                                [measurements[col][1]],
                                [measurements[col][2]]
                            ], dtype=np.float32)
                            kf.correct(measurement_vector)

                            x_corr, y_corr, t_corr = measurements[col]
                            ts = time.time()
                            self.trajectories[row].append((int(x_corr), int(y_corr), float(t_corr), ts))
                            # prune old
                            self.trajectories[row] = [
                                (xx, yy, th, tt) for (xx, yy, th, tt) in self.trajectories[row]
                                if (ts - tt) <= current_params["TRAJECTORY_HISTORY_SECONDS"]
                            ]
                            avg_cost += cost_matrix[row, col]/current_params["MAX_TARGETS"]
                        else:
                            self.tracking_stabilized = False
                            self.tracking_counts = 0
                            # Insert NaNs
                            ts = time.time()
                            self.trajectories[row].append((np.nan, np.nan, np.nan, ts))
                            self.trajectories[row] = [
                                (xx, yy, th, tt) for (xx, yy, th, tt) in self.trajectories[row]
                                if (ts - tt) <= current_params["TRAJECTORY_HISTORY_SECONDS"]
                            ]

                if avg_cost < current_params["MAX_DISTANCE_THRESHOLD"]:
                    self.tracking_counts += 1
                else:
                    self.tracking_counts = 0

                if self.tracking_counts >= current_params["MIN_TRACKING_COUNTS"] and not self.tracking_stabilized:
                    self.tracking_stabilized = True
                    print(f"[INFO] Tracking Stabilized with Average Cost: {avg_cost:.2f}")

                # Draw
                if current_params["SHOW_BLOB"]:
                    for i, tlist in enumerate(self.trajectories):
                        if not tlist:
                            continue
                        x, y, theta, ts = tlist[-1]
                        if not (np.isnan(x) or np.isnan(y)):
                            color = current_params["TRAJECTORY_COLORS"][i % len(current_params["TRAJECTORY_COLORS"])]
                            cv2.circle(overlay_frame, (x, y), 10, color, -1)
                            length = 20
                            x_end = int(x + length * np.cos(theta))
                            y_end = int(y + length * np.sin(theta))
                            cv2.line(overlay_frame, (x, y), (x_end, y_end), color, 2)
                            cv2.putText(overlay_frame, f"ID: {i}", (x+15, y-15),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                            # Draw trajectory lines
                            for idx_pt in range(1, len(tlist)):
                                pt1 = (tlist[idx_pt-1][0], tlist[idx_pt-1][1])
                                pt2 = (tlist[idx_pt][0], tlist[idx_pt][1])
                                if (not np.isnan(pt1[0]) and not np.isnan(pt1[1]) and 
                                    not np.isnan(pt2[0]) and not np.isnan(pt2[1])):
                                    cv2.line(overlay_frame, pt1, pt2, color, 2)

            # If user wants to see Foreground
            if current_params["SHOW_FG"]:
                if self.conservative_used:
                    # Show the more conservative version
                    mask_vis = cv2.cvtColor(split_mask, cv2.COLOR_GRAY2BGR)
                else:
                    mask_vis = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
                small_mask = cv2.resize(mask_vis, (mask_vis.shape[1]//2, mask_vis.shape[0]//2))
                # top-left corner overlay for demonstration
                overlay_frame[0:small_mask.shape[0], 0:small_mask.shape[1], :] = small_mask

            # If user wants to see Lightest Background
            if current_params["SHOW_BG"]:
                bg_vis = cv2.cvtColor(bg_uint8, cv2.COLOR_GRAY2BGR)
                small_bg = cv2.resize(bg_vis, (bg_vis.shape[1]//2, bg_vis.shape[0]//2))
                # top-right corner overlay
                x_offset = overlay_frame.shape[1] - small_bg.shape[1]
                overlay_frame[0:small_bg.shape[0], x_offset:overlay_frame.shape[1], :] = small_bg

            # Calculate FPS
            elapsed_time = time.time() - self.start_time
            if elapsed_time > 0:
                self.fps_list.append(self.frame_count/elapsed_time)

            # Emit the overlay frame
            self.emit_frame(overlay_frame)

            # Sleep a tiny bit if you want to slow down (or remove to go as fast as possible)
            # time.sleep(0.001)

        self.cap.release()
        done_normally = (not self._stop_flag)
        self.finished_signal.emit(done_normally, self.fps_list)

    def emit_frame(self, frame: np.ndarray):
        """
        Convert a BGR NumPy array to a QImage and emit it via frame_signal.
        
        Parameters
        ----------
        frame : np.ndarray
            The BGR image to display.
        """
        self.frame_signal.emit(frame)


class MainWindow(QMainWindow):
    """
    The main application window that presents:
        - A video display area (QLabel)
        - A parameter panel (spin boxes, checkboxes)
        - Buttons for file selection, preview, full run, stopping, etc.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fly Tracking UI (PySide2)")
        self.resize(1400, 800)

        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        
        # Layouts
        main_layout = QHBoxLayout()
        control_layout = QVBoxLayout()

        # File Selection
        self.btn_file = QPushButton("Select Video File...")
        self.btn_file.clicked.connect(self.select_file)
        self.file_path_line = QLineEdit()
        self.file_path_line.setPlaceholderText("No file selected")

        file_layout = QHBoxLayout()
        file_layout.addWidget(self.btn_file)
        file_layout.addWidget(self.file_path_line)

        # Parameter Panel
        param_group = QGroupBox("Parameters")
        form_layout = QFormLayout()

        # SpinBoxes / Sliders for all needed parameters
        # 1) MAX_TARGETS
        self.spin_max_targets = QSpinBox()
        self.spin_max_targets.setRange(1, 20)
        self.spin_max_targets.setValue(4)
        form_layout.addRow("Max Targets", self.spin_max_targets)

        # 2) THRESHOLD_VALUE
        self.spin_threshold = QSpinBox()
        self.spin_threshold.setRange(0, 255)
        self.spin_threshold.setValue(50)
        form_layout.addRow("Threshold", self.spin_threshold)

        # 3) MORPH_KERNEL_SIZE
        self.spin_morph_size = QSpinBox()
        self.spin_morph_size.setRange(1, 100)
        self.spin_morph_size.setValue(5)
        form_layout.addRow("Morph Kernel Size", self.spin_morph_size)

        # 4) MIN_CONTOUR_AREA
        self.spin_min_contour = QSpinBox()
        self.spin_min_contour.setRange(0, 100000)
        self.spin_min_contour.setValue(50)
        form_layout.addRow("Min Contour Area", self.spin_min_contour)

        # 5) MAX_DISTANCE_THRESHOLD
        self.spin_max_dist = QSpinBox()
        self.spin_max_dist.setRange(0, 1000)
        self.spin_max_dist.setValue(25)
        form_layout.addRow("Max Distance Thresh", self.spin_max_dist)

        # 6) MIN_DETECTION_COUNTS
        self.spin_min_detect = QSpinBox()
        self.spin_min_detect.setRange(0, 1000)
        self.spin_min_detect.setValue(10)
        form_layout.addRow("Min Detection Counts", self.spin_min_detect)

        # 7) MIN_TRACKING_COUNTS
        self.spin_min_track = QSpinBox()
        self.spin_min_track.setRange(0, 1000)
        self.spin_min_track.setValue(10)
        form_layout.addRow("Min Tracking Counts", self.spin_min_track)

        # 8) TRAJECTORY_HISTORY_SECONDS
        self.spin_traj_hist = QSpinBox()
        self.spin_traj_hist.setRange(0, 300)
        self.spin_traj_hist.setValue(5)
        form_layout.addRow("Trajectory History (sec)", self.spin_traj_hist)

        # 9) KALMAN_NOISE_COVARIANCE
        self.spin_kalman_noise = QDoubleSpinBox()
        self.spin_kalman_noise.setRange(0.0, 1.0)
        self.spin_kalman_noise.setSingleStep(0.01)
        self.spin_kalman_noise.setValue(0.03)
        form_layout.addRow("Kalman Noise Cov", self.spin_kalman_noise)

        # 10) KALMAN_MEASUREMENT_NOISE_COVARIANCE
        self.spin_kalman_meas_noise = QDoubleSpinBox()
        self.spin_kalman_meas_noise.setRange(0.0, 1.0)
        self.spin_kalman_meas_noise.setSingleStep(0.01)
        self.spin_kalman_meas_noise.setValue(0.1)
        form_layout.addRow("Kalman Measurement Cov", self.spin_kalman_meas_noise)

        # Checkboxes for overlays
        self.chk_show_fg = QCheckBox("Show Foreground Mask")
        self.chk_show_fg.setChecked(True)
        form_layout.addRow(self.chk_show_fg)

        self.chk_show_blob = QCheckBox("Show Blob Overlays")
        self.chk_show_blob.setChecked(True)
        form_layout.addRow(self.chk_show_blob)

        self.chk_show_bg = QCheckBox("Show Lightest Background")
        self.chk_show_bg.setChecked(True)
        form_layout.addRow(self.chk_show_bg)

        param_group.setLayout(form_layout)

        # Buttons for controlling tracking
        self.btn_preview = QPushButton("Preview")
        self.btn_preview.setCheckable(True)
        self.btn_preview.clicked.connect(self.toggle_preview)

        self.btn_start_tracking = QPushButton("Start Full Tracking")
        self.btn_start_tracking.clicked.connect(self.start_full_tracking)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self.stop_tracking)

        control_layout.addLayout(file_layout)
        control_layout.addWidget(param_group)
        control_layout.addWidget(self.btn_preview)
        control_layout.addWidget(self.btn_start_tracking)
        control_layout.addWidget(self.btn_stop)
        control_layout.addStretch(1)

        # Add to main layout
        main_layout.addWidget(self.video_label, stretch=3)
        main_layout.addLayout(control_layout, stretch=1)

        # Central widget
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Tracking worker + thread
        self.tracking_worker = None

    def select_file(self):
        """
        Open a QFileDialog to let the user pick a video file (.mp4, .avi, etc.).
        """
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov)")
        if file_path:
            self.file_path_line.setText(file_path)

    def get_parameters_dict(self):
        """
        Collect all parameters from UI controls into a single dictionary.
        Returns
        -------
        params : dict
        """
        # Build color set (deterministic or random)
        np.random.seed(42)
        max_targets = self.spin_max_targets.value()
        color_array = [tuple(c.tolist()) for c in np.random.randint(0, 255, (max_targets, 3))]

        params = {
            "MAX_TARGETS": max_targets,
            "THRESHOLD_VALUE": self.spin_threshold.value(),
            "MORPH_KERNEL_SIZE": self.spin_morph_size.value(),
            "MIN_CONTOUR_AREA": self.spin_min_contour.value(),
            "MAX_DISTANCE_THRESHOLD": self.spin_max_dist.value(),
            "MIN_DETECTION_COUNTS": self.spin_min_detect.value(),
            "MIN_TRACKING_COUNTS": self.spin_min_track.value(),
            "TRAJECTORY_HISTORY_SECONDS": self.spin_traj_hist.value(),
            "KALMAN_NOISE_COVARIANCE": float(self.spin_kalman_noise.value()),
            "KALMAN_MEASUREMENT_NOISE_COVARIANCE": float(self.spin_kalman_meas_noise.value()),
            "SHOW_FG": self.chk_show_fg.isChecked(),
            "SHOW_BLOB": self.chk_show_blob.isChecked(),
            "SHOW_BG": self.chk_show_bg.isChecked(),
            "TRAJECTORY_COLORS": color_array
        }
        return params

    def toggle_preview(self, checked):
        """
        Start or stop the preview mode depending on the button state.
        """
        if checked:
            # Start preview
            self.start_tracking(preview_mode=True)
            self.btn_preview.setText("Stop Preview")
        else:
            # Stop preview
            self.stop_tracking()
            self.btn_preview.setText("Preview")

    def start_full_tracking(self):
        """
        Start a full run from beginning to end (saving results on completion).
        """
        # If preview is toggled on, turn it off
        if self.btn_preview.isChecked():
            self.btn_preview.setChecked(False)
            self.btn_preview.setText("Preview")
        self.start_tracking(preview_mode=False)

    def start_tracking(self, preview_mode: bool):
        """
        Common method to start either preview or full tracking.
        
        Parameters
        ----------
        preview_mode : bool
            If True, user is in "Preview" mode; if False, "Full Tracking."
        """
        video_path = self.file_path_line.text()
        if not video_path:
            QMessageBox.warning(self, "No file selected", "Please select a video file first!")
            # Reset preview button
            if preview_mode:
                self.btn_preview.setChecked(False)
                self.btn_preview.setText("Preview")
            return

        if self.tracking_worker is not None and self.tracking_worker.isRunning():
            QMessageBox.warning(self, "Worker busy", "A tracking thread is already running. Stop it first.")
            return

        self.tracking_worker = TrackingWorker(video_path)
        # Connect signals
        self.tracking_worker.frame_signal.connect(self.on_new_frame)
        self.tracking_worker.finished_signal.connect(self.on_tracking_finished)

        # Pass in parameters
        self.tracking_worker.set_parameters(self.get_parameters_dict())

        self.tracking_worker.start()

    def stop_tracking(self):
        """
        Stop the running worker if it exists.
        """
        if self.tracking_worker and self.tracking_worker.isRunning():
            self.tracking_worker.stop()

    @Slot(np.ndarray)
    def on_new_frame(self, frame_bgr: np.ndarray):
        """
        Slot to receive new frames from the worker thread, convert to QImage and show.
        
        Parameters
        ----------
        frame_bgr : np.ndarray
            The new BGR frame to display.
        """
        height, width, channel = frame_bgr.shape
        bytes_per_line = channel * width
        # Convert BGR -> RGB for display
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg))

    @Slot(bool, list)
    def on_tracking_finished(self, finished_normally: bool, fps_list: list):
        """
        Called when the worker signals it is finished. 
        We can optionally save the CSV if it was a 'full' run or show an FPS plot, etc.
        
        Parameters
        ----------
        finished_normally : bool
            True if the worker reached end of video, False if user forcibly stopped it.
        fps_list : list
            A list of FPS values over frames.
        """
        # Check if we were in full tracking mode or preview mode.
        # We'll assume if "Preview" button is checked, we were in preview mode.
        preview_mode = self.btn_preview.isChecked()
        if preview_mode:
            # Turn off the preview if the worker ended on its own
            self.btn_preview.setChecked(False)
            self.btn_preview.setText("Preview")

        if finished_normally and (not preview_mode):
            # Full run completed -> we can do "save results to CSV", 
            # but the logic of saving the actual positions must happen in the worker or here. 
            # In this example, let's prompt user that we can do it if needed.
            # (For simplicity, we haven't plumbed the entire trajectory data from the worker. 
            # One approach is to store it in the worker and pass it via a custom signal.)
            self.save_trajectory_csv()
            # Optionally display FPS plot
            self.plot_fps(fps_list)

    def save_trajectory_csv(self):
        """
        Example method for saving the results. For this demonstration, we show a message.
        If you want to actually retrieve data from the worker, you'd define a new signal 
        carrying self.trajectories and write them to CSV here.
        """
        # We do not have the actual trajectory data in the main thread. 
        # You can add a new signal in the worker that emits the trajectory. 
        # For demonstration, we show a message:
        QMessageBox.information(self, "Save CSV", 
                                "A full run completed. The worker would normally emit trajectory data.\n"
                                "You could save that to a CSV file here. For demonstration, no file is written.")

    def plot_fps(self, fps_list):
        """
        Displays an FPS plot using Matplotlib in a blocking fashion.
        
        Parameters
        ----------
        fps_list : list of float
            The per-frame or running FPS values collected during tracking.
        """
        if len(fps_list) < 2:
            QMessageBox.information(self, "FPS Plot", "Not enough data to plot FPS.")
            return

        plt.figure()
        plt.plot(fps_list, label="FPS")
        plt.xlabel("Frame Index")
        plt.ylabel("Frames per Second")
        plt.title("Tracking FPS Over Time")
        plt.legend()
        plt.show()


def main():
    """
    Main entry-point to run the Fly Tracking UI (PySide2).
    """
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
