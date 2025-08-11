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

# """
# Full Production-Ready PySide2 Script for Fly/Animal Multi-Object Tracking with:
# 1) Local Watershed-Based Merge Splitting
# 2) Brightness/Contrast/Gamma Adjustments for Preprocessing
# 3) Kalman + Hungarian Multi-Object Tracking
# 4) Memory-Safe Practices (Explicit Cleanup, GC)

# Features:
# ---------
# - Liberal mask generation (light morphological open/close).
# - Local ROI refinement using watershed for suspicious merges.
# - Brightness, Contrast, Gamma adjustments to improve poor video quality.
# - User-friendly PySide2 UI with trackbars/spinboxes:
#    * Threshold, morphological kernel size, min contour area, etc.
#    * Merge area threshold to identify suspicious merges.
#    * Brightness, Contrast, Gamma controls to tune the image on the fly.
# - Background model: “lightest pixel” approach (one pass, not adaptive).
# - Memory safety: explicit object deletion + gc.collect() after each frame.
# - Optional real-time “Preview” mode or full-run “Tracking” mode.
# - Resizable frames to reduce memory usage if needed.
# - Basic visualization of Foreground Mask & Lightest Background in corners.

# Setup:
# ------
# To use:
# 1) Install dependencies (PySide2, OpenCV, NumPy, SciPy, matplotlib).
# 2) Run the script: python tracking_local_watershed.py
# 3) Click “Select Video File...” to pick a .mp4 or .avi video.
# 4) Adjust parameters on the right panel.
# 5) “Preview” to see a live preview (stop with the same button).
# 6) “Full Tracking” to process the entire video. Press “Stop” to cancel.

# Additional Notes:
# ----------------
# - The user can implement CSV saving of final trajectories by hooking
#   the data from the worker's Kalman filters and writing them out in
#   on_tracking_finished if needed. Currently, we only display a message.
# - If you observe that sometimes the entire bounding box is used for a merge,
#   it may be that your threshold or background model is marking most of the
#   frame as foreground. Adjust brightness/contrast/gamma or reset the background
#   logic as needed. Also ensure your MERGE_AREA_THRESHOLD is not set too low.
# """

# import sys
# import time
# import gc
# import numpy as np
# import cv2
# from scipy.spatial import distance
# from scipy.optimize import linear_sum_assignment
# import matplotlib.pyplot as plt

# from PySide2.QtCore import (
#     Qt, QThread, Signal, Slot, QMutex
# )
# from PySide2.QtGui import (
#     QImage, QPixmap
# )
# from PySide2.QtWidgets import (
#     QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout,
#     QHBoxLayout, QFileDialog, QSpinBox, QDoubleSpinBox, QCheckBox, QMessageBox,
#     QGroupBox, QFormLayout, QLineEdit
# )


# def apply_image_adjustments(
#     gray_frame: np.ndarray,
#     brightness: float,
#     contrast: float,
#     gamma: float
# ) -> np.ndarray:
#     """
#     Apply brightness, contrast, and gamma corrections to a grayscale image.

#     Parameters
#     ----------
#     gray_frame : np.ndarray
#         8-bit grayscale image (shape: H x W).
#     brightness : float
#         Value added to each pixel after contrast scaling. Range e.g. [-100..100].
#     contrast : float
#         Scaling factor for pixel values. e.g. 1.0 means no change, 1.2 means 20% higher contrast.
#     gamma : float
#         Gamma correction factor. 1.0 means no change, <1.0 darkens midtones, >1.0 brightens.

#     Returns
#     -------
#     adjusted : np.ndarray
#         The adjusted grayscale image, uint8.
#     """
#     # 1) Brightness & Contrast using OpenCV convertScaleAbs
#     #    new_pixel = alpha * old_pixel + beta
#     adjusted = cv2.convertScaleAbs(gray_frame, alpha=contrast, beta=brightness)

#     # 2) Gamma Correction
#     #    pixel_out = 255 * ((pixel_in / 255)^(1/gamma))
#     if abs(gamma - 1.0) > 1e-3:
#         look_up_table = np.array([
#             np.clip(((i / 255.0) ** (1.0 / gamma)) * 255.0, 0, 255)
#             for i in range(256)
#         ], dtype=np.uint8)
#         adjusted = cv2.LUT(adjusted, look_up_table)

#     return adjusted


# class TrackingWorker(QThread):
#     """
#     A QThread worker that performs multi-object tracking on a video file using:
#     1) Brightness/Contrast/Gamma corrections
#     2) Lightest-background approach
#     3) Local watershed-based refinement of suspicious merges
#     4) Kalman filters + Hungarian assignment

#     Signals:
#     --------
#     frame_signal: emits an RGB frame (numpy array) for display
#     finished_signal: emits a tuple (finished_normally: bool, fps_list: list)
#     """
#     frame_signal = Signal(np.ndarray)
#     finished_signal = Signal(bool, list)

#     def __init__(self, video_path: str, parent=None):
#         super().__init__(parent)
#         self.video_path = video_path

#         self.params_mutex = QMutex()
#         self.parameters = {}

#         self._stop_flag = False
#         self.cap = None

#         # Kalman/tracking structures
#         self.kalman_filters = []
#         self.track_ids = []
#         self.trajectories = []

#         # Background model
#         self.background_model_lightest = None

#         # Tracking flags
#         self.detection_initialized = False
#         self.tracking_stabilized = False
#         self.detection_counts = 0
#         self.tracking_counts = 0

#         # Others
#         self.fps_list = []
#         self.frame_count = 0
#         self.start_time = 0

#     def set_parameters(self, new_params: dict):
#         """ Safely update tracking parameters. """
#         self.params_mutex.lock()
#         self.parameters = new_params
#         self.params_mutex.unlock()

#     def get_current_params(self):
#         """ Return a copy of the parameters. """
#         self.params_mutex.lock()
#         p = dict(self.parameters)
#         self.params_mutex.unlock()
#         return p

#     def stop(self):
#         """ Signal this worker to stop. """
#         self._stop_flag = True

#     def init_kalman_filters(self, p):
#         """
#         Initialize Kalman filters based on number of targets in p.
#         """
#         kf_list = []
#         for _ in range(p["MAX_TARGETS"]):
#             kf = cv2.KalmanFilter(5, 3)
#             # Measurement: x, y, theta
#             kf.measurementMatrix = np.array([
#                 [1, 0, 0, 0, 0],
#                 [0, 1, 0, 0, 0],
#                 [0, 0, 1, 0, 0]
#             ], np.float32)

#             # Transition: x->x+vx, y->y+vy, theta->theta
#             kf.transitionMatrix = np.array([
#                 [1, 0, 0, 1, 0],
#                 [0, 1, 0, 0, 1],
#                 [0, 0, 1, 0, 0],
#                 [0, 0, 0, 1, 0],
#                 [0, 0, 0, 0, 1]
#             ], np.float32)

#             kf.processNoiseCov = np.eye(5, dtype=np.float32) * p["KALMAN_NOISE_COVARIANCE"]
#             kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * p["KALMAN_MEASUREMENT_NOISE_COVARIANCE"]
#             kf.errorCovPre = np.eye(5, dtype=np.float32)
#             kf_list.append(kf)
#         return kf_list

#     def run(self):
#         """ Main loop for reading frames, segmenting, tracking, and signal emission. """
#         gc.collect()
#         self._stop_flag = False
#         p = self.get_current_params()

#         # Attempt to open capture
#         self.cap = cv2.VideoCapture(self.video_path)
#         if not self.cap.isOpened():
#             print(f"[ERROR] Cannot open video file: {self.video_path}")
#             self.finished_signal.emit(True, [])
#             return

#         # Re-initialize
#         self.background_model_lightest = None
#         self.kalman_filters = self.init_kalman_filters(p)
#         self.track_ids = np.arange(p["MAX_TARGETS"])
#         self.trajectories = [[] for _ in range(p["MAX_TARGETS"])]

#         self.detection_initialized = False
#         self.tracking_stabilized = False
#         self.detection_counts = 0
#         self.tracking_counts = 0

#         self.fps_list.clear()
#         self.frame_count = 0
#         self.start_time = time.time()

#         try:
#             while not self._stop_flag:
#                 ret, frame = self.cap.read()
#                 if not ret:
#                     break

#                 self.frame_count += 1

#                 # Optionally resize frame
#                 if p.get("RESIZE_FACTOR", 1.0) < 1.0:
#                     rsz = p["RESIZE_FACTOR"]
#                     frame = cv2.resize(frame,
#                                        (int(frame.shape[1]*rsz), int(frame.shape[0]*rsz)),
#                                        interpolation=cv2.INTER_AREA)

#                 # Convert to grayscale
#                 gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#                 # Apply brightness/contrast/gamma
#                 bright = p.get("BRIGHTNESS", 0.0)
#                 contr = p.get("CONTRAST", 1.0)
#                 gamm  = p.get("GAMMA", 1.0)
#                 gray_frame = apply_image_adjustments(gray_frame, bright, contr, gamm)

#                 # Initialize background if needed
#                 if self.background_model_lightest is None:
#                     self.background_model_lightest = gray_frame.astype(np.float32)
#                     # Show the first frame
#                     self.emit_frame(frame)
#                     self._cleanup_locals(frame, gray_frame)
#                     continue

#                 # Update "lightest" background
#                 self.background_model_lightest = np.maximum(
#                     self.background_model_lightest, gray_frame.astype(np.float32)
#                 )
#                 bg_uint8 = cv2.convertScaleAbs(self.background_model_lightest)

#                 # Create "liberal" mask
#                 fg_mask = cv2.absdiff(bg_uint8, gray_frame)
#                 _, fg_mask = cv2.threshold(fg_mask, p["THRESHOLD_VALUE"], 255, cv2.THRESH_BINARY)

#                 # Morphological open/close
#                 kernel = cv2.getStructuringElement(
#                     cv2.MORPH_ELLIPSE, (p["MORPH_KERNEL_SIZE"], p["MORPH_KERNEL_SIZE"])
#                 )
#                 fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
#                 fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

#                 # Copy to main_mask
#                 main_mask = fg_mask.copy()

#                 # Identify suspicious merges: large area or insufficient total contours
#                 contours, _ = cv2.findContours(main_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#                 merge_area_thr = p.get("MERGE_AREA_THRESHOLD", 1500)

#                 total_contours = sum(1 for cnt in contours if cv2.contourArea(cnt) > 0)
#                 suspicious_bboxes = []
#                 for cnt in contours:
#                     area = cv2.contourArea(cnt)
#                     if area < 1:
#                         continue
#                     x, y, w, h = cv2.boundingRect(cnt)
#                     if area > merge_area_thr or total_contours < p["MAX_TARGETS"]:
#                         suspicious_bboxes.append((x, y, w, h))

#                 # Local watershed splitting
#                 for (x, y, w, h) in suspicious_bboxes:
#                     sub_gray = gray_frame[y:y+h, x:x+w]
#                     sub_mask = main_mask[y:y+h, x:x+w]

#                     # skip if trivially small
#                     if np.count_nonzero(sub_mask) < 10:
#                         continue

#                     refined_mask = self._local_watershed_split(sub_gray, sub_mask, p)
#                     main_mask[y:y+h, x:x+w] = refined_mask

#                 # final contours for measurement
#                 final_contours, _ = cv2.findContours(main_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#                 measurements, sizes = [], []
#                 for cnt in final_contours:
#                     area = cv2.contourArea(cnt)
#                     if area < p["MIN_CONTOUR_AREA"]:
#                         continue
#                     if len(cnt) >= 5:
#                         ellipse = cv2.fitEllipse(cnt)
#                         (cx, cy), (MA, ma), angle = ellipse
#                         measurements.append(np.array([cx, cy, np.deg2rad(angle)], dtype=np.float32))
#                         sizes.append(area)

#                 # Keep largest up to MAX_TARGETS
#                 if len(measurements) > p["MAX_TARGETS"]:
#                     sorted_idx = np.argsort(sizes)[::-1]
#                     measurements = [measurements[i] for i in sorted_idx[:p["MAX_TARGETS"]]]

#                 # Check detection
#                 if len(measurements) == p["MAX_TARGETS"]:
#                     self.detection_counts += 1
#                 else:
#                     self.detection_counts = 0

#                 if self.detection_counts >= p["MIN_DETECTION_COUNTS"]:
#                     self.detection_initialized = True

#                 overlay_frame = frame.copy()
#                 if self.detection_initialized and measurements:
#                     # Possibly init KF states on first iteration
#                     if self.frame_count == 1:
#                         h_, w_ = gray_frame.shape
#                         for kf in self.kalman_filters:
#                             kf.statePre = np.array([
#                                 np.random.randint(0, w_),
#                                 np.random.randint(0, h_),
#                                 0, 0, 0
#                             ], dtype=np.float32)
#                             kf.statePost = kf.statePre.copy()

#                     # Predict
#                     predicted_positions = []
#                     for i, kf in enumerate(self.kalman_filters):
#                         pred = kf.predict()
#                         predicted_positions.append(pred[:3].flatten())
#                     predicted_positions = np.array(predicted_positions, dtype=np.float32)

#                     # Hungarian assignment
#                     cost_matrix = np.zeros((p["MAX_TARGETS"], len(measurements)), dtype=np.float32)
#                     for i, pred_pos in enumerate(predicted_positions):
#                         for j, meas in enumerate(measurements):
#                             pos_cost = distance.euclidean(pred_pos[:2], meas[:2])
#                             angle_diff = abs(pred_pos[2] - meas[2])
#                             angle_diff = min(angle_diff, 2*np.pi - angle_diff)
#                             cost_matrix[i, j] = pos_cost + angle_diff

#                     row_idx, col_idx = linear_sum_assignment(cost_matrix)

#                     avg_cost = 0.0
#                     for row, col in zip(row_idx, col_idx):
#                         if row < p["MAX_TARGETS"] and col < len(measurements):
#                             if (not self.tracking_stabilized) or (
#                                 cost_matrix[row, col] < p["MAX_DISTANCE_THRESHOLD"]
#                             ):
#                                 kf = self.kalman_filters[row]
#                                 measure_vec = np.array([
#                                     [measurements[col][0]],
#                                     [measurements[col][1]],
#                                     [measurements[col][2]]
#                                 ], dtype=np.float32)
#                                 kf.correct(measure_vec)

#                                 x_corr, y_corr, theta_corr = measurements[col]
#                                 ts = time.time()
#                                 self.trajectories[row].append(
#                                     (int(x_corr), int(y_corr), float(theta_corr), ts)
#                                 )
#                                 self._prune_traj(row, ts, p["TRAJECTORY_HISTORY_SECONDS"])
#                                 avg_cost += cost_matrix[row, col] / p["MAX_TARGETS"]
#                             else:
#                                 self.tracking_stabilized = False
#                                 self.tracking_counts = 0
#                                 ts = time.time()
#                                 self.trajectories[row].append((np.nan, np.nan, np.nan, ts))
#                                 self._prune_traj(row, ts, p["TRAJECTORY_HISTORY_SECONDS"])

#                     if avg_cost < p["MAX_DISTANCE_THRESHOLD"]:
#                         self.tracking_counts += 1
#                     else:
#                         self.tracking_counts = 0

#                     if self.tracking_counts >= p["MIN_TRACKING_COUNTS"] and not self.tracking_stabilized:
#                         self.tracking_stabilized = True
#                         print(f"[INFO] Tracking Stabilized (avg cost={avg_cost:.2f})")

#                     # Draw overlays
#                     if p["SHOW_BLOB"]:
#                         self._draw_overlays(overlay_frame, self.trajectories, p)

#                 # Show FG mask in corner
#                 if p["SHOW_FG"]:
#                     small = cv2.cvtColor(main_mask, cv2.COLOR_GRAY2BGR)
#                     scale = 0.3
#                     sw = int(small.shape[1]*scale)
#                     sh = int(small.shape[0]*scale)
#                     small = cv2.resize(small, (sw, sh), interpolation=cv2.INTER_AREA)
#                     overlay_frame[0:sh, 0:sw] = small

#                 # Show BG in corner
#                 if p["SHOW_BG"]:
#                     tmp_bg = cv2.cvtColor(bg_uint8, cv2.COLOR_GRAY2BGR)
#                     scale = 0.3
#                     sw = int(tmp_bg.shape[1]*scale)
#                     sh = int(tmp_bg.shape[0]*scale)
#                     small_bg = cv2.resize(tmp_bg, (sw, sh), interpolation=cv2.INTER_AREA)
#                     x_offset = overlay_frame.shape[1] - sw
#                     overlay_frame[0:sh, x_offset:overlay_frame.shape[1]] = small_bg

#                 # Compute FPS
#                 elapsed = time.time() - self.start_time
#                 if elapsed > 0:
#                     self.fps_list.append(self.frame_count / elapsed)

#                 self.emit_frame(overlay_frame)
#                 self._cleanup_locals(
#                     frame, gray_frame, bg_uint8, fg_mask, kernel, main_mask,
#                     overlay_frame, contours, final_contours
#                 )
#                 gc.collect()

#         except Exception as e:
#             print(f"[ERROR] Exception in worker: {e}")
#             self._stop_flag = True
#         finally:
#             if self.cap:
#                 self.cap.release()

#         done_normally = not self._stop_flag
#         self.finished_signal.emit(done_normally, self.fps_list)
#         gc.collect()

#     def _local_watershed_split(self, sub_gray: np.ndarray, sub_mask: np.ndarray, p: dict):
#         """
#         Perform a local watershed-based segmentation in the bounding box region
#         to split merged objects.

#         Steps:
#         1) Erode sub_mask slightly to separate close objects.
#         2) Distance transform the result.
#         3) Threshold the distance map to get peaks.
#         4) Convert those peaks to markers for watershed.
#         5) Apply cv2.watershed on a 3-channel sub region => new markers.
#         6) Build a refined mask from the marker labels (any label>0 => 1, boundary => 0).
#         """
#         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
#         local_mask = cv2.morphologyEx(sub_mask, cv2.MORPH_ERODE, kernel, iterations=1)

#         dist = cv2.distanceTransform(local_mask, cv2.DIST_L2, 3)
#         cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)

#         # Local maxima => peaks
#         _, peaks = cv2.threshold(dist, 0.3, 1.0, cv2.THRESH_BINARY)
#         peaks_8u = (peaks * 255).astype(np.uint8)

#         contours_peaks, _ = cv2.findContours(peaks_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         markers = np.zeros_like(sub_mask, dtype=np.int32)
#         for i, cnt in enumerate(contours_peaks, 1):
#             cv2.drawContours(markers, [cnt], -1, i, -1)

#         sub_bgr = cv2.cvtColor(sub_gray, cv2.COLOR_GRAY2BGR)
#         cv2.watershed(sub_bgr, markers)

#         refined = np.zeros_like(sub_mask)
#         refined[markers > 0] = 255
#         refined[markers == -1] = 0

#         del kernel, dist, peaks, peaks_8u, contours_peaks, markers, sub_bgr
#         return refined

#     def emit_frame(self, bgr_frame: np.ndarray):
#         """
#         Convert BGR to RGB and emit the frame signal.
#         """
#         rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
#         self.frame_signal.emit(rgb)

#     def _cleanup_locals(self, *args):
#         """Explicitly delete references to local large objects, then rely on gc.collect()."""
#         for obj in args:
#             del obj

#     def _prune_traj(self, idx, current_ts, history_sec):
#         """
#         Keep only the last 'history_sec' seconds of trajectory.
#         """
#         self.trajectories[idx] = [
#             (x, y, th, t) for (x, y, th, t) in self.trajectories[idx]
#             if (current_ts - t) <= history_sec
#         ]

#     def _draw_overlays(self, frame_bgr: np.ndarray, trajectories: list, p: dict):
#         """
#         Draw circles, orientation lines, and short trajectories on frame_bgr.
#         """
#         for i, tlist in enumerate(trajectories):
#             if not tlist:
#                 continue
#             x, y, theta, _ = tlist[-1]
#             if np.isnan(x) or np.isnan(y):
#                 continue

#             color = p["TRAJECTORY_COLORS"][i % len(p["TRAJECTORY_COLORS"])]
#             color_int = tuple(int(c) for c in color)

#             cv2.circle(frame_bgr, (x, y), 10, color_int, -1)
#             length = 20
#             x_end = int(x + length * np.cos(theta))
#             y_end = int(y + length * np.sin(theta))
#             cv2.line(frame_bgr, (x, y), (x_end, y_end), color_int, 2)
#             cv2.putText(frame_bgr, f"ID: {i}", (x+15, y-15),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_int, 2)

#             # Draw trajectory lines
#             for pt_i in range(1, len(tlist)):
#                 pt1 = (tlist[pt_i-1][0], tlist[pt_i-1][1])
#                 pt2 = (tlist[pt_i][0], tlist[pt_i][1])
#                 if not (np.isnan(pt1[0]) or np.isnan(pt1[1]) or
#                         np.isnan(pt2[0]) or np.isnan(pt2[1])):
#                     cv2.line(frame_bgr, pt1, pt2, color_int, 2)


# class MainWindow(QMainWindow):
#     """
#     A PySide2 GUI for multi-object tracking with local merges splitting.
#     Includes brightness, contrast, gamma controls and memory safety.
#     """
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("FlyTracking with Local Merge Splitting + Brightness/Contrast/Gamma")
#         self.resize(1400, 800)

#         # Video display
#         self.video_label = QLabel(self)
#         self.video_label.setAlignment(Qt.AlignCenter)
#         self.video_label.setStyleSheet("background-color: black;")

#         # Layout
#         main_layout = QHBoxLayout()
#         control_layout = QVBoxLayout()

#         # File selection
#         self.btn_file = QPushButton("Select Video File...")
#         self.btn_file.clicked.connect(self.select_file)
#         self.file_path_line = QLineEdit()
#         self.file_path_line.setPlaceholderText("No file selected")

#         file_layout = QHBoxLayout()
#         file_layout.addWidget(self.btn_file)
#         file_layout.addWidget(self.file_path_line)

#         # Parameter panel
#         param_group = QGroupBox("Parameters")
#         form_layout = QFormLayout()

#         self.spin_max_targets = QSpinBox()
#         self.spin_max_targets.setRange(1, 20)
#         self.spin_max_targets.setValue(4)
#         form_layout.addRow("Max Targets", self.spin_max_targets)

#         self.spin_threshold = QSpinBox()
#         self.spin_threshold.setRange(0, 255)
#         self.spin_threshold.setValue(50)
#         form_layout.addRow("Threshold", self.spin_threshold)

#         self.spin_morph_size = QSpinBox()
#         self.spin_morph_size.setRange(1, 50)
#         self.spin_morph_size.setValue(5)
#         form_layout.addRow("Morph Kernel Size", self.spin_morph_size)

#         self.spin_min_contour = QSpinBox()
#         self.spin_min_contour.setRange(0, 20000)
#         self.spin_min_contour.setValue(50)
#         form_layout.addRow("Min Contour Area", self.spin_min_contour)

#         self.spin_max_dist = QSpinBox()
#         self.spin_max_dist.setRange(0, 1000)
#         self.spin_max_dist.setValue(25)
#         form_layout.addRow("Max Distance Thresh", self.spin_max_dist)

#         self.spin_min_detect = QSpinBox()
#         self.spin_min_detect.setRange(0, 1000)
#         self.spin_min_detect.setValue(10)
#         form_layout.addRow("Min Detection Counts", self.spin_min_detect)

#         self.spin_min_track = QSpinBox()
#         self.spin_min_track.setRange(0, 1000)
#         self.spin_min_track.setValue(10)
#         form_layout.addRow("Min Tracking Counts", self.spin_min_track)

#         self.spin_traj_hist = QSpinBox()
#         self.spin_traj_hist.setRange(0, 300)
#         self.spin_traj_hist.setValue(5)
#         form_layout.addRow("Trajectory History (sec)", self.spin_traj_hist)

#         self.spin_kalman_noise = QDoubleSpinBox()
#         self.spin_kalman_noise.setRange(0.0, 1.0)
#         self.spin_kalman_noise.setSingleStep(0.01)
#         self.spin_kalman_noise.setValue(0.03)
#         form_layout.addRow("Kalman Noise Cov", self.spin_kalman_noise)

#         self.spin_kalman_meas_noise = QDoubleSpinBox()
#         self.spin_kalman_meas_noise.setRange(0.0, 1.0)
#         self.spin_kalman_meas_noise.setSingleStep(0.01)
#         self.spin_kalman_meas_noise.setValue(0.1)
#         form_layout.addRow("Kalman Meas Cov", self.spin_kalman_meas_noise)

#         self.spin_resize_factor = QDoubleSpinBox()
#         self.spin_resize_factor.setRange(0.1, 1.0)
#         self.spin_resize_factor.setSingleStep(0.1)
#         self.spin_resize_factor.setValue(1.0)
#         form_layout.addRow("Resize Factor", self.spin_resize_factor)

#         self.spin_merge_area_thr = QSpinBox()
#         self.spin_merge_area_thr.setRange(100, 100000)
#         self.spin_merge_area_thr.setValue(1500)
#         form_layout.addRow("Merge Area Threshold", self.spin_merge_area_thr)

#         # Brightness / Contrast / Gamma
#         self.spin_brightness = QDoubleSpinBox()
#         self.spin_brightness.setRange(-255, 255)
#         self.spin_brightness.setSingleStep(5.0)
#         self.spin_brightness.setValue(0.0)
#         form_layout.addRow("Brightness Offset", self.spin_brightness)

#         self.spin_contrast = QDoubleSpinBox()
#         self.spin_contrast.setRange(0.0, 3.0)
#         self.spin_contrast.setSingleStep(0.1)
#         self.spin_contrast.setValue(1.0)
#         form_layout.addRow("Contrast Scale", self.spin_contrast)

#         self.spin_gamma = QDoubleSpinBox()
#         self.spin_gamma.setRange(0.1, 3.0)
#         self.spin_gamma.setSingleStep(0.1)
#         self.spin_gamma.setValue(1.0)
#         form_layout.addRow("Gamma Correction", self.spin_gamma)

#         # Checkboxes
#         self.chk_show_fg = QCheckBox("Show Foreground Mask")
#         self.chk_show_fg.setChecked(True)
#         form_layout.addRow(self.chk_show_fg)

#         self.chk_show_blob = QCheckBox("Show Blob Overlays")
#         self.chk_show_blob.setChecked(True)
#         form_layout.addRow(self.chk_show_blob)

#         self.chk_show_bg = QCheckBox("Show Lightest Background")
#         self.chk_show_bg.setChecked(True)
#         form_layout.addRow(self.chk_show_bg)

#         param_group.setLayout(form_layout)

#         # Buttons
#         self.btn_preview = QPushButton("Preview")
#         self.btn_preview.setCheckable(True)
#         self.btn_preview.clicked.connect(self.toggle_preview)

#         self.btn_start_tracking = QPushButton("Full Tracking")
#         self.btn_start_tracking.clicked.connect(self.start_full_tracking)

#         self.btn_stop = QPushButton("Stop")
#         self.btn_stop.clicked.connect(self.stop_tracking)

#         control_layout.addLayout(file_layout)
#         control_layout.addWidget(param_group)
#         control_layout.addWidget(self.btn_preview)
#         control_layout.addWidget(self.btn_start_tracking)
#         control_layout.addWidget(self.btn_stop)
#         control_layout.addStretch(1)

#         main_layout.addWidget(self.video_label, stretch=3)
#         main_layout.addLayout(control_layout, stretch=1)

#         central_widget = QWidget()
#         central_widget.setLayout(main_layout)
#         self.setCentralWidget(central_widget)

#         self.tracking_worker = None

#     def select_file(self):
#         file_path, _ = QFileDialog.getOpenFileName(
#             self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov)"
#         )
#         if file_path:
#             self.file_path_line.setText(file_path)

#     def get_parameters_dict(self):
#         """
#         Gather all UI parameters into a dictionary for the worker.
#         """
#         np.random.seed(42)
#         max_targets = self.spin_max_targets.value()
#         color_array = [tuple(c.tolist()) for c in np.random.randint(0, 255, (max_targets, 3))]

#         p = {
#             "MAX_TARGETS": max_targets,
#             "THRESHOLD_VALUE": self.spin_threshold.value(),
#             "MORPH_KERNEL_SIZE": self.spin_morph_size.value(),
#             "MIN_CONTOUR_AREA": self.spin_min_contour.value(),
#             "MAX_DISTANCE_THRESHOLD": self.spin_max_dist.value(),
#             "MIN_DETECTION_COUNTS": self.spin_min_detect.value(),
#             "MIN_TRACKING_COUNTS": self.spin_min_track.value(),
#             "TRAJECTORY_HISTORY_SECONDS": self.spin_traj_hist.value(),
#             "KALMAN_NOISE_COVARIANCE": float(self.spin_kalman_noise.value()),
#             "KALMAN_MEASUREMENT_NOISE_COVARIANCE": float(self.spin_kalman_meas_noise.value()),
#             "RESIZE_FACTOR": float(self.spin_resize_factor.value()),
#             "MERGE_AREA_THRESHOLD": self.spin_merge_area_thr.value(),
#             "BRIGHTNESS": float(self.spin_brightness.value()),
#             "CONTRAST": float(self.spin_contrast.value()),
#             "GAMMA": float(self.spin_gamma.value()),
#             "SHOW_FG": self.chk_show_fg.isChecked(),
#             "SHOW_BLOB": self.chk_show_blob.isChecked(),
#             "SHOW_BG": self.chk_show_bg.isChecked(),
#             "TRAJECTORY_COLORS": color_array
#         }
#         return p

#     def toggle_preview(self, checked):
#         if checked:
#             # Start preview
#             self.start_tracking(preview_mode=True)
#             self.btn_preview.setText("Stop Preview")
#         else:
#             # Stop
#             self.stop_tracking()
#             self.btn_preview.setText("Preview")

#     def start_full_tracking(self):
#         if self.btn_preview.isChecked():
#             self.btn_preview.setChecked(False)
#             self.btn_preview.setText("Preview")
#         self.start_tracking(preview_mode=False)

#     def start_tracking(self, preview_mode: bool):
#         video_path = self.file_path_line.text()
#         if not video_path:
#             QMessageBox.warning(self, "No file selected", "Please select a video file first!")
#             if preview_mode:
#                 self.btn_preview.setChecked(False)
#                 self.btn_preview.setText("Preview")
#             return

#         if self.tracking_worker and self.tracking_worker.isRunning():
#             QMessageBox.warning(self, "Worker busy", "A tracking thread is already running. Stop it first.")
#             return

#         self.tracking_worker = TrackingWorker(video_path)
#         self.tracking_worker.set_parameters(self.get_parameters_dict())
#         self.tracking_worker.frame_signal.connect(self.on_new_frame)
#         self.tracking_worker.finished_signal.connect(self.on_tracking_finished)
#         self.tracking_worker.start()

#     def stop_tracking(self):
#         if self.tracking_worker and self.tracking_worker.isRunning():
#             self.tracking_worker.stop()

#     @Slot(np.ndarray)
#     def on_new_frame(self, rgb_frame: np.ndarray):
#         """
#         Convert RGB array -> QImage -> QPixmap -> video_label display.
#         """
#         h, w, c = rgb_frame.shape
#         bytes_per_line = c * w
#         qimg = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
#         self.video_label.setPixmap(QPixmap.fromImage(qimg))

#     @Slot(bool, list)
#     def on_tracking_finished(self, finished_normally: bool, fps_list: list):
#         """
#         Called when the worker is finished. If it's a full run,
#         we can handle saving CSV or other post-processing here.
#         """
#         preview_mode = self.btn_preview.isChecked()
#         if preview_mode:
#             self.btn_preview.setChecked(False)
#             self.btn_preview.setText("Preview")

#         if finished_normally and not preview_mode:
#             QMessageBox.information(
#                 self, "Tracking Finished",
#                 "Full run completed. You can implement CSV saving or final analysis here."
#             )
#             self.plot_fps(fps_list)

#         gc.collect()

#     def plot_fps(self, fps_list):
#         """
#         Basic FPS plot if enough data is available.
#         """
#         if len(fps_list) < 2:
#             QMessageBox.information(self, "FPS Plot", "Not enough data to plot.")
#             return
#         plt.figure()
#         plt.plot(fps_list, label="FPS")
#         plt.xlabel("Frame Index")
#         plt.ylabel("Frames Per Second")
#         plt.title("Tracking FPS Over Time")
#         plt.legend()
#         plt.show()


# def main():
#     app = QApplication(sys.argv)
#     window = MainWindow()
#     window.show()
#     sys.exit(app.exec_())


# if __name__ == "__main__":
#     main()

"""
Full Production-Ready PySide2 Script for Multi-Fly Tracking with:
1) Brightness/Contrast/Gamma
2) Lightest-Pixel Background
3) Local "Conservative" Splitting for Merges (heavy erosion) applied only after detection is stable
4) Saving Trajectories to a User-Specified CSV
5) Memory-Safe Practices (explicit cleanup + gc)

Usage:
------
1. Install dependencies: PySide2, OpenCV, NumPy, SciPy, matplotlib.
2. Run: python multi_fly_tracking.py
3. In the UI:
   - "Select Video File..." for your .mp4 or .avi
   - Set parameters (threshold, morphological kernel size, min contour area, etc.)
   - Adjust brightness/contrast/gamma as needed
   - Optionally set "Output CSV Path" to store final trajectories
   - "Preview" toggles live preview
   - "Full Tracking" processes entire video until the end or user stops
   - A heavier morphological approach is used locally only for suspicious merges,
     but ONLY after we've successfully detected all targets for the first time.
"""

import sys
import time
import gc
import csv
import numpy as np
import cv2
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

from PySide2.QtCore import (
    Qt, QThread, Signal, Slot, QMutex
)
from PySide2.QtGui import (
    QImage, QPixmap
)
from PySide2.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QFileDialog, QSpinBox, QDoubleSpinBox, QCheckBox, QMessageBox,
    QGroupBox, QFormLayout, QLineEdit
)


def apply_image_adjustments(gray_frame: np.ndarray,
                            brightness: float,
                            contrast: float,
                            gamma: float) -> np.ndarray:
    """
    Apply brightness, contrast, and gamma corrections to a grayscale image.

    Parameters
    ----------
    gray_frame : np.ndarray
        8-bit grayscale frame.
    brightness : float
        Pixel offset added after contrast scaling. Negative darkens, positive brightens.
    contrast : float
        Pixel scale factor. 1.0 => no change, >1 => more contrast, <1 => less contrast.
    gamma : float
        Gamma correction factor. 1.0 => no change, <1 => darkens midtones, >1 => brightens midtones.

    Returns
    -------
    adjusted : np.ndarray
        The corrected 8-bit grayscale image.
    """
    # 1) Brightness/Contrast
    adjusted = cv2.convertScaleAbs(gray_frame, alpha=contrast, beta=brightness)

    # 2) Gamma
    if abs(gamma - 1.0) > 1e-3:
        look_up_table = np.array([
            np.clip(((i / 255.0) ** (1.0 / gamma)) * 255.0, 0, 255)
            for i in range(256)
        ], dtype=np.uint8)
        adjusted = cv2.LUT(adjusted, look_up_table)

    return adjusted


class TrackingWorker(QThread):
    """
    A QThread worker for multi-object tracking with:
      - Lightest-pixel background modeling
      - Local "conservative" morphological splitting
      - Kalman + Hungarian assignment
      - Optional brightness/contrast/gamma

    Signals
    -------
    frame_signal : np.ndarray
        Emitted with the current overlay frame (RGB).
    finished_signal : (bool, list, list)
        (finished_normally, fps_list, final_trajectories)
        - finished_normally : True if end-of-video, False if user forced stop
        - fps_list : List of FPS values over frames
        - final_trajectories : The worker's trajectories for each target
    """
    frame_signal = Signal(np.ndarray)
    finished_signal = Signal(bool, list, list)  # includes trajectories

    def __init__(self, video_path: str, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.params_mutex = QMutex()
        self.parameters = {}

        self._stop_flag = False
        self.cap = None

        # Kalman/tracking structures
        self.kalman_filters = []
        self.track_ids = []
        self.trajectories = []

        # Background model
        self.background_model_lightest = None

        # Tracking flags
        self.detection_initialized = False  # becomes True after all targets detected for MIN_DETECTION_COUNTS
        self.tracking_stabilized = False
        self.detection_counts = 0
        self.tracking_counts = 0

        # Others
        self.fps_list = []
        self.frame_count = 0
        self.start_time = 0

    def set_parameters(self, new_params: dict):
        """Thread-safe parameter update."""
        self.params_mutex.lock()
        self.parameters = new_params
        self.params_mutex.unlock()

    def get_current_params(self):
        self.params_mutex.lock()
        p = dict(self.parameters)
        self.params_mutex.unlock()
        return p

    def stop(self):
        """Signal thread to end ASAP."""
        self._stop_flag = True

    def init_kalman_filters(self, p):
        """Initialize multiple Kalman filters, one per target."""
        kf_list = []
        for _ in range(p["MAX_TARGETS"]):
            kf = cv2.KalmanFilter(5, 3)
            kf.measurementMatrix = np.array([
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0]
            ], np.float32)
            kf.transitionMatrix = np.array([
                [1, 0, 0, 1, 0],
                [0, 1, 0, 0, 1],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1]
            ], np.float32)
            kf.processNoiseCov = np.eye(5, dtype=np.float32) * p["KALMAN_NOISE_COVARIANCE"]
            kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * p["KALMAN_MEASUREMENT_NOISE_COVARIANCE"]
            kf.errorCovPre = np.eye(5, dtype=np.float32)
            kf_list.append(kf)
        return kf_list

    def run(self):
        """Main thread loop for reading frames, applying detection+tracking, and emitting results."""
        gc.collect()
        self._stop_flag = False
        p = self.get_current_params()

        # Attempt to open
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            print(f"[ERROR] Cannot open video file: {self.video_path}")
            self.finished_signal.emit(True, [], [])
            return

        # Re-init
        self.background_model_lightest = None
        self.kalman_filters = self.init_kalman_filters(p)
        self.track_ids = np.arange(p["MAX_TARGETS"])
        self.trajectories = [[] for _ in range(p["MAX_TARGETS"])]

        self.detection_initialized = False
        self.tracking_stabilized = False
        self.detection_counts = 0
        self.tracking_counts = 0

        self.fps_list = []
        self.frame_count = 0
        self.start_time = time.time()

        try:
            while not self._stop_flag:
                ret, frame = self.cap.read()
                if not ret:
                    break

                self.frame_count += 1
                current_params = self.get_current_params()

                # Optionally resize
                if current_params.get("RESIZE_FACTOR", 1.0) < 1.0:
                    rsz = current_params["RESIZE_FACTOR"]
                    new_w = int(frame.shape[1] * rsz)
                    new_h = int(frame.shape[0] * rsz)
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

                # Convert to gray
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Brightness/Contrast/Gamma
                br = current_params.get("BRIGHTNESS", 0.0)
                ct = current_params.get("CONTRAST", 1.0)
                gm = current_params.get("GAMMA", 1.0)
                gray_frame = apply_image_adjustments(gray_frame, br, ct, gm)

                # Initialize background if needed
                if self.background_model_lightest is None:
                    self.background_model_lightest = gray_frame.astype(np.float32)
                    # Show first frame
                    self.emit_frame(frame)
                    self._cleanup_locals(frame, gray_frame)
                    continue

                # Update "lightest" background
                self.background_model_lightest = np.maximum(self.background_model_lightest,
                                                            gray_frame.astype(np.float32))
                bg_uint8 = cv2.convertScaleAbs(self.background_model_lightest)

                # Liberal mask
                fg_mask = cv2.absdiff(bg_uint8, gray_frame)
                _, fg_mask = cv2.threshold(fg_mask, current_params["THRESHOLD_VALUE"], 255, cv2.THRESH_BINARY)

                # Morphological open/close
                ksize = current_params["MORPH_KERNEL_SIZE"]
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

                # We'll do an initial set of contours without local-splitting
                main_mask = fg_mask.copy()
                contours, _ = cv2.findContours(main_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Quick ellipse fit to see how many targets we get
                measurements_firstpass = []
                sizes_firstpass = []
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area < current_params["MIN_CONTOUR_AREA"]:
                        continue
                    if len(cnt) >= 5:
                        ellipse = cv2.fitEllipse(cnt)
                        (cx, cy), (MA, ma), angle = ellipse
                        measurements_firstpass.append(np.array([cx, cy, np.deg2rad(angle)], dtype=np.float32))
                        sizes_firstpass.append(area)

                # If we found more than needed, keep largest
                if len(measurements_firstpass) > current_params["MAX_TARGETS"]:
                    sorted_idx = np.argsort(sizes_firstpass)[::-1]
                    measurements_firstpass = [measurements_firstpass[i] for i in sorted_idx[:current_params["MAX_TARGETS"]]]

                # Update detection counts
                if len(measurements_firstpass) == current_params["MAX_TARGETS"]:
                    self.detection_counts += 1
                else:
                    self.detection_counts = 0

                if self.detection_counts >= current_params["MIN_DETECTION_COUNTS"]:
                    self.detection_initialized = True

                # Now apply local conservative splitting *only if* detection_initialized
                if self.detection_initialized:
                    # Identify suspicious merges
                    merge_area_thr = current_params.get("MERGE_AREA_THRESHOLD", 1500)
                    total_contours = sum(1 for cnt in contours if cv2.contourArea(cnt) > 0)
                    suspicious_bboxes = []
                    for cnt in contours:
                        area = cv2.contourArea(cnt)
                        if area < 1:
                            continue
                        x, y, w, h = cv2.boundingRect(cnt)
                        # If area is large or if too few contours => suspicious
                        if area > merge_area_thr or total_contours < current_params["MAX_TARGETS"]:
                            suspicious_bboxes.append((x, y, w, h))

                    # Locally refine suspicious merges
                    for (bx, by, bw, bh) in suspicious_bboxes:
                        sub_mask = main_mask[by:by+bh, bx:bx+bw]
                        if np.count_nonzero(sub_mask) < 10:
                            continue
                        # Use a heavier morphological approach
                        refined_mask = self._local_conservative_split(sub_mask, current_params)
                        main_mask[by:by+bh, bx:bx+bw] = refined_mask

                # Final measurement pass
                final_contours, _ = cv2.findContours(main_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                measurements, sizes = [], []
                for cnt in final_contours:
                    area = cv2.contourArea(cnt)
                    if area < current_params["MIN_CONTOUR_AREA"]:
                        continue
                    if len(cnt) >= 5:
                        ellipse = cv2.fitEllipse(cnt)
                        (cx, cy), (MA, ma), angle = ellipse
                        measurements.append(np.array([cx, cy, np.deg2rad(angle)], dtype=np.float32))
                        sizes.append(area)

                if len(measurements) > current_params["MAX_TARGETS"]:
                    sorted_idx = np.argsort(sizes)[::-1]
                    measurements = [measurements[i] for i in sorted_idx[:current_params["MAX_TARGETS"]]]

                # Tracking pipeline
                overlay_frame = frame.copy()
                if self.detection_initialized and measurements:
                    if self.frame_count == 1:
                        h_, w_ = gray_frame.shape
                        for kf in self.kalman_filters:
                            kf.statePre = np.array([
                                np.random.randint(0, w_),
                                np.random.randint(0, h_),
                                0, 0, 0
                            ], dtype=np.float32)
                            kf.statePost = kf.statePre.copy()

                    # Predict
                    predicted_positions = []
                    for kf in self.kalman_filters:
                        pred = kf.predict()
                        predicted_positions.append(pred[:3].flatten())
                    predicted_positions = np.array(predicted_positions, dtype=np.float32)

                    # Hungarian assignment
                    cost_matrix = np.zeros((current_params["MAX_TARGETS"], len(measurements)), dtype=np.float32)
                    for i, pred_pos in enumerate(predicted_positions):
                        for j, meas in enumerate(measurements):
                            pos_cost = distance.euclidean(pred_pos[:2], meas[:2])
                            angle_diff = abs(pred_pos[2] - meas[2])
                            angle_diff = min(angle_diff, 2*np.pi - angle_diff)
                            cost_matrix[i, j] = pos_cost + angle_diff

                    row_idx, col_idx = linear_sum_assignment(cost_matrix)
                    avg_cost = 0.0
                    for row, col in zip(row_idx, col_idx):
                        if row < current_params["MAX_TARGETS"] and col < len(measurements):
                            if (not self.tracking_stabilized) or (cost_matrix[row, col] < current_params["MAX_DISTANCE_THRESHOLD"]):
                                kf = self.kalman_filters[row]
                                measure_vec = np.array([
                                    [measurements[col][0]],
                                    [measurements[col][1]],
                                    [measurements[col][2]]
                                ], dtype=np.float32)
                                kf.correct(measure_vec)
                                x_corr, y_corr, theta_corr = measurements[col]
                                ts = time.time()
                                self.trajectories[row].append((int(x_corr), int(y_corr), float(theta_corr), ts))
                                self._prune_traj(row, ts, current_params["TRAJECTORY_HISTORY_SECONDS"])
                                avg_cost += cost_matrix[row, col] / current_params["MAX_TARGETS"]
                            else:
                                self.tracking_stabilized = False
                                self.tracking_counts = 0
                                ts = time.time()
                                self.trajectories[row].append((np.nan, np.nan, np.nan, ts))
                                self._prune_traj(row, ts, current_params["TRAJECTORY_HISTORY_SECONDS"])

                    if avg_cost < current_params["MAX_DISTANCE_THRESHOLD"]:
                        self.tracking_counts += 1
                    else:
                        self.tracking_counts = 0

                    if self.tracking_counts >= current_params["MIN_TRACKING_COUNTS"] and not self.tracking_stabilized:
                        self.tracking_stabilized = True
                        print(f"[INFO] Tracking Stabilized (avg cost={avg_cost:.2f})")

                    # Draw overlays
                    if current_params["SHOW_BLOB"]:
                        self._draw_overlays(overlay_frame, self.trajectories, current_params)

                # Foreground mask corner
                if current_params["SHOW_FG"]:
                    mask_bgr = cv2.cvtColor(main_mask, cv2.COLOR_GRAY2BGR)
                    scale = 0.3
                    sw = int(mask_bgr.shape[1]*scale)
                    sh = int(mask_bgr.shape[0]*scale)
                    small_mask = cv2.resize(mask_bgr, (sw, sh), interpolation=cv2.INTER_AREA)
                    overlay_frame[0:sh, 0:sw] = small_mask

                # Lightest background corner
                if current_params["SHOW_BG"]:
                    bg_bgr = cv2.cvtColor(bg_uint8, cv2.COLOR_GRAY2BGR)
                    scale = 0.3
                    sw = int(bg_bgr.shape[1]*scale)
                    sh = int(bg_bgr.shape[0]*scale)
                    small_bg = cv2.resize(bg_bgr, (sw, sh), interpolation=cv2.INTER_AREA)
                    x_offset = overlay_frame.shape[1] - sw
                    overlay_frame[0:sh, x_offset:overlay_frame.shape[1]] = small_bg

                # Compute FPS
                elapsed = time.time() - self.start_time
                if elapsed > 0:
                    self.fps_list.append(self.frame_count / elapsed)

                # Emit frame
                self.emit_frame(overlay_frame)
                self._cleanup_locals(
                    frame, gray_frame, bg_uint8, fg_mask, kernel, main_mask,
                    overlay_frame, contours, final_contours
                )
                gc.collect()

        except Exception as e:
            print(f"[ERROR] Exception in worker thread: {e}")
            self._stop_flag = True
        finally:
            if self.cap:
                self.cap.release()

        done_normally = (not self._stop_flag)
        # Emit final results, including trajectories
        self.finished_signal.emit(done_normally, self.fps_list, self.trajectories)
        gc.collect()

    def _local_conservative_split(self, sub_mask: np.ndarray, p: dict) -> np.ndarray:
        """
        A heavier morphological approach to separate merges locally.
        For instance, multiple erosions with a slightly bigger kernel.

        sub_mask : np.ndarray, local portion of the main mask in suspicious bounding box
        p : dict, user parameters

        Returns
        -------
        refined : np.ndarray
            The updated sub-mask after "conservative" morphological ops.
        """
        # Let user specify how big the kernel or how many iterations for this local step
        ksize = p.get("CONSERVATIVE_KERNEL_SIZE", 5)
        erode_iterations = p.get("CONSERVATIVE_ERODE_ITER", 2)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        # Heavier erode
        refined = cv2.erode(sub_mask, kernel, iterations=erode_iterations)

        # Possibly do an additional open to remove small lumps
        refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel)

        return refined

    def emit_frame(self, bgr_frame: np.ndarray):
        """Convert BGR to RGB, then emit via frame_signal."""
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        self.frame_signal.emit(rgb_frame)

    def _cleanup_locals(self, *args):
        """Delete references to large objects, then rely on gc."""
        for obj in args:
            del obj

    def _prune_traj(self, idx, current_ts, history_sec):
        """Keep only the last 'history_sec' seconds of trajectory."""
        self.trajectories[idx] = [
            (x, y, th, t) for (x, y, th, t) in self.trajectories[idx]
            if (current_ts - t) <= history_sec
        ]

    def _draw_overlays(self, frame_bgr: np.ndarray, trajectories: list, p: dict):
        """
        Draw circles, orientation lines, and short trajectories on frame_bgr.
        """
        for i, tlist in enumerate(trajectories):
            if not tlist:
                continue
            x, y, theta, _ = tlist[-1]
            if np.isnan(x) or np.isnan(y):
                continue

            color = p["TRAJECTORY_COLORS"][i % len(p["TRAJECTORY_COLORS"])]
            color_int = tuple(int(c) for c in color)

            cv2.circle(frame_bgr, (x, y), 10, color_int, -1)
            length = 20
            x_end = int(x + length * np.cos(theta))
            y_end = int(y + length * np.sin(theta))
            cv2.line(frame_bgr, (x, y), (x_end, y_end), color_int, 2)
            cv2.putText(frame_bgr, f"ID: {i}", (x+15, y-15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_int, 2)

            # Trajectory lines
            for pt_i in range(1, len(tlist)):
                pt1 = (tlist[pt_i-1][0], tlist[pt_i-1][1])
                pt2 = (tlist[pt_i][0], tlist[pt_i][1])
                if not (np.isnan(pt1[0]) or np.isnan(pt1[1]) or
                        np.isnan(pt2[0]) or np.isnan(pt2[1])):
                    cv2.line(frame_bgr, pt1, pt2, color_int, 2)


class MainWindow(QMainWindow):
    """
    Main PySide2 GUI:
      - File selection
      - Parameter controls
      - Output CSV path
      - Preview or Full Tracking
      - Save final trajectories to CSV if Full Tracking completes.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Object Tracking (Local Conservative Splitting)")

        self.resize(1400, 800)
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")

        main_layout = QHBoxLayout()
        control_layout = QVBoxLayout()

        # File selection
        self.btn_file = QPushButton("Select Video File...")
        self.btn_file.clicked.connect(self.select_file)
        self.file_path_line = QLineEdit()
        self.file_path_line.setPlaceholderText("No file selected")

        file_layout = QHBoxLayout()
        file_layout.addWidget(self.btn_file)
        file_layout.addWidget(self.file_path_line)

        # CSV Path
        self.btn_csv = QPushButton("Output CSV Path...")
        self.btn_csv.clicked.connect(self.select_csv)
        self.csv_path_line = QLineEdit()
        self.csv_path_line.setPlaceholderText("No CSV selected")

        csv_layout = QHBoxLayout()
        csv_layout.addWidget(self.btn_csv)
        csv_layout.addWidget(self.csv_path_line)

        # Parameter group
        param_group = QGroupBox("Parameters")
        form_layout = QFormLayout()

        # Basic detection/tracking
        self.spin_max_targets = QSpinBox(); self.spin_max_targets.setRange(1, 20); self.spin_max_targets.setValue(4)
        form_layout.addRow("Max Targets", self.spin_max_targets)

        self.spin_threshold = QSpinBox(); self.spin_threshold.setRange(0, 255); self.spin_threshold.setValue(50)
        form_layout.addRow("Threshold", self.spin_threshold)

        self.spin_morph_size = QSpinBox(); self.spin_morph_size.setRange(1, 50); self.spin_morph_size.setValue(5)
        form_layout.addRow("Morph Kernel Size", self.spin_morph_size)

        self.spin_min_contour = QSpinBox(); self.spin_min_contour.setRange(0, 20000); self.spin_min_contour.setValue(50)
        form_layout.addRow("Min Contour Area", self.spin_min_contour)

        self.spin_max_dist = QSpinBox(); self.spin_max_dist.setRange(0, 2000); self.spin_max_dist.setValue(25)
        form_layout.addRow("Max Distance Thresh", self.spin_max_dist)

        self.spin_min_detect = QSpinBox(); self.spin_min_detect.setRange(0, 1000); self.spin_min_detect.setValue(10)
        form_layout.addRow("Min Detection Counts", self.spin_min_detect)

        self.spin_min_track = QSpinBox(); self.spin_min_track.setRange(0, 1000); self.spin_min_track.setValue(10)
        form_layout.addRow("Min Tracking Counts", self.spin_min_track)

        self.spin_traj_hist = QSpinBox(); self.spin_traj_hist.setRange(0, 300); self.spin_traj_hist.setValue(5)
        form_layout.addRow("Trajectory History (sec)", self.spin_traj_hist)

        # Kalman
        self.spin_kalman_noise = QDoubleSpinBox(); self.spin_kalman_noise.setRange(0.0, 1.0); self.spin_kalman_noise.setValue(0.03)
        self.spin_kalman_noise.setSingleStep(0.01)
        form_layout.addRow("Kalman Noise Cov", self.spin_kalman_noise)

        self.spin_kalman_meas_noise = QDoubleSpinBox(); self.spin_kalman_meas_noise.setRange(0.0, 1.0); self.spin_kalman_meas_noise.setValue(0.1)
        self.spin_kalman_meas_noise.setSingleStep(0.01)
        form_layout.addRow("Kalman Meas Cov", self.spin_kalman_meas_noise)

        # Resizing
        self.spin_resize_factor = QDoubleSpinBox(); self.spin_resize_factor.setRange(0.1, 1.0); self.spin_resize_factor.setValue(1.0)
        self.spin_resize_factor.setSingleStep(0.1)
        form_layout.addRow("Resize Factor", self.spin_resize_factor)

        # Merge area + local conservative
        self.spin_merge_area_thr = QSpinBox(); self.spin_merge_area_thr.setRange(100, 100000); self.spin_merge_area_thr.setValue(1500)
        form_layout.addRow("Merge Area Threshold", self.spin_merge_area_thr)

        self.spin_conservative_kernel = QSpinBox(); self.spin_conservative_kernel.setRange(1, 50); self.spin_conservative_kernel.setValue(5)
        form_layout.addRow("Conservative Kernel Size", self.spin_conservative_kernel)

        self.spin_conservative_iter = QSpinBox(); self.spin_conservative_iter.setRange(1, 10); self.spin_conservative_iter.setValue(2)
        form_layout.addRow("Conservative Erode Iter", self.spin_conservative_iter)

        # Brightness/Contrast/Gamma
        self.spin_brightness = QDoubleSpinBox(); self.spin_brightness.setRange(-255, 255); self.spin_brightness.setValue(0.0); self.spin_brightness.setSingleStep(5)
        form_layout.addRow("Brightness", self.spin_brightness)

        self.spin_contrast = QDoubleSpinBox(); self.spin_contrast.setRange(0.0, 3.0); self.spin_contrast.setValue(1.0); self.spin_contrast.setSingleStep(0.1)
        form_layout.addRow("Contrast", self.spin_contrast)

        self.spin_gamma = QDoubleSpinBox(); self.spin_gamma.setRange(0.1, 3.0); self.spin_gamma.setValue(1.0); self.spin_gamma.setSingleStep(0.1)
        form_layout.addRow("Gamma", self.spin_gamma)

        # Checkboxes
        self.chk_show_fg = QCheckBox("Show Foreground Mask"); self.chk_show_fg.setChecked(True)
        form_layout.addRow(self.chk_show_fg)

        self.chk_show_blob = QCheckBox("Show Blob Overlays"); self.chk_show_blob.setChecked(True)
        form_layout.addRow(self.chk_show_blob)

        self.chk_show_bg = QCheckBox("Show Lightest Background"); self.chk_show_bg.setChecked(True)
        form_layout.addRow(self.chk_show_bg)

        param_group.setLayout(form_layout)

        # Buttons: Preview / Full Tracking / Stop
        self.btn_preview = QPushButton("Preview")
        self.btn_preview.setCheckable(True)
        self.btn_preview.clicked.connect(self.toggle_preview)

        self.btn_start_tracking = QPushButton("Full Tracking")
        self.btn_start_tracking.clicked.connect(self.start_full_tracking)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self.stop_tracking)

        # Layout assembly
        control_layout.addLayout(file_layout)
        control_layout.addLayout(csv_layout)
        control_layout.addWidget(param_group)
        control_layout.addWidget(self.btn_preview)
        control_layout.addWidget(self.btn_start_tracking)
        control_layout.addWidget(self.btn_stop)
        control_layout.addStretch(1)

        main_layout.addWidget(self.video_label, stretch=3)
        main_layout.addLayout(control_layout, stretch=1)
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self.tracking_worker = None

        # For storing final trajectories
        self.final_trajectories = []

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov)")
        if file_path:
            self.file_path_line.setText(file_path)

    def select_csv(self):
        csv_path, _ = QFileDialog.getSaveFileName(self, "Select CSV File to Save", "", "CSV Files (*.csv)")
        if csv_path:
            self.csv_path_line.setText(csv_path)

    def get_parameters_dict(self):
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
            "RESIZE_FACTOR": float(self.spin_resize_factor.value()),
            "MERGE_AREA_THRESHOLD": self.spin_merge_area_thr.value(),

            "CONSERVATIVE_KERNEL_SIZE": self.spin_conservative_kernel.value(),
            "CONSERVATIVE_ERODE_ITER": self.spin_conservative_iter.value(),

            "BRIGHTNESS": float(self.spin_brightness.value()),
            "CONTRAST": float(self.spin_contrast.value()),
            "GAMMA": float(self.spin_gamma.value()),

            "SHOW_FG": self.chk_show_fg.isChecked(),
            "SHOW_BLOB": self.chk_show_blob.isChecked(),
            "SHOW_BG": self.chk_show_bg.isChecked(),
            "TRAJECTORY_COLORS": color_array
        }
        return params

    def toggle_preview(self, checked):
        if checked:
            self.start_tracking(preview_mode=True)
            self.btn_preview.setText("Stop Preview")
        else:
            self.stop_tracking()
            self.btn_preview.setText("Preview")

    def start_full_tracking(self):
        if self.btn_preview.isChecked():
            self.btn_preview.setChecked(False)
            self.btn_preview.setText("Preview")
        self.start_tracking(preview_mode=False)

    def start_tracking(self, preview_mode: bool):
        video_path = self.file_path_line.text()
        if not video_path:
            QMessageBox.warning(self, "No file selected", "Please select a video file first!")
            if preview_mode:
                self.btn_preview.setChecked(False)
                self.btn_preview.setText("Preview")
            return

        if self.tracking_worker and self.tracking_worker.isRunning():
            QMessageBox.warning(self, "Worker busy", "A tracking thread is already running. Stop it first.")
            return

        self.tracking_worker = TrackingWorker(video_path)
        self.tracking_worker.set_parameters(self.get_parameters_dict())
        self.tracking_worker.frame_signal.connect(self.on_new_frame)
        self.tracking_worker.finished_signal.connect(self.on_tracking_finished)
        self.tracking_worker.start()

    def stop_tracking(self):
        if self.tracking_worker and self.tracking_worker.isRunning():
            self.tracking_worker.stop()

    @Slot(np.ndarray)
    def on_new_frame(self, rgb_frame: np.ndarray):
        """Receive frames from worker -> QImage -> QPixmap -> Display."""
        h, w, c = rgb_frame.shape
        bytes_per_line = c * w
        qimg = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg))

    @Slot(bool, list, list)
    def on_tracking_finished(self, finished_normally: bool, fps_list: list, final_trajectories: list):
        """Called when worker finishes. We can save CSV if it's a full run."""
        preview_mode = self.btn_preview.isChecked()
        if preview_mode:
            self.btn_preview.setChecked(False)
            self.btn_preview.setText("Preview")

        if finished_normally and not preview_mode:
            self.final_trajectories = final_trajectories
            QMessageBox.information(
                self, "Tracking Finished",
                "Full run completed. If an output CSV path is provided, results will be saved."
            )
            self.plot_fps(fps_list)
            self.save_trajectories_to_csv()

        gc.collect()

    def plot_fps(self, fps_list):
        if len(fps_list) < 2:
            QMessageBox.information(self, "FPS Plot", "Not enough data to plot.")
            return
        plt.figure()
        plt.plot(fps_list, label="FPS")
        plt.xlabel("Frame Index")
        plt.ylabel("Frames Per Second")
        plt.title("Tracking FPS Over Time")
        plt.legend()
        plt.show()

    def save_trajectories_to_csv(self):
        """
        Write the final trajectories to the user-specified CSV, if any.
        Format: [TargetID, FrameIndexInTarget, X, Y, Theta, Timestamp]
        """
        csv_path = self.csv_path_line.text()
        if not csv_path:
            # No user-specified path => skip
            return
        try:
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["TargetID", "Index", "X", "Y", "Theta", "Timestamp"])
                for target_id, track_list in enumerate(self.final_trajectories):
                    for idx, (x, y, th, tstamp) in enumerate(track_list):
                        writer.writerow([target_id, idx, x, y, th, tstamp])
            QMessageBox.information(self, "Saved CSV", f"Trajectories saved to {csv_path}")
        except Exception as e:
            QMessageBox.warning(self, "CSV Error", f"Could not save CSV: {e}")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
