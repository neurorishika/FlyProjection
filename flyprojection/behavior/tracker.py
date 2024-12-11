import cv2
import numpy as np
import time
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment

def process_contours(contours_list, n_targets, min_contour_area, max_contour_area):
    """
    Extract valid target measurements (x, y, angle) from contours based on size and shape.
    Keeps only the largest n_targets contours if more are found.

    Parameters
    ----------
    contours_list : list
        List of contours from which to extract measurements.

    Returns
    -------
    measurements : list of numpy.ndarray
        List of measurements [x, y, angle_in_radians] for valid contours.
    """
    measurements = []
    sizes = []
    for cnt in contours_list:
        size = cv2.contourArea(cnt)
        if size < min_contour_area or size > max_contour_area:
            continue
        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            (x, y), _, angle = ellipse
            angle_radians = np.deg2rad(angle)
            measurements.append(np.array([x, y, angle_radians], dtype=np.float32))
            sizes.append(size)

    # Keep only largest n_targets contours if more are found
    if len(measurements) > n_targets:
        sorted_indices = np.argsort(sizes)[::-1][:n_targets]
        measurements = [measurements[i] for i in sorted_indices]
    return measurements
class FastTracker:
    def __init__(self, 
                 camera, 
                 n_targets,
                 kalman_noise_covariance=0.03,
                 kalman_measurement_noise_covariance=0.1,
                 morph_kernel_size=5,
                 threshold_value=50,
                 trajectory_thickness=2,
                 trajectory_history_seconds=5,
                 min_contour_area=50,
                 max_contour_area=500,
                 max_distance_threshold=25,
                 min_detection_counts=10,
                 min_tracking_counts=10,
                 smoothing_alpha=0.5,
                 moving_threshold=5,
                 debug=False):
        """
        Initialize the FastTracker class, which tracks multiple targets in a video stream from a given camera.
        
        The tracker uses a combination of background modeling, contour detection, 
        Kalman filtering, and trajectory management to identify and track multiple targets 
        over time. It supports target position/velocity smoothing and orientation estimation.

        Parameters
        ----------
        camera : object
            The camera object providing frames (assumed to have a .get_array() method that returns a grayscale image).
        n_targets : int
            The number of targets to track.
        kalman_noise_covariance : float, optional
            Process noise covariance for the Kalman filters.
        kalman_measurement_noise_covariance : float, optional
            Measurement noise covariance for the Kalman filters.
        morph_kernel_size : int, optional
            Kernel size for morphological operations used in foreground mask processing.
        threshold_value : int, optional
            Threshold value for background subtraction.
        trajectory_thickness : int, optional
            Thickness of the lines drawn for trajectories (debug display).
        trajectory_history_seconds : float, optional
            How many seconds of trajectory history to keep for each target.
        min_contour_area : int, optional
            Minimum contour area for a detection to be considered a valid target.
        max_contour_area : int, optional
            Maximum contour area to consider a valid target.
        max_distance_threshold : float, optional
            Maximum allowed distance between prediction and measurement before tracking is reset.
        min_detection_counts : int, optional
            Minimum number of consecutive frames where targets are fully detected before initialization.
        min_tracking_counts : int, optional
            Minimum number of consecutive frames of stable tracking before considered stabilized.
        smoothing_alpha : float, optional
            Exponential smoothing factor for smoothing velocities and angles.
        moving_threshold : float, optional
            Threshold speed below which the target is considered stationary.
        debug : bool, optional
            If True, display debugging windows and print debug statements.
        """
        self.camera = camera
        self.n_targets = n_targets
        self.kalman_noise_covariance = kalman_noise_covariance
        self.kalman_measurement_noise_covariance = kalman_measurement_noise_covariance
        self.morph_kernel_size = morph_kernel_size
        self.threshold_value = threshold_value
        self.trajectory_thickness = trajectory_thickness
        self.trajectory_history_seconds = trajectory_history_seconds
        self.min_contour_area = min_contour_area
        self.max_contour_area = max_contour_area
        self.max_distance_threshold = max_distance_threshold
        self.min_detection_counts = min_detection_counts
        self.min_tracking_counts = min_tracking_counts
        self.smoothing_alpha = smoothing_alpha
        self.moving_threshold = moving_threshold
        self.debug = debug

        self.background_model_lightest = None
        self.detection_initialized = False
        self.tracking_stabilized = False
        self.detection_counts = 0
        self.tracking_counts = 0
        self.conservative_used = False
        self.frame_count = 0
        self.start_time = time.time()

        # Initialize trajectories for each target
        self.trajectories = [[] for _ in range(self.n_targets)]  
        self.track_ids = np.arange(self.n_targets)

        # Precompute morphological kernel
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_kernel_size, self.morph_kernel_size))

        # Initialize Kalman Filters for multiple targets
        self.kalman_filters = [cv2.KalmanFilter(5, 3) for _ in range(self.n_targets)]
        for i, kf in enumerate(self.kalman_filters):
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

            kf.processNoiseCov = np.eye(5, dtype=np.float32) * self.kalman_noise_covariance
            kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * self.kalman_measurement_noise_covariance

            kf.statePre = np.array([
                np.random.randint(0, self.camera.WIDTH), 
                np.random.randint(0, self.camera.HEIGHT), 
                0, 0, 0
            ], np.float32)
            kf.statePost = kf.statePre.copy()

            kf.errorCovPre = np.eye(5, dtype=np.float32)

        # Assign random colors for trajectory visualization
        self.trajectory_colors = [
            tuple(color) for color in np.random.randint(0, 255, (self.n_targets, 3)).tolist()
        ]

        # Debug windows setup if requested
        if self.debug:
            cv2.namedWindow('Foreground Mask', cv2.WINDOW_NORMAL)
            cv2.namedWindow('Blob Detection', cv2.WINDOW_NORMAL)
            cv2.namedWindow('Lightest Background', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Foreground Mask', 500, 500)
            cv2.resizeWindow('Blob Detection', 500, 500)
            cv2.resizeWindow('Lightest Background', 500, 500)

    def process_next_frame(self, return_camera_image = False):
        """
        Process the next frame from the camera and update the tracking estimates.

        Steps:
        1. Acquire the next frame from the camera.
        2. Update the background model with the maximum pixel intensities seen so far.
        3. Perform background subtraction and morphological operations to get a clean foreground mask.
        4. Extract contours to find target positions.
        5. Use Kalman filters to predict target positions and assign measurements to filters.
        6. Update trajectories and compute raw and smoothed velocities, directions, and angles.
        7. Return current tracking estimates, or None if tracking not yet initialized.

        Parameters
        ----------
        return_camera_image : bool, optional
            If True, return the camera frame along with the tracking estimates.

        Returns
        -------
        estimates : list of dict or None
            A list of dictionaries with tracking information for each target, or None if not initialized.
        camera_frame : numpy.ndarray or None
            The current camera frame if requested, or None.

        """
        gray_frame = self.camera.get_array()
        if gray_frame is None:
            if return_camera_image:
                return None, None
            else:
                return None
        current_time = time.time()

        # Initialize the background model on the first frame
        if self.background_model_lightest is None:
            self.background_model_lightest = gray_frame.astype(np.float32)
            if return_camera_image:
                return None, gray_frame
            else:
                return None

        # Update the background model with max intensities
        np.maximum(self.background_model_lightest, gray_frame.astype(np.float32), out=self.background_model_lightest)

        # Convert background model to uint8
        background_model_lightest_uint8 = cv2.convertScaleAbs(self.background_model_lightest)

        # Foreground mask computation
        fg_mask = cv2.absdiff(background_model_lightest_uint8, gray_frame)
        _, fg_mask = cv2.threshold(fg_mask, self.threshold_value, 255, cv2.THRESH_BINARY)

        # Morphological operations to clean up noise
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel)

        # Create a conservative split mask
        split_mask = cv2.erode(fg_mask, self.kernel, iterations=2)

        # Find contours on both the original and conservative masks
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        conservative_contours, _ = cv2.findContours(split_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Process the main and conservative contours
        measurements = process_contours(contours, self.n_targets, self.min_contour_area, self.max_contour_area)

        # If not enough contours, try conservative approach if already initialized
        if self.detection_initialized and len(measurements) < self.n_targets:
            if self.debug:
                print("Splitting Targets")
            self.conservative_used = True
            measurements = process_contours(conservative_contours, self.n_targets, self.min_contour_area, self.max_contour_area)
        else:
            self.conservative_used = False

        # Check if all targets are detected for initialization
        if len(measurements) == self.n_targets:
            self.detection_counts += 1
        else:
            self.detection_counts = 0

        if self.detection_counts >= self.min_detection_counts:
            self.detection_initialized = True

        # If detection is initialized, update tracking
        if self.detection_initialized and measurements:
            predicted_positions = np.array([kf.predict()[:3] for kf in self.kalman_filters])  # (n_targets, 3)
            measurements_array = np.array(measurements)

            # Compute cost matrix
            diffs = predicted_positions[:, np.newaxis, :2] - measurements_array[np.newaxis, :, :2]
            position_costs = np.linalg.norm(diffs, axis=2)
            angle_diffs = np.abs(predicted_positions[:, np.newaxis, 2] - measurements_array[np.newaxis, :, 2])
            angle_diffs = np.minimum(angle_diffs, 2 * np.pi - angle_diffs)
            cost_matrix = position_costs + angle_diffs

            row_indices, col_indices = linear_sum_assignment(cost_matrix)

            avg_cost = 0
            for row, col in zip(row_indices, col_indices):
                if row < self.n_targets and col < len(measurements):
                    if not self.tracking_stabilized or cost_matrix[row, col] < self.max_distance_threshold:
                        measurement_vector = measurements_array[col].reshape(3, 1)
                        self.kalman_filters[row].correct(measurement_vector)
                        self.trajectories[row].append({
                            'x': measurements_array[col][0],
                            'y': measurements_array[col][1],
                            'theta': measurements_array[col][2],
                            'time': current_time
                        })
                        self.trajectories[row] = [
                            entry for entry in self.trajectories[row] if current_time - entry['time'] <= self.trajectory_history_seconds
                        ]
                        avg_cost += cost_matrix[row, col] / self.n_targets
                    else:
                        self.tracking_stabilized = False
                        self.tracking_counts = 0
                        self.trajectories[row].append({
                            'x': np.nan,
                            'y': np.nan,
                            'theta': np.nan,
                            'time': current_time
                        })
                        self.trajectories[row] = [
                            entry for entry in self.trajectories[row] if current_time - entry['time'] <= self.trajectory_history_seconds
                        ]

            if avg_cost < self.max_distance_threshold:
                self.tracking_counts += 1
            else:
                self.tracking_counts = 0

            if self.tracking_counts >= self.min_tracking_counts and not self.tracking_stabilized:
                self.tracking_stabilized = True
                if self.debug:
                    print(f"Tracking Stabilized with Average Cost: {avg_cost}")

        # Prepare estimates for return
        estimates = []
        if self.detection_initialized:
            for i, _ in enumerate(self.kalman_filters):
                if len(self.trajectories[i]) >= 2:
                    current_entry = self.trajectories[i][-1]
                    prev_entry = self.trajectories[i][-2]
                    delta_time = current_entry['time'] - prev_entry['time']

                    if delta_time > 0 and not np.isnan([current_entry['x'], current_entry['y'], prev_entry['x'], prev_entry['y']]).any():
                        dx = current_entry['x'] - prev_entry['x']
                        dy = current_entry['y'] - prev_entry['y']
                        vx_raw = dx / delta_time
                        vy_raw = dy / delta_time
                        v = np.sqrt(vx_raw**2 + vy_raw**2)
                        d = np.arctan2(vy_raw, vx_raw) if v > 0 else np.nan
                    else:
                        v = np.nan
                        d = np.nan
                        vx_raw = np.nan
                        vy_raw = np.nan

                    a = current_entry['theta']

                    current_entry['raw_velocity'] = v
                    current_entry['raw_direction'] = d
                    current_entry['raw_angle'] = a
                    current_entry['raw_vx'] = vx_raw
                    current_entry['raw_vy'] = vy_raw

                    prev_smoothed_vx = prev_entry.get('smoothed_vx', np.nan)
                    prev_smoothed_vy = prev_entry.get('smoothed_vy', np.nan)
                    alpha = self.smoothing_alpha

                    def safe_smooth(old_val, new_val, alpha):
                        if np.isnan(new_val):
                            return old_val
                        if np.isnan(old_val):
                            return new_val
                        return old_val * (1 - alpha) + new_val * alpha

                    smoothed_vx = safe_smooth(prev_smoothed_vx, vx_raw, alpha)
                    smoothed_vy = safe_smooth(prev_smoothed_vy, vy_raw, alpha)

                    if np.isnan(smoothed_vx) or np.isnan(smoothed_vy):
                        smoothed_velocity = np.nan
                        smoothed_direction = np.nan
                    else:
                        smoothed_velocity = np.sqrt(smoothed_vx**2 + smoothed_vy**2)
                        if smoothed_velocity > self.moving_threshold:
                            smoothed_direction = np.arctan2(smoothed_vy, smoothed_vx)
                        else:
                            smoothed_direction = prev_entry.get('smoothed_direction', np.nan)

                    prev_smoothed_angle = prev_entry.get('smoothed_angle', a)
                    smoothed_angle = safe_smooth(prev_smoothed_angle, a, alpha)
                    smoothed_angular_velocity = (smoothed_angle - prev_smoothed_angle) / delta_time

                    current_entry['smoothed_velocity'] = smoothed_velocity
                    current_entry['smoothed_direction'] = smoothed_direction
                    current_entry['smoothed_angle'] = smoothed_angle
                    current_entry['smoothed_vx'] = smoothed_vx
                    current_entry['smoothed_vy'] = smoothed_vy
                    current_entry['smoothed_angular_velocity'] = smoothed_angular_velocity

                    estimates.append({
                        'id': self.track_ids[i],
                        'position': (current_entry['x'], current_entry['y']),
                        'velocity': v,
                        'velocity_smooth': smoothed_velocity,
                        'angle': a,
                        'angle_smooth': smoothed_angle,
                        'direction': d,
                        'direction_smooth': smoothed_direction,
                        'angular_velocity_smooth': smoothed_angular_velocity,
                        'time_since_start': current_time - self.start_time
                    })
                else:
                    current_entry = self.trajectories[i][-1]
                    a = current_entry['theta']
                    current_entry['raw_velocity'] = np.nan
                    current_entry['raw_direction'] = np.nan
                    current_entry['raw_angle'] = a
                    current_entry['raw_vx'] = np.nan
                    current_entry['raw_vy'] = np.nan

                    current_entry['smoothed_velocity'] = np.nan
                    current_entry['smoothed_direction'] = np.nan
                    current_entry['smoothed_angle'] = a
                    current_entry['smoothed_vx'] = np.nan
                    current_entry['smoothed_vy'] = np.nan
                    current_entry['smoothed_angular_velocity'] = np.nan

                    estimates.append({
                        'id': self.track_ids[i],
                        'position': (current_entry['x'], current_entry['y']),
                        'velocity': np.nan,
                        'velocity_smooth': np.nan,
                        'angle': a,
                        'angle_smooth': a,
                        'direction': np.nan,
                        'direction_smooth': np.nan,
                        'angular_velocity_smooth': np.nan,
                        'time_since_start': current_time - self.start_time
                    })
        else:
            estimates = None

        self.frame_count += 1

        if self.debug:
            color_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
            self.display_frames(color_frame, fg_mask)

        if return_camera_image:
            return estimates, gray_frame
        else:
            return estimates

    def display_frames(self, color_frame, fg_mask):
        """
        Display debugging windows with visualizations of foreground mask, 
        detected targets, orientations, IDs, and trajectories.

        Parameters
        ----------
        color_frame : numpy.ndarray
            The current color frame for displaying detections and trajectories.
        fg_mask : numpy.ndarray
            The current foreground mask.
        """
        if self.detection_initialized:
            for i in range(self.n_targets):
                if len(self.trajectories[i]) > 0:
                    current_entry = self.trajectories[i][-1]
                    x, y, theta = current_entry['x'], current_entry['y'], current_entry['theta']
                    if not np.isnan([x, y]).any():
                        cv2.circle(color_frame, (int(x), int(y)), 10, self.trajectory_colors[i % len(self.trajectory_colors)], -1)
                        length = 20
                        x_end = int(x + length * np.cos(theta))
                        y_end = int(y + length * np.sin(theta))
                        cv2.line(color_frame, (int(x), int(y)), (x_end, y_end), self.trajectory_colors[i % len(self.trajectory_colors)], 2)
                        cv2.putText(color_frame, f"ID: {self.track_ids[i]}", (int(x) + 15, int(y) - 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                        for point_idx in range(1, len(self.trajectories[i])):
                            prev_entry = self.trajectories[i][point_idx - 1]
                            curr_entry = self.trajectories[i][point_idx]
                            if not np.isnan([prev_entry['x'], prev_entry['y'], curr_entry['x'], curr_entry['y']]).any():
                                pt1 = (int(prev_entry['x']), int(prev_entry['y']))
                                pt2 = (int(curr_entry['x']), int(curr_entry['y']))
                                cv2.line(color_frame, pt1, pt2, self.trajectory_colors[i % len(self.trajectory_colors)], self.trajectory_thickness)

        cv2.imshow('Foreground Mask', fg_mask)
        cv2.imshow('Blob Detection', color_frame)
        background_model_lightest_uint8 = cv2.convertScaleAbs(self.background_model_lightest)
        cv2.imshow('Lightest Background', cv2.cvtColor(background_model_lightest_uint8, cv2.COLOR_GRAY2BGR))
        cv2.waitKey(1)

    def release_resources(self):
        """
        Release OpenCV windows and any other allocated resources.
        """
        cv2.destroyAllWindows()

    def __enter__(self):
        """
        Enter the runtime context related to this object. 
        The object itself is returned and can be used as a context variable.

        Returns
        -------
        FastTracker
            The current instance of the FastTracker class.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the runtime context and perform cleanup. 
        If an exception occurred, it can be handled here if desired.

        Parameters
        ----------
        exc_type : Exception or None
            The exception type if raised, otherwise None.
        exc_val : Exception instance or None
            The exception value if raised, otherwise None.
        exc_tb : traceback or None
            The traceback if an exception was raised, otherwise None.

        Returns
        -------
        bool
            If True, the exception is suppressed; otherwise, it propagates.
        """
        self.release_resources()
        print("Successfully closed tracking resources. Exiting FastTracker context.")
