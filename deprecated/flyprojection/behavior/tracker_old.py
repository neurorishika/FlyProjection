import cv2
import numpy as np
import time
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment

class Tracker:
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
                 debug=False):
        """
        Initializes the Tracker class with all necessary parameters and objects.
        """
        # Parameters
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
        self.debug = debug

        # Initialize background model
        self.background_model_lightest = None

        self.detection_initialized = False  # Flag to start tracking after all targets are detected
        self.tracking_stabilized = False  # Flag to indicate that tracking is stabilized

        self.detection_counts = 0
        self.tracking_counts = 0
        self.conservative_used = False

        self.frame_count = 0
        self.start_time = time.time()
        self.time_since_start = 0

        self.trajectories = [[] for _ in range(self.n_targets)]  # Trajectories of all targets
        self.track_ids = np.arange(self.n_targets)  # IDs of all targets

        # Kalman Filters for tracking multiple targets
        self.kalman_filters = [cv2.KalmanFilter(5, 3) for _ in range(self.n_targets)]
        for i, kf in enumerate(self.kalman_filters):
            # Measurement matrix: Maps state to measurements (x, y, theta)
            kf.measurementMatrix = np.array([[1, 0, 0, 0, 0],  # x
                                              [0, 1, 0, 0, 0],  # y
                                              [0, 0, 1, 0, 0]], # theta
                                             np.float32)
            
            # Transition matrix: Defines how the state evolves
            kf.transitionMatrix = np.array([[1, 0, 0, 1, 0],  # x -> x + vx
                                             [0, 1, 0, 0, 1],  # y -> y + vy
                                             [0, 0, 1, 0, 0],  # theta -> theta
                                             [0, 0, 0, 1, 0],  # vx -> vx
                                             [0, 0, 0, 0, 1]], # vy -> vy
                                            np.float32)
            
            # Process noise covariance: Uncertainty in the model's predictions
            kf.processNoiseCov = np.eye(5, dtype=np.float32) * self.kalman_noise_covariance

            # Measurement noise covariance: Uncertainty in the measurements
            kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * self.kalman_measurement_noise_covariance

            # Initial state vector: Random initial positions and velocities
            kf.statePre = np.array([np.random.randint(0, self.camera.WIDTH), np.random.randint(0, self.camera.HEIGHT), 0, 0, 0], np.float32)  # x, y, theta, vx, vy
            kf.statePost = kf.statePre.copy()

            # Error covariance matrix: Initial uncertainty in the state
            kf.errorCovPre = np.eye(5, dtype=np.float32)
        
        # Colors for trajectories
        self.trajectory_colors = [tuple(color) for color in np.random.randint(0, 255, (self.n_targets, 3)).tolist()]  # Random colors for trajectories

        if self.debug:
            cv2.namedWindow('Foreground Mask', cv2.WINDOW_NORMAL)
            cv2.namedWindow('Blob Detection', cv2.WINDOW_NORMAL)
            cv2.namedWindow('Lightest Background', cv2.WINDOW_NORMAL)

            # resize the windows
            cv2.resizeWindow('Foreground Mask', 500, 500)
            cv2.resizeWindow('Blob Detection', 500, 500)
            cv2.resizeWindow('Lightest Background', 500, 500)

    def process_next_frame(self):
        """
        Processes the next frame from the camera, performs tracking, and returns updated estimates for position, velocity, and angle.
        """
        # Capture frame-by-frame
        gray_frame = self.camera.get_array()
        if gray_frame is None:
            return None
        current_time = time.time()
        
        # Convert gray_frame to RGB for visualization purposes
        color_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
        
        # Initialize lightest background model with the first frame
        if self.background_model_lightest is None:
            self.background_model_lightest = gray_frame.astype(np.float32)
            return None  # Need at least one frame to initialize background

        # Update lightest background model by keeping the lightest pixels
        self.background_model_lightest = np.maximum(self.background_model_lightest, gray_frame.astype(np.float32))
        background_model_lightest_uint8 = cv2.convertScaleAbs(self.background_model_lightest)

        # Background Subtraction to get foreground mask
        fg_mask = cv2.absdiff(background_model_lightest_uint8, gray_frame)
        _, fg_mask = cv2.threshold(fg_mask, self.threshold_value, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_kernel_size, self.morph_kernel_size))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        # Create a conservative version of the foreground mask to split the targets if needed
        split_mask = cv2.erode(fg_mask, kernel, iterations=2)
        
        # Find contours of the foreground objects
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        conservative_contours, _ = cv2.findContours(split_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Fit ellipses to detected contours and track them, keeping only the largest contours
        measurements = []
        sizes = []

        if len(contours) > 0:
            for cnt in contours:
                # Filter by contour area
                size = cv2.contourArea(cnt)
                if size < self.min_contour_area or size > self.max_contour_area:
                    continue
                
                if len(cnt) >= 5:  # Minimum points to fit an ellipse
                    ellipse = cv2.fitEllipse(cnt)
                    (x, y), (MA, ma), angle = ellipse
                    angle_radians = np.deg2rad(angle)  # Convert angle to radians
                    measurements.append(np.array([np.float32(x), np.float32(y), np.float32(angle_radians)]))
                    sizes.append(size)
                    
            # Keep only the largest contours
            if len(measurements) > self.n_targets:
                sorted_indices = np.argsort(sizes)[::-1]
                measurements = [measurements[i] for i in sorted_indices[:self.n_targets]]
            
        # If the number of contours is less than the number of targets, use conservative mask
        if self.detection_initialized and len(measurements) < self.n_targets:
            if self.debug:
                print("Splitting Targets")
            self.conservative_used = True
            measurements = []
            sizes = []
            # Use the conservative mask instead
            for cnt in conservative_contours:
                # Filter by contour area
                size = cv2.contourArea(cnt)
                if size < self.min_contour_area or size > self.max_contour_area:
                    continue
                
                if len(cnt) >= 5:
                    ellipse = cv2.fitEllipse(cnt)
                    (x, y), (MA, ma), angle = ellipse
                    angle_radians = np.deg2rad(angle)
                    measurements.append(np.array([np.float32(x), np.float32(y), np.float32(angle_radians)]))
                    sizes.append(size)

                # Keep only the largest contours
                if len(measurements) > self.n_targets:
                    sorted_indices = np.argsort(sizes)[::-1]
                    measurements = [measurements[i] for i in sorted_indices[:self.n_targets]]
        else:
            self.conservative_used = False

        # Start tracking only if all targets are detected
        if len(measurements) == self.n_targets:
            self.detection_counts += 1
        else:
            self.detection_counts = 0

        if self.detection_counts >= self.min_detection_counts:
            self.detection_initialized = True

        if self.detection_initialized and measurements:
            predicted_positions = np.zeros((self.n_targets, 3))  # Predicted positions of all Kalman filters
            # Predict next positions of all Kalman filters
            for i, kf in enumerate(self.kalman_filters):
                predicted = kf.predict()  # Now a 5-element vector: [x, y, theta, vx, vy]
                predicted_positions[i] = predicted[:3]  # Extract x, y, and theta

            # Create cost matrix for Hungarian algorithm
            cost_matrix = np.zeros((self.n_targets, len(measurements)))
            for i, predicted_pos in enumerate(predicted_positions):
                for j, measurement in enumerate(measurements):
                    position_cost = distance.euclidean(predicted_pos[:2], measurement[:2])  # x, y
                    angle_diff = np.abs(predicted_pos[2] - measurement[2])  # θ
                    angle_diff = min(angle_diff, 2 * np.pi - angle_diff)  # Ensure shortest angular distance
                    angle_cost = angle_diff
                    cost_matrix[i, j] = position_cost + angle_cost  # Combine position and angle costs

            # Solve assignment problem using Hungarian algorithm
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            # Assign measurements to Kalman filters
            avg_cost = 0
            for row, col in zip(row_indices, col_indices):
                if row < self.n_targets and col < len(measurements):
                    # Check if the predicted position is close to the measurement
                    if not self.tracking_stabilized or cost_matrix[row, col] < self.max_distance_threshold:
                        # Correct the Kalman filter with the measurement
                        measurement_vector = np.array([[measurements[col][0]],  # x
                                                        [measurements[col][1]],  # y
                                                        [measurements[col][2]]], # theta
                                                    np.float32)
                        self.kalman_filters[row].correct(measurement_vector)
                        # Update the trajectory for the assigned Kalman filter
                        self.trajectories[row].append({
                            'x': measurements[col][0],
                            'y': measurements[col][1],
                            'theta': measurements[col][2],
                            'time': current_time
                        })
                        # Keep only the last TRAJECTORY_HISTORY_SECONDS seconds of trajectory
                        self.trajectories[row] = [entry for entry in self.trajectories[row] if current_time - entry['time'] <= self.trajectory_history_seconds]
                        avg_cost += cost_matrix[row, col]/self.n_targets
                    else:
                        self.tracking_stabilized = False
                        self.tracking_counts = 0
                        # Update the trajectory for the assigned Kalman filter with NaN values
                        self.trajectories[row].append({
                            'x': np.nan,
                            'y': np.nan,
                            'theta': np.nan,
                            'time': current_time
                        })
                        # Keep only the last TRAJECTORY_HISTORY_SECONDS seconds of trajectory
                        self.trajectories[row] = [entry for entry in self.trajectories[row] if current_time - entry['time'] <= self.trajectory_history_seconds]
                        
            # Check if the tracking is stabilized
            if avg_cost < self.max_distance_threshold:
                self.tracking_counts += 1
            else:
                self.tracking_counts = 0

            if self.tracking_counts >= self.min_tracking_counts and not self.tracking_stabilized:
                self.tracking_stabilized = True
                if self.debug:
                    print(f"Tracking Stabilized with Average Cost: {avg_cost}")
        
        # Prepare estimates to return
        estimates = []
        if self.detection_initialized:
            for i, _ in enumerate(self.kalman_filters):
                if len(self.trajectories[i]) >= 2:
                    current_entry = self.trajectories[i][-1]
                    prev_entry = self.trajectories[i][-2]
                    delta_time = current_entry['time'] - prev_entry['time']

                    # Compute raw velocity and direction
                    if delta_time > 0 and not np.isnan([current_entry['x'], current_entry['y'], prev_entry['x'], prev_entry['y']]).any():
                        v = distance.euclidean(
                            [current_entry['x'], current_entry['y']],
                            [prev_entry['x'], prev_entry['y']]
                        ) / delta_time

                        dx = current_entry['x'] - prev_entry['x']
                        dy = current_entry['y'] - prev_entry['y']
                        d = np.arctan2(dy, dx) if (dx != 0 or dy != 0) else np.nan
                    else:
                        v = np.nan
                        d = np.nan

                    # Raw angle from ellipse fit
                    a = current_entry['theta']

                    # Store raw values in the current entry
                    current_entry['raw_velocity'] = v
                    current_entry['raw_direction'] = d
                    current_entry['raw_angle'] = a

                    # Retrieve previous smoothed values or initialize them
                    prev_smoothed_velocity = prev_entry.get('smoothed_velocity', v)
                    prev_smoothed_direction = prev_entry.get('smoothed_direction', d)
                    prev_smoothed_angle = prev_entry.get('smoothed_angle', a)

                    alpha = self.smoothing_alpha

                    def smooth(old, new):
                        if np.isnan(new):
                            return old
                        elif np.isnan(old):
                            return new
                        else:
                            return alpha * new + (1 - alpha) * old

                    # Compute smoothed values
                    smoothed_velocity = smooth(prev_smoothed_velocity, v)
                    smoothed_direction = smooth(prev_smoothed_direction, d)
                    smoothed_angle = smooth(prev_smoothed_angle, a)

                    # Store smoothed values in the current entry
                    current_entry['smoothed_velocity'] = smoothed_velocity
                    current_entry['smoothed_direction'] = smoothed_direction
                    current_entry['smoothed_angle'] = smoothed_angle

                    # Return both raw and smoothed values in estimates
                    estimates.append({
                        'id': self.track_ids[i],
                        'position': (current_entry['x'], current_entry['y']),
                        'velocity': v,
                        'velocity_smooth': smoothed_velocity,
                        'angle': a,
                        'angle_smooth': smoothed_angle,
                        'direction': d,
                        'direction_smooth': smoothed_direction,
                        'time_since_start': current_time - self.start_time
                    })
                else:
                    # Not enough data to compute velocity and direction yet
                    current_entry = self.trajectories[i][-1]
                    a = current_entry['theta']

                    # Since this is the first or second point, we have no previous smoothed values
                    current_entry['raw_velocity'] = np.nan
                    current_entry['raw_direction'] = np.nan
                    current_entry['raw_angle'] = a

                    # Just set smoothed values to raw (or nan if raw is nan)
                    current_entry['smoothed_velocity'] = np.nan
                    current_entry['smoothed_direction'] = np.nan
                    current_entry['smoothed_angle'] = a

                    estimates.append({
                        'id': self.track_ids[i],
                        'position': (current_entry['x'], current_entry['y']),
                        'velocity': np.nan,
                        'velocity_smooth': np.nan,
                        'angle': a,
                        'angle_smooth': a,
                        'direction': np.nan,
                        'direction_smooth': np.nan,
                        'time_since_start': current_time - self.start_time
                    })
        else:
            estimates = None

        self.frame_count += 1

        # Optionally, display frames for debugging
        if self.debug:
            self.display_frames(gray_frame, color_frame, fg_mask, current_time)

        return estimates

    def display_frames(self, gray_frame, color_frame, fg_mask, current_time):
        """
        Displays the frames for debugging purposes.
        """
        # Draw tracked objects, IDs, and trajectories if tracking has started
        if self.detection_initialized:
            for i, _ in enumerate(self.kalman_filters):
                if len(self.trajectories[i]) > 0:
                    current_entry = self.trajectories[i][-1]
                    x, y, theta = current_entry['x'], current_entry['y'], current_entry['theta']
                    if not np.isnan(x) and not np.isnan(y):
                        # Draw the circle representing the object
                        cv2.circle(color_frame, (int(x), int(y)), 10, self.trajectory_colors[i % len(self.trajectory_colors)], -1)
                        # Draw orientation line
                        length = 20  # Length of orientation line
                        x_end = int(x + length * np.cos(theta))
                        y_end = int(y + length * np.sin(theta))
                        cv2.line(color_frame, (int(x), int(y)), (x_end, y_end), self.trajectory_colors[i % len(self.trajectory_colors)], 2)
                        # Display the ID of the tracked object
                        cv2.putText(color_frame, f"ID: {self.track_ids[i]}", (int(x) + 15, int(y) - 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                        # Draw the trajectory of each target
                        for point_idx in range(1, len(self.trajectories[i])):
                            if all([not np.isnan(self.trajectories[i][point_idx - 1]['x']), not np.isnan(self.trajectories[i][point_idx - 1]['y']),
                                    not np.isnan(self.trajectories[i][point_idx]['x']), not np.isnan(self.trajectories[i][point_idx]['y'])]):
                                pt1 = (int(self.trajectories[i][point_idx - 1]['x']), int(self.trajectories[i][point_idx - 1]['y']))
                                pt2 = (int(self.trajectories[i][point_idx]['x']), int(self.trajectories[i][point_idx]['y']))
                                cv2.line(color_frame, pt1, pt2, self.trajectory_colors[i % len(self.trajectory_colors)], self.trajectory_thickness)

        # Display the resulting frames
        cv2.imshow('Foreground Mask', fg_mask)
        cv2.imshow('Blob Detection', color_frame)
        background_model_lightest_uint8 = cv2.convertScaleAbs(self.background_model_lightest)
        cv2.imshow('Lightest Background', cv2.cvtColor(background_model_lightest_uint8, cv2.COLOR_GRAY2BGR))
        cv2.waitKey(1)  # Display images

    def release_resources(self):
        """
        Releases resources like OpenCV windows.
        """
        cv2.destroyAllWindows()

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
                 debug=False):
        """
        Initializes the Tracker class with all necessary parameters and objects.
        """
        # Parameters
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
        self.debug = debug

        # Initialize background model
        self.background_model_lightest = None

        self.detection_initialized = False  # Flag to start tracking after all targets are detected
        self.tracking_stabilized = False  # Flag to indicate that tracking is stabilized

        self.detection_counts = 0
        self.tracking_counts = 0
        self.conservative_used = False

        self.frame_count = 0
        self.start_time = time.time()

        self.trajectories = [[] for _ in range(self.n_targets)]  # Trajectories of all targets
        self.track_ids = np.arange(self.n_targets)  # IDs of all targets

        # Precompute morphological kernel
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_kernel_size, self.morph_kernel_size))

        # Kalman Filters for tracking multiple targets
        self.kalman_filters = [cv2.KalmanFilter(5, 3) for _ in range(self.n_targets)]
        for i, kf in enumerate(self.kalman_filters):
            # Measurement matrix: Maps state to measurements (x, y, theta)
            kf.measurementMatrix = np.array([[1, 0, 0, 0, 0],  # x
                                              [0, 1, 0, 0, 0],  # y
                                              [0, 0, 1, 0, 0]], # theta
                                             np.float32)
            
            # Transition matrix: Defines how the state evolves
            kf.transitionMatrix = np.array([[1, 0, 0, 1, 0],  # x -> x + vx
                                             [0, 1, 0, 0, 1],  # y -> y + vy
                                             [0, 0, 1, 0, 0],  # theta -> theta
                                             [0, 0, 0, 1, 0],  # vx -> vx
                                             [0, 0, 0, 0, 1]], # vy -> vy
                                            np.float32)
            
            # Process noise covariance: Uncertainty in the model's predictions
            kf.processNoiseCov = np.eye(5, dtype=np.float32) * self.kalman_noise_covariance

            # Measurement noise covariance: Uncertainty in the measurements
            kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * self.kalman_measurement_noise_covariance

            # Initial state vector: Random initial positions and velocities
            kf.statePre = np.array([np.random.randint(0, self.camera.WIDTH), np.random.randint(0, self.camera.HEIGHT), 0, 0, 0], np.float32)  # x, y, theta, vx, vy
            kf.statePost = kf.statePre.copy()

            # Error covariance matrix: Initial uncertainty in the state
            kf.errorCovPre = np.eye(5, dtype=np.float32)
        
        # Colors for trajectories
        self.trajectory_colors = [tuple(color) for color in np.random.randint(0, 255, (self.n_targets, 3)).tolist()]  # Random colors for trajectories

        if self.debug:
            cv2.namedWindow('Foreground Mask', cv2.WINDOW_NORMAL)
            cv2.namedWindow('Blob Detection', cv2.WINDOW_NORMAL)
            cv2.namedWindow('Lightest Background', cv2.WINDOW_NORMAL)

            # Resize the windows
            cv2.resizeWindow('Foreground Mask', 500, 500)
            cv2.resizeWindow('Blob Detection', 500, 500)
            cv2.resizeWindow('Lightest Background', 500, 500)

    def process_next_frame(self):
        """
        Processes the next frame from the camera, performs tracking, and returns updated estimates for position, velocity, and angle.
        """
        # Capture frame-by-frame
        gray_frame = self.camera.get_array()
        if gray_frame is None:
            return None
        current_time = time.time()

        # Initialize lightest background model with the first frame
        if self.background_model_lightest is None:
            self.background_model_lightest = gray_frame.astype(np.float32)
            return None  # Need at least one frame to initialize background

        # Update lightest background model by keeping the lightest pixels (in-place operation)
        np.maximum(self.background_model_lightest, gray_frame.astype(np.float32), out=self.background_model_lightest)

        # Convert to uint8 for further processing
        background_model_lightest_uint8 = cv2.convertScaleAbs(self.background_model_lightest)

        # Background Subtraction to get foreground mask
        fg_mask = cv2.absdiff(background_model_lightest_uint8, gray_frame)
        _, fg_mask = cv2.threshold(fg_mask, self.threshold_value, 255, cv2.THRESH_BINARY)

        # Apply morphological operations to reduce noise using precomputed kernel
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel)

        # Create a conservative version of the foreground mask to split the targets if needed
        split_mask = cv2.erode(fg_mask, self.kernel, iterations=2)

        # Find contours of the foreground objects
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        conservative_contours, _ = cv2.findContours(split_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Function to process contours
        def process_contours(contours_list):
            measurements = []
            sizes = []
            for cnt in contours_list:
                # Filter by contour area
                size = cv2.contourArea(cnt)
                if size < self.min_contour_area or size > self.max_contour_area:
                    continue

                if len(cnt) >= 5:  # Minimum points to fit an ellipse
                    ellipse = cv2.fitEllipse(cnt)
                    (x, y), _, angle = ellipse
                    angle_radians = np.deg2rad(angle)  # Convert angle to radians
                    measurements.append(np.array([x, y, angle_radians], dtype=np.float32))
                    sizes.append(size)

            # Keep only the largest contours
            if len(measurements) > self.n_targets:
                sorted_indices = np.argsort(sizes)[::-1][:self.n_targets]
                measurements = [measurements[i] for i in sorted_indices]
            return measurements

        # Process contours
        measurements = process_contours(contours)

        # If the number of contours is less than the number of targets, use conservative mask
        if self.detection_initialized and len(measurements) < self.n_targets:
            if self.debug:
                print("Splitting Targets")
            self.conservative_used = True
            measurements = process_contours(conservative_contours)
        else:
            self.conservative_used = False

        # Start tracking only if all targets are detected
        if len(measurements) == self.n_targets:
            self.detection_counts += 1
        else:
            self.detection_counts = 0

        if self.detection_counts >= self.min_detection_counts:
            self.detection_initialized = True

        if self.detection_initialized and measurements:
            # Predict next positions of all Kalman filters
            predicted_positions = np.array([kf.predict()[:3] for kf in self.kalman_filters])  # Shape: (n_targets, 3)

            # Convert measurements to numpy array
            measurements_array = np.array(measurements)  # Shape: (n_measurements, 3)

            # Vectorized cost matrix computation
            diffs = predicted_positions[:, np.newaxis, :2] - measurements_array[np.newaxis, :, :2]
            position_costs = np.linalg.norm(diffs, axis=2)

            angle_diffs = np.abs(predicted_positions[:, np.newaxis, 2] - measurements_array[np.newaxis, :, 2])
            angle_diffs = np.minimum(angle_diffs, 2 * np.pi - angle_diffs)
            cost_matrix = position_costs + angle_diffs  # Combine position and angle costs

            # Solve assignment problem using Hungarian algorithm
            row_indices, col_indices = linear_sum_assignment(cost_matrix)

            # Assign measurements to Kalman filters
            avg_cost = 0
            for row, col in zip(row_indices, col_indices):
                if row < self.n_targets and col < len(measurements):
                    # Check if the predicted position is close to the measurement
                    if not self.tracking_stabilized or cost_matrix[row, col] < self.max_distance_threshold:
                        # Correct the Kalman filter with the measurement
                        measurement_vector = measurements_array[col].reshape(3, 1)
                        self.kalman_filters[row].correct(measurement_vector)
                        # Update the trajectory for the assigned Kalman filter
                        self.trajectories[row].append({
                            'x': measurements_array[col][0],
                            'y': measurements_array[col][1],
                            'theta': measurements_array[col][2],
                            'time': current_time
                        })
                        # Keep only the last TRAJECTORY_HISTORY_SECONDS seconds of trajectory
                        self.trajectories[row] = [
                            entry for entry in self.trajectories[row] if current_time - entry['time'] <= self.trajectory_history_seconds
                        ]
                        avg_cost += cost_matrix[row, col] / self.n_targets
                    else:
                        self.tracking_stabilized = False
                        self.tracking_counts = 0
                        # Update the trajectory for the assigned Kalman filter with NaN values
                        self.trajectories[row].append({
                            'x': np.nan,
                            'y': np.nan,
                            'theta': np.nan,
                            'time': current_time
                        })
                        # Keep only the last TRAJECTORY_HISTORY_SECONDS seconds of trajectory
                        self.trajectories[row] = [
                            entry for entry in self.trajectories[row] if current_time - entry['time'] <= self.trajectory_history_seconds
                        ]
            # Check if the tracking is stabilized
            if avg_cost < self.max_distance_threshold:
                self.tracking_counts += 1
            else:
                self.tracking_counts = 0

            if self.tracking_counts >= self.min_tracking_counts and not self.tracking_stabilized:
                self.tracking_stabilized = True
                if self.debug:
                    print(f"Tracking Stabilized with Average Cost: {avg_cost}")

        # Prepare estimates to return
        estimates = []
        if self.detection_initialized:
            for i, _ in enumerate(self.kalman_filters):
                if len(self.trajectories[i]) >= 2:
                    current_entry = self.trajectories[i][-1]
                    prev_entry = self.trajectories[i][-2]
                    delta_time = current_entry['time'] - prev_entry['time']

                    # Compute raw velocity and direction
                    if delta_time > 0 and not np.isnan([current_entry['x'], current_entry['y'], prev_entry['x'], prev_entry['y']]).any():
                        v = distance.euclidean(
                            [current_entry['x'], current_entry['y']],
                            [prev_entry['x'], prev_entry['y']]
                        ) / delta_time

                        dx = current_entry['x'] - prev_entry['x']
                        dy = current_entry['y'] - prev_entry['y']
                        d = np.arctan2(dy, dx) if (dx != 0 or dy != 0) else np.nan
                    else:
                        v = np.nan
                        d = np.nan

                    # Raw angle from ellipse fit
                    a = current_entry['theta']

                    # Store raw values in the current entry
                    current_entry['raw_velocity'] = v
                    current_entry['raw_direction'] = d
                    current_entry['raw_angle'] = a

                    # Retrieve previous smoothed values or initialize them
                    prev_smoothed_velocity = prev_entry.get('smoothed_velocity', v)
                    prev_smoothed_direction = prev_entry.get('smoothed_direction', d)
                    prev_smoothed_angle = prev_entry.get('smoothed_angle', a)

                    alpha = self.smoothing_alpha

                    def smooth(old, new):
                        if np.isnan(new):
                            return old
                        elif np.isnan(old):
                            return new
                        else:
                            return alpha * new + (1 - alpha) * old

                    # Compute smoothed values
                    smoothed_velocity = smooth(prev_smoothed_velocity, v)
                    smoothed_direction = smooth(prev_smoothed_direction, d)
                    smoothed_angle = smooth(prev_smoothed_angle, a)

                    # Store smoothed values in the current entry
                    current_entry['smoothed_velocity'] = smoothed_velocity
                    current_entry['smoothed_direction'] = smoothed_direction
                    current_entry['smoothed_angle'] = smoothed_angle

                    # Return both raw and smoothed values in estimates
                    estimates.append({
                        'id': self.track_ids[i],
                        'position': (current_entry['x'], current_entry['y']),
                        'velocity': v,
                        'velocity_smooth': smoothed_velocity,
                        'angle': a,
                        'angle_smooth': smoothed_angle,
                        'direction': d,
                        'direction_smooth': smoothed_direction,
                        'time_since_start': current_time - self.start_time
                    })
                else:
                    # Not enough data to compute velocity and direction yet
                    current_entry = self.trajectories[i][-1]
                    a = current_entry['theta']

                    # Since this is the first or second point, we have no previous smoothed values
                    current_entry['raw_velocity'] = np.nan
                    current_entry['raw_direction'] = np.nan
                    current_entry['raw_angle'] = a

                    # Just set smoothed values to raw (or nan if raw is nan)
                    current_entry['smoothed_velocity'] = np.nan
                    current_entry['smoothed_direction'] = np.nan
                    current_entry['smoothed_angle'] = a

                    estimates.append({
                        'id': self.track_ids[i],
                        'position': (current_entry['x'], current_entry['y']),
                        'velocity': np.nan,
                        'velocity_smooth': np.nan,
                        'angle': a,
                        'angle_smooth': a,
                        'direction': np.nan,
                        'direction_smooth': np.nan,
                        'time_since_start': current_time - self.start_time
                    })
        else:
            estimates = None

        self.frame_count += 1

        # Optionally, display frames for debugging
        if self.debug:
            # Convert gray_frame to RGB for visualization purposes (only when debugging)
            color_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
            self.display_frames(color_frame, fg_mask)

        return estimates

    def display_frames(self, color_frame, fg_mask):
        """
        Displays the frames for debugging purposes.
        """
        # Draw tracked objects, IDs, and trajectories if tracking has started
        if self.detection_initialized:
            for i in range(self.n_targets):
                if len(self.trajectories[i]) > 0:
                    current_entry = self.trajectories[i][-1]
                    x, y, theta = current_entry['x'], current_entry['y'], current_entry['theta']
                    if not np.isnan([x, y]).any():
                        # Draw the circle representing the object
                        cv2.circle(color_frame, (int(x), int(y)), 10, self.trajectory_colors[i % len(self.trajectory_colors)], -1)
                        # Draw orientation line
                        length = 20  # Length of orientation line
                        x_end = int(x + length * np.cos(theta))
                        y_end = int(y + length * np.sin(theta))
                        cv2.line(color_frame, (int(x), int(y)), (x_end, y_end), self.trajectory_colors[i % len(self.trajectory_colors)], 2)
                        # Display the ID of the tracked object
                        cv2.putText(color_frame, f"ID: {self.track_ids[i]}", (int(x) + 15, int(y) - 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                        # Draw the trajectory of each target
                        for point_idx in range(1, len(self.trajectories[i])):
                            prev_entry = self.trajectories[i][point_idx - 1]
                            curr_entry = self.trajectories[i][point_idx]
                            if not np.isnan([prev_entry['x'], prev_entry['y'], curr_entry['x'], curr_entry['y']]).any():
                                pt1 = (int(prev_entry['x']), int(prev_entry['y']))
                                pt2 = (int(curr_entry['x']), int(curr_entry['y']))
                                cv2.line(color_frame, pt1, pt2, self.trajectory_colors[i % len(self.trajectory_colors)], self.trajectory_thickness)

        # Display the resulting frames
        cv2.imshow('Foreground Mask', fg_mask)
        cv2.imshow('Blob Detection', color_frame)
        background_model_lightest_uint8 = cv2.convertScaleAbs(self.background_model_lightest)
        cv2.imshow('Lightest Background', cv2.cvtColor(background_model_lightest_uint8, cv2.COLOR_GRAY2BGR))
        cv2.waitKey(1)  # Display images

    def release_resources(self):
        """
        Releases resources like OpenCV windows.
        """
        cv2.destroyAllWindows()
