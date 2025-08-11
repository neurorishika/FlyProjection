"""
Production-Ready PySide2 Script for Multi-Fly Tracking with:
1) Random-frame background priming (lightest-pixel)
2) Optional Otsu-based two-region split in the primed background
3) Local "conservative" morphological splitting
4) Kalman + Hungarian multi-object tracking
5) Real-time CSV writing
6) Pruned vs. full trajectories
7) Zoom in/out (QScrollArea + "Zoom Factor")
8) Memory-safe design
9) Configuration Persistence (tracking_config.json)
10) Orientation logic:
    - If speed < VELOCITY_THRESHOLD, allow small orientation changes up to ±MAX_ORIENT_DELTA_STOPPED (deg) from last orientation;
      if the change is >90°, force an immediate 180° flip.
    - If speed >= VELOCITY_THRESHOLD and "Instant Flip Orientation?" is enabled, compare motion heading with the ellipse angle;
      if the difference >90°, flip by 180°.
    - Additionally, the cost function now includes shape information (area and aspect ratio) from the ellipse fit.
       (Position cost can be computed via Mahalanobis distance if enabled.)
       
Usage:
------
1) pip install pyside2 opencv-python numpy scipy matplotlib
2) python multi_fly_tracking.py
3) In the UI, all parameters (detection, tracking, background, orientation, shape cost weights, etc.) are configurable.
   The configuration is saved/loaded from tracking_config.json.
"""

import sys, time, gc, csv, random, queue, threading, json, os
import numpy as np, cv2
from collections import deque
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

from PySide2.QtCore import Qt, QThread, Signal, Slot, QMutex
from PySide2.QtGui import QImage, QPixmap
from PySide2.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QPushButton,
                               QVBoxLayout, QHBoxLayout, QFileDialog, QSpinBox, QDoubleSpinBox,
                               QCheckBox, QMessageBox, QGroupBox, QFormLayout, QLineEdit, QScrollArea)

CONFIG_FILENAME = "tracking_config.json"

def wrap_angle_degs(deg: float) -> float:
    """Wrap an angle in degrees to the range [-180, 180)."""
    deg = deg % 360
    if deg >= 180:
        deg -= 360
    return deg

class CSVWriterThread(threading.Thread):
    """A CSV-writing thread that receives row data via a queue and writes asynchronously."""
    def __init__(self, csv_path: str, header=None):
        super().__init__()
        self.csv_path = csv_path
        self.header = header or []
        self.queue = queue.Queue()
        self._stop_flag = False
        self.csv_file = open(self.csv_path, "w", newline="")
        self.writer = csv.writer(self.csv_file)
        if self.header:
            self.writer.writerow(self.header)
    def run(self):
        while not self._stop_flag:
            try:
                row = self.queue.get(timeout=0.3)
                self.writer.writerow(row)
                self.queue.task_done()
            except queue.Empty:
                pass
        self.csv_file.flush()
        self.csv_file.close()
    def enqueue(self, row_data):
        self.queue.put(row_data)
    def stop(self):
        self._stop_flag = True

def apply_image_adjustments(gray_frame: np.ndarray, brightness: float, contrast: float, gamma: float) -> np.ndarray:
    """Apply brightness, contrast, and gamma corrections to a grayscale image."""
    adjusted = cv2.convertScaleAbs(gray_frame, alpha=contrast, beta=brightness)
    if abs(gamma - 1.0) > 1e-3:
        lut = np.array([np.clip(((i / 255.0) ** (1.0 / gamma)) * 255.0, 0, 255) for i in range(256)], dtype=np.uint8)
        adjusted = cv2.LUT(adjusted, lut)
    return adjusted

class TrackingWorker(QThread):
    frame_signal = Signal(np.ndarray)
    finished_signal = Signal(bool, list, list)  # (finished_normally, fps_list, final_trajectories_full)
    def __init__(self, video_path: str, csv_writer_thread=None, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.csv_writer_thread = csv_writer_thread
        self.params_mutex = QMutex()
        self.parameters = {}
        self._stop_flag = False
        self.cap = None
        # Tracking structures
        self.kalman_filters = []
        self.track_ids = []
        self.trajectories_pruned = []
        self.trajectories_full = []
        self.position_deques = []
        self.orientation_last = []
        self.last_shape_info = []  # (area, aspect_ratio) for each target
        # Background/detection
        self.background_model_lightest = None
        self.region_mask = None
        self.detection_initialized = False
        self.tracking_stabilized = False
        self.detection_counts = 0
        self.tracking_counts = 0
        # Stats
        self.fps_list = []
        self.frame_count = 0
        self.start_time = 0

    def set_parameters(self, new_params: dict):
        self.params_mutex.lock()
        self.parameters = new_params
        self.params_mutex.unlock()
    def get_current_params(self):
        self.params_mutex.lock()
        p = dict(self.parameters)
        self.params_mutex.unlock()
        return p
    def stop(self):
        self._stop_flag = True
    def init_kalman_filters(self, p):
        kf_list = []
        for _ in range(p["MAX_TARGETS"]):
            kf = cv2.KalmanFilter(5, 3)
            kf.measurementMatrix = np.array([[1,0,0,0,0],
                                             [0,1,0,0,0],
                                             [0,0,1,0,0]], np.float32)
            kf.transitionMatrix = np.array([[1,0,0,1,0],
                                            [0,1,0,0,1],
                                            [0,0,1,0,0],
                                            [0,0,0,1,0],
                                            [0,0,0,0,1]], np.float32)
            kf.processNoiseCov = np.eye(5, dtype=np.float32)*p["KALMAN_NOISE_COVARIANCE"]
            kf.measurementNoiseCov = np.eye(3, dtype=np.float32)*p["KALMAN_MEASUREMENT_NOISE_COVARIANCE"]
            kf.errorCovPre = np.eye(5, dtype=np.float32)
            kf_list.append(kf)
        return kf_list
    def prime_lightest_background(self, cap, p):
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            print("[WARNING] No total_frames, skipping prime.")
            return
        prime_count = p.get("BACKGROUND_PRIME_FRAMES", 10)
        prime_count = min(prime_count, total_frames)
        if prime_count < 1:
            return
        br, ct, gm = p["BRIGHTNESS"], p["CONTRAST"], p["GAMMA"]
        idxs = random.sample(range(total_frames), prime_count)
        bg_temp = None
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = apply_image_adjustments(gray, br, ct, gm)
            if bg_temp is None:
                bg_temp = gray.astype(np.float32)
            else:
                bg_temp = np.maximum(bg_temp, gray.astype(np.float32))
        if bg_temp is not None:
            self.background_model_lightest = bg_temp
            print(f"[INFO] Background primed from {prime_count} frames.")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if p.get("USE_OTSU_SPLIT", False):
            bg_uint8 = cv2.convertScaleAbs(bg_temp)
            retval, _ = cv2.threshold(bg_uint8, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            split_val = retval
            print(f"[INFO] Otsu determined BG_SPLIT_VALUE={split_val:.2f}")
            if split_val <= 0:
                print("[WARNING] Otsu result <=0, skipping region split.")
                self.region_mask = None
            else:
                self.create_region_mask(bg_uint8, split_val)
        else:
            split_val = p.get("BG_SPLIT_VALUE", 0)
            if split_val > 0:
                bg_uint8 = cv2.convertScaleAbs(bg_temp)
                self.create_region_mask(bg_uint8, split_val)
            else:
                self.region_mask = None
    def create_region_mask(self, bg_uint8: np.ndarray, split_val: float):
        self.region_mask = np.zeros_like(bg_uint8, dtype=np.uint8)
        self.region_mask[bg_uint8 < split_val] = 1
        self.region_mask[bg_uint8 >= split_val] = 2
        print(f"[INFO] region_mask created with split={split_val}.")
    def run(self):
        gc.collect()
        self._stop_flag = False
        p = self.get_current_params()
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            print(f"[ERROR] Cannot open {self.video_path}")
            self.finished_signal.emit(True, [], [])
            return
        self.prime_lightest_background(self.cap, p)
        self.kalman_filters = self.init_kalman_filters(p)
        self.track_ids = np.arange(p["MAX_TARGETS"])
        self.trajectories_pruned = [[] for _ in range(p["MAX_TARGETS"])]
        self.trajectories_full = [[] for _ in range(p["MAX_TARGETS"])]
        self.position_deques = [deque(maxlen=2) for _ in range(p["MAX_TARGETS"])]
        self.orientation_last = [None] * p["MAX_TARGETS"]
        self.last_shape_info = [None] * p["MAX_TARGETS"]
        self.detection_initialized = False
        self.tracking_stabilized = False
        self.detection_counts = 0
        self.tracking_counts = 0
        self.fps_list = []
        self.frame_count = 0
        self.start_time = time.time()
        local_counts_for_targets = [0] * p["MAX_TARGETS"]
        regionA_thresh = p.get("REGIONA_THRESHOLD", p["THRESHOLD_VALUE"])
        regionB_thresh = p.get("REGIONB_THRESHOLD", p["THRESHOLD_VALUE"])
        velocity_threshold = float(p.get("VELOCITY_THRESHOLD", 2.0))
        instant_flip_enabled = p.get("INSTANT_FLIP_ORIENTATION", True)
        max_orient_delta_stopped = float(p.get("max_orient_delta_stopped", 30.0))
        # New cost weights:
        W_POSITION = float(p.get("W_POSITION", 1.0))
        W_ORIENTATION = float(p.get("W_ORIENTATION", 1.0))
        W_AREA = float(p.get("W_AREA", 0.001))
        W_ASPECT = float(p.get("W_ASPECT", 0.1))
        use_mahal = p.get("USE_MAHALANOBIS", True)
        try:
            while not self._stop_flag:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    break
                self.frame_count += 1
                current_p = self.get_current_params()
                if current_p["RESIZE_FACTOR"] < 1.0:
                    f = current_p["RESIZE_FACTOR"]
                    new_w = int(frame.shape[1] * f)
                    new_h = int(frame.shape[0] * f)
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = apply_image_adjustments(gray, current_p["BRIGHTNESS"], current_p["CONTRAST"], current_p["GAMMA"])
                if self.background_model_lightest is None:
                    self.background_model_lightest = gray.astype(np.float32)
                    self.emit_frame(frame)
                    self._cleanup_locals(frame, gray)
                    continue
                self.background_model_lightest = np.maximum(self.background_model_lightest, gray.astype(np.float32))
                bg_uint8 = cv2.convertScaleAbs(self.background_model_lightest)
                diff = cv2.absdiff(bg_uint8, gray)
                if self.region_mask is not None:
                    maskA = (self.region_mask == 1) & (diff > regionA_thresh)
                    maskB = (self.region_mask == 2) & (diff > regionB_thresh)
                    combined = np.zeros_like(diff, dtype=np.uint8)
                    combined[maskA | maskB] = 255
                    fg_mask = combined
                else:
                    _, fg_mask = cv2.threshold(diff, current_p["THRESHOLD_VALUE"], 255, cv2.THRESH_BINARY)
                ksz = current_p["MORPH_KERNEL_SIZE"]
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
                main_mask = fg_mask.copy()
                contours, _ = cv2.findContours(main_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                meas_first, size_first = [], []
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area < current_p["MIN_CONTOUR_AREA"]:
                        continue
                    if len(cnt) >= 5:
                        (cx, cy), (ax1, ax2), raw_angle_deg = cv2.fitEllipse(cnt)
                        if ax1 < ax2:
                            ax1, ax2 = ax2, ax1
                            raw_angle_deg = (raw_angle_deg + 90) % 180
                        angle_rad = np.deg2rad(raw_angle_deg)
                        meas_first.append(np.array([cx, cy, angle_rad], np.float32))
                        size_first.append(area)
                if len(meas_first) > current_p["MAX_TARGETS"]:
                    sidx = np.argsort(size_first)[::-1]
                    meas_first = [meas_first[i] for i in sidx[:current_p["MAX_TARGETS"]]]
                if len(meas_first) == current_p["MAX_TARGETS"]:
                    self.detection_counts += 1
                else:
                    self.detection_counts = 0
                if self.detection_counts >= current_p["MIN_DETECTION_COUNTS"]:
                    self.detection_initialized = True
                if self.detection_initialized:
                    suspicious_bboxes = []
                    total_c = sum(1 for c in contours if cv2.contourArea(c) > 0)
                    for c in contours:
                        a = cv2.contourArea(c)
                        if a < 1:
                            continue
                        x, y, w, h = cv2.boundingRect(c)
                        if a > current_p["MERGE_AREA_THRESHOLD"] or total_c < current_p["MAX_TARGETS"]:
                            suspicious_bboxes.append((x, y, w, h))
                    for (bx, by, bw, bh) in suspicious_bboxes:
                        sub_mask = main_mask[by:by+bh, bx:bx+bw]
                        if np.count_nonzero(sub_mask) < 10:
                            continue
                        refined = self._local_conservative_split(sub_mask, current_p)
                        main_mask[by:by+bh, bx:bx+bw] = refined
                final_contours, _ = cv2.findContours(main_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                measurements, sizes, shapes = [], [], []
                for cnt in final_contours:
                    area = cv2.contourArea(cnt)
                    if area < current_p["MIN_CONTOUR_AREA"]:
                        continue
                    if len(cnt) >= 5:
                        (cx, cy), (ax1, ax2), raw_deg = cv2.fitEllipse(cnt)
                        if ax1 < ax2:
                            ax1, ax2 = ax2, ax1
                            raw_deg = (raw_deg + 90) % 180
                        rad = np.deg2rad(raw_deg)
                        measurements.append(np.array([cx, cy, rad], np.float32))
                        sizes.append(area)
                        shapes.append((np.pi*(ax1/2)*(ax2/2), ax1/ax2))
                if len(measurements) > current_p["MAX_TARGETS"]:
                    sidx = np.argsort(sizes)[::-1]
                    measurements = [measurements[i] for i in sidx[:current_p["MAX_TARGETS"]]]
                    shapes = [shapes[i] for i in sidx[:current_p["MAX_TARGETS"]]]
                overlay_frame = frame.copy()
                if self.detection_initialized and measurements:
                    if self.frame_count == 1:
                        hh, ww = gray.shape
                        for i in range(current_p["MAX_TARGETS"]):
                            kf = self.kalman_filters[i]
                            kf.statePre = np.array([np.random.randint(0, ww),
                                                      np.random.randint(0, hh),
                                                      0, 0, 0], np.float32)
                            kf.statePost = kf.statePre.copy()
                            self.position_deques[i] = deque(maxlen=2)
                            self.orientation_last[i] = None
                            self.last_shape_info[i] = None
                    predicted = []
                    for kf in self.kalman_filters:
                        pr = kf.predict()
                        predicted.append(pr[:3].flatten())
                    predicted = np.array(predicted, dtype=np.float32)
                    cost_matrix = np.zeros((current_p["MAX_TARGETS"], len(measurements)), dtype=np.float32)
                    for i in range(current_p["MAX_TARGETS"]):
                        for j in range(len(measurements)):
                            if use_mahal:
                                P = self.kalman_filters[i].errorCovPre[0:2, 0:2]
                                diff = measurements[j][:2] - predicted[i][:2]
                                try:
                                    invP = np.linalg.inv(P)
                                    pos_cost = np.sqrt(np.dot(diff.T, np.dot(invP, diff)))
                                except:
                                    pos_cost = np.linalg.norm(diff)
                            else:
                                pos_cost = np.linalg.norm(measurements[j][:2] - predicted[i][:2])
                            ori_diff = abs(predicted[i][2] - measurements[j][2])
                            ori_diff = min(ori_diff, 2*np.pi - ori_diff)
                            if self.last_shape_info[i] is not None:
                                prev_area, prev_aspect = self.last_shape_info[i]
                            else:
                                prev_area, prev_aspect = shapes[j]
                            shape_area_diff = abs(shapes[j][0] - prev_area)
                            shape_aspect_diff = abs(shapes[j][1] - prev_aspect)
                            cost_matrix[i, j] = (current_p.get("W_POSITION", 1.0) * pos_cost +
                                                 current_p.get("W_ORIENTATION", 1.0) * ori_diff +
                                                 current_p.get("W_AREA", 0.001) * shape_area_diff +
                                                 current_p.get("W_ASPECT", 0.1) * shape_aspect_diff)
                    row_idx, col_idx = linear_sum_assignment(cost_matrix)
                    avg_cost = 0.0
                    for row, col in zip(row_idx, col_idx):
                        if row < current_p["MAX_TARGETS"] and col < len(measurements):
                            if (not self.tracking_stabilized) or (cost_matrix[row, col] < current_p["MAX_DISTANCE_THRESHOLD"]):
                                kf = self.kalman_filters[row]
                                measure_vec = np.array([[measurements[col][0]],
                                                         [measurements[col][1]],
                                                         [measurements[col][2]]], np.float32)
                                kf.correct(measure_vec)
                                x_corr, y_corr, theta_ellipse = measurements[col]
                                ts = self.frame_count
                                self.position_deques[row].append((x_corr, y_corr))
                                speed = 0.0
                                if len(self.position_deques[row]) == 2:
                                    (px1, py1) = self.position_deques[row][0]
                                    (px2, py2) = self.position_deques[row][1]
                                    vx = (px2 - px1)
                                    vy = (py2 - py1)
                                    speed = np.hypot(vx, vy)
                                final_angle = theta_ellipse
                                old_angle = self.orientation_last[row]
                                if speed < velocity_threshold:
                                    if old_angle is not None:
                                        old_deg = np.degrees(old_angle)
                                        new_deg = np.degrees(theta_ellipse)
                                        delta = wrap_angle_degs(new_deg - old_deg)
                                        if abs(delta) > 90:
                                            new_deg = (new_deg + 180) % 360
                                        else:
                                            if abs(delta) > max_orient_delta_stopped:
                                                new_deg = old_deg + np.sign(delta) * max_orient_delta_stopped
                                        final_angle = np.radians(new_deg) % (2*np.pi)
                                    else:
                                        final_angle = theta_ellipse
                                else:
                                    if instant_flip_enabled:
                                        motion_angle = np.arctan2(vy, vx)
                                        diff = (motion_angle - theta_ellipse + np.pi) % (2*np.pi) - np.pi
                                        if abs(diff) > np.pi/2:
                                            final_angle = (theta_ellipse + np.pi) % (2*np.pi)
                                        else:
                                            final_angle = theta_ellipse
                                    else:
                                        final_angle = theta_ellipse
                                self.orientation_last[row] = final_angle
                                self.last_shape_info[row] = shapes[col]
                                self.trajectories_full[row].append((int(x_corr), int(y_corr), float(final_angle), ts))
                                self.trajectories_pruned[row].append((int(x_corr), int(y_corr), float(final_angle), ts))
                                self._prune_short_history(row, ts, current_p["TRAJECTORY_HISTORY_SECONDS"])
                                if self.csv_writer_thread:
                                    idx_val = local_counts_for_targets[row]
                                    csv_row = [row, idx_val, int(x_corr), int(y_corr), float(final_angle), ts]
                                    self.csv_writer_thread.enqueue(csv_row)
                                    local_counts_for_targets[row] += 1
                                avg_cost += cost_matrix[row, col] / current_p["MAX_TARGETS"]
                            else:
                                self.tracking_stabilized = False
                                self.tracking_counts = 0
                                ts = self.frame_count
                                self.trajectories_full[row].append((np.nan, np.nan, np.nan, ts))
                                self.trajectories_pruned[row].append((np.nan, np.nan, np.nan, ts))
                                self._prune_short_history(row, ts, current_p["TRAJECTORY_HISTORY_SECONDS"])
                    if avg_cost < current_p["MAX_DISTANCE_THRESHOLD"]:
                        self.tracking_counts += 1
                    else:
                        self.tracking_counts = 0
                    if self.tracking_counts >= current_p["MIN_TRACKING_COUNTS"] and not self.tracking_stabilized:
                        self.tracking_stabilized = True
                        print(f"[INFO] Tracking stabilized (avg cost={avg_cost:.2f})")
                    if current_p["SHOW_BLOB"]:
                        self._draw_overlays(overlay_frame, self.trajectories_pruned, current_p)
                if current_p["SHOW_FG"]:
                    small_fg = cv2.resize(main_mask, (0, 0), fx=0.3, fy=0.3)
                    overlay_frame[0:small_fg.shape[0], 0:small_fg.shape[1]] = cv2.cvtColor(small_fg, cv2.COLOR_GRAY2BGR)
                if current_p["SHOW_BG"]:
                    bg_bgr = cv2.cvtColor(bg_uint8, cv2.COLOR_GRAY2BGR)
                    small_bg = cv2.resize(bg_bgr, (0, 0), fx=0.3, fy=0.3)
                    xoff = overlay_frame.shape[1] - small_bg.shape[1]
                    overlay_frame[0:small_bg.shape[0], xoff:] = small_bg
                elapsed = time.time() - self.start_time
                if elapsed > 0:
                    self.fps_list.append(self.frame_count / elapsed)
                self.emit_frame(overlay_frame)
                self._cleanup_locals(frame, gray, diff, bg_uint8, fg_mask, kernel, main_mask,
                                     overlay_frame, contours, final_contours)
                gc.collect()
        except Exception as e:
            print(f"[ERROR] Worker exception: {e}")
            self._stop_flag = True
        finally:
            if self.cap:
                self.cap.release()
        done_normally = not self._stop_flag
        self.finished_signal.emit(done_normally, self.fps_list, self.trajectories_full)
        gc.collect()
    def _local_conservative_split(self, sub_mask: np.ndarray, p: dict) -> np.ndarray:
        ksize = p["CONSERVATIVE_KERNEL_SIZE"]
        iters = p["CONSERVATIVE_ERODE_ITER"]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        refined = cv2.erode(sub_mask, kernel, iterations=iters)
        refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel)
        return refined
    def emit_frame(self, bgr_frame: np.ndarray):
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        self.frame_signal.emit(rgb_frame)
    def _cleanup_locals(self, *objs):
        for obj in objs:
            del obj
    def _prune_short_history(self, idx, current_ts, history_sec):
        self.trajectories_pruned[idx] = [
            (x, y, th, t)
            for (x, y, th, t) in self.trajectories_pruned[idx]
            if (current_ts - t) <= history_sec
        ]
    def _draw_overlays(self, frame_bgr: np.ndarray, pruned_trajs: list, p: dict):
        for i, tlist in enumerate(pruned_trajs):
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
            cv2.putText(frame_bgr, f"ID:{i}", (x+15, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_int, 2)
            for pt_i in range(1, len(tlist)):
                pt1 = (tlist[pt_i-1][0], tlist[pt_i-1][1])
                pt2 = (tlist[pt_i][0], tlist[pt_i][1])
                if not (np.isnan(pt1[0]) or np.isnan(pt1[1]) or np.isnan(pt2[0]) or np.isnan(pt2[1])):
                    cv2.line(frame_bgr, pt1, pt2, color_int, 2)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Fly Tracking with Partial Orientation & Extended Shape Features")
        self.resize(1200, 800)
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        self.scroll_area.setWidget(self.video_label)
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
        # CSV path
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
        form_layout.addRow("Global Threshold", self.spin_threshold)
        self.spin_morph_size = QSpinBox(); self.spin_morph_size.setRange(1, 50); self.spin_morph_size.setValue(5)
        form_layout.addRow("Morph Kernel Size", self.spin_morph_size)
        self.spin_min_contour = QSpinBox(); self.spin_min_contour.setRange(0, 100000); self.spin_min_contour.setValue(50)
        form_layout.addRow("Min Contour Area", self.spin_min_contour)
        self.spin_max_dist = QSpinBox(); self.spin_max_dist.setRange(0, 2000); self.spin_max_dist.setValue(25)
        form_layout.addRow("Max Distance Thresh", self.spin_max_dist)
        self.spin_min_detect = QSpinBox(); self.spin_min_detect.setRange(0, 1000); self.spin_min_detect.setValue(10)
        form_layout.addRow("Min Detection Counts", self.spin_min_detect)
        self.spin_min_track = QSpinBox(); self.spin_min_track.setRange(0, 1000); self.spin_min_track.setValue(10)
        form_layout.addRow("Min Tracking Counts", self.spin_min_track)
        self.spin_traj_hist = QSpinBox(); self.spin_traj_hist.setRange(0, 300); self.spin_traj_hist.setValue(5)
        form_layout.addRow("Trajectory History (sec)", self.spin_traj_hist)
        self.spin_bg_prime = QSpinBox(); self.spin_bg_prime.setRange(0, 5000); self.spin_bg_prime.setValue(10)
        form_layout.addRow("BACKGROUND_PRIME_FRAMES", self.spin_bg_prime)
        # Region splitting
        self.chk_otsu_split = QCheckBox("Use Otsu Split?"); self.chk_otsu_split.setChecked(False)
        form_layout.addRow(self.chk_otsu_split)
        self.spin_bg_split = QSpinBox(); self.spin_bg_split.setRange(0, 255); self.spin_bg_split.setValue(0)
        form_layout.addRow("BG_SPLIT_VALUE", self.spin_bg_split)
        self.spin_regionA_thresh = QSpinBox(); self.spin_regionA_thresh.setRange(0, 255); self.spin_regionA_thresh.setValue(40)
        form_layout.addRow("REGIONA_THRESHOLD", self.spin_regionA_thresh)
        self.spin_regionB_thresh = QSpinBox(); self.spin_regionB_thresh.setRange(0, 255); self.spin_regionB_thresh.setValue(60)
        form_layout.addRow("REGIONB_THRESHOLD", self.spin_regionB_thresh)
        # Kalman parameters
        self.spin_kalman_noise = QDoubleSpinBox(); self.spin_kalman_noise.setRange(0.0, 1.0); self.spin_kalman_noise.setValue(0.03); self.spin_kalman_noise.setSingleStep(0.01)
        form_layout.addRow("Kalman Noise Cov", self.spin_kalman_noise)
        self.spin_kalman_meas_noise = QDoubleSpinBox(); self.spin_kalman_meas_noise.setRange(0.0, 1.0); self.spin_kalman_meas_noise.setValue(0.1); self.spin_kalman_meas_noise.setSingleStep(0.01)
        form_layout.addRow("Kalman Meas Cov", self.spin_kalman_meas_noise)
        # Resize
        self.spin_resize_factor = QDoubleSpinBox(); self.spin_resize_factor.setRange(0.1, 1.0); self.spin_resize_factor.setValue(1.0); self.spin_resize_factor.setSingleStep(0.1)
        form_layout.addRow("Resize Factor", self.spin_resize_factor)
        # Merges
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
        # Orientation parameters
        self.spin_velocity_threshold = QDoubleSpinBox(); self.spin_velocity_threshold.setRange(0.0, 100.0); self.spin_velocity_threshold.setValue(2.0); self.spin_velocity_threshold.setSingleStep(0.1)
        form_layout.addRow("VELOCITY_THRESHOLD", self.spin_velocity_threshold)
        self.chk_instant_flip = QCheckBox("Instant Flip Orientation?"); self.chk_instant_flip.setChecked(True)
        form_layout.addRow(self.chk_instant_flip)
        self.spin_max_orient_delta = QDoubleSpinBox(); self.spin_max_orient_delta.setRange(1.0, 180.0); self.spin_max_orient_delta.setValue(30.0); self.spin_max_orient_delta.setSingleStep(1.0)
        form_layout.addRow("MAX_ORIENT_DELTA_STOPPED (deg)", self.spin_max_orient_delta)
        # New cost weights for shape and position:
        self.spin_W_POSITION = QDoubleSpinBox(); self.spin_W_POSITION.setRange(0.0, 10.0); self.spin_W_POSITION.setValue(1.0); self.spin_W_POSITION.setSingleStep(0.1)
        form_layout.addRow("W_POSITION", self.spin_W_POSITION)
        self.spin_W_ORIENTATION = QDoubleSpinBox(); self.spin_W_ORIENTATION.setRange(0.0, 10.0); self.spin_W_ORIENTATION.setValue(1.0); self.spin_W_ORIENTATION.setSingleStep(0.1)
        form_layout.addRow("W_ORIENTATION", self.spin_W_ORIENTATION)
        self.spin_W_AREA = QDoubleSpinBox(); self.spin_W_AREA.setRange(0.0, 1.0); self.spin_W_AREA.setValue(0.001); self.spin_W_AREA.setSingleStep(0.0005)
        form_layout.addRow("W_AREA", self.spin_W_AREA)
        self.spin_W_ASPECT = QDoubleSpinBox(); self.spin_W_ASPECT.setRange(0.0, 10.0); self.spin_W_ASPECT.setValue(0.1); self.spin_W_ASPECT.setSingleStep(0.1)
        form_layout.addRow("W_ASPECT", self.spin_W_ASPECT)
        self.chk_use_mahal = QCheckBox("Use Mahalanobis for Position?")
        self.chk_use_mahal.setChecked(True)
        form_layout.addRow(self.chk_use_mahal)
        # Display options
        self.chk_show_fg = QCheckBox("Show Foreground Mask"); self.chk_show_fg.setChecked(True)
        form_layout.addRow(self.chk_show_fg)
        self.chk_show_blob = QCheckBox("Show Blob Overlays"); self.chk_show_blob.setChecked(True)
        form_layout.addRow(self.chk_show_blob)
        self.chk_show_bg = QCheckBox("Show Lightest Background"); self.chk_show_bg.setChecked(True)
        form_layout.addRow(self.chk_show_bg)
        # Zoom
        self.spin_zoom = QDoubleSpinBox(); self.spin_zoom.setRange(0.1, 5.0); self.spin_zoom.setValue(1.0); self.spin_zoom.setSingleStep(0.1)
        form_layout.addRow("Zoom Factor", self.spin_zoom)
        param_group.setLayout(form_layout)
        # Buttons
        self.btn_preview = QPushButton("Preview")
        self.btn_preview.setCheckable(True)
        self.btn_preview.clicked.connect(self.toggle_preview)
        self.btn_start_tracking = QPushButton("Full Tracking")
        self.btn_start_tracking.clicked.connect(self.start_full_tracking)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self.stop_tracking)
        control_layout.addLayout(file_layout)
        control_layout.addLayout(csv_layout)
        control_layout.addWidget(param_group)
        control_layout.addWidget(self.btn_preview)
        control_layout.addWidget(self.btn_start_tracking)
        control_layout.addWidget(self.btn_stop)
        control_layout.addStretch(1)
        main_layout.addWidget(self.scroll_area, stretch=1)
        main_layout.addLayout(control_layout, stretch=0)
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        self.tracking_worker = None
        self.csv_writer_thread = None
        self.final_trajectories_full = []
        self.load_config()

    def load_config(self):
        if os.path.isfile(CONFIG_FILENAME):
            try:
                with open(CONFIG_FILENAME, "r") as f:
                    cfg = json.load(f)
                self.file_path_line.setText(cfg.get("file_path", ""))
                self.csv_path_line.setText(cfg.get("csv_path", ""))
                self.spin_max_targets.setValue(cfg.get("max_targets", 4))
                self.spin_threshold.setValue(cfg.get("threshold_value", 50))
                self.spin_morph_size.setValue(cfg.get("morph_kernel_size", 5))
                self.spin_min_contour.setValue(cfg.get("min_contour_area", 50))
                self.spin_max_dist.setValue(cfg.get("max_dist_thresh", 25))
                self.spin_min_detect.setValue(cfg.get("min_detect_counts", 10))
                self.spin_min_track.setValue(cfg.get("min_track_counts", 10))
                self.spin_traj_hist.setValue(cfg.get("traj_history", 5))
                self.spin_bg_prime.setValue(cfg.get("bg_prime_frames", 10))
                self.chk_otsu_split.setChecked(cfg.get("use_otsu_split", False))
                self.spin_bg_split.setValue(cfg.get("bg_split_value", 0))
                self.spin_regionA_thresh.setValue(cfg.get("regionA_thresh", 40))
                self.spin_regionB_thresh.setValue(cfg.get("regionB_thresh", 60))
                self.spin_kalman_noise.setValue(cfg.get("kalman_noise", 0.03))
                self.spin_kalman_meas_noise.setValue(cfg.get("kalman_meas_noise", 0.1))
                self.spin_resize_factor.setValue(cfg.get("resize_factor", 1.0))
                self.spin_merge_area_thr.setValue(cfg.get("merge_area_thr", 1500))
                self.spin_conservative_kernel.setValue(cfg.get("conservative_kernel", 5))
                self.spin_conservative_iter.setValue(cfg.get("conservative_iter", 2))
                self.spin_brightness.setValue(cfg.get("brightness", 0.0))
                self.spin_contrast.setValue(cfg.get("contrast", 1.0))
                self.spin_gamma.setValue(cfg.get("gamma", 1.0))
                self.spin_velocity_threshold.setValue(cfg.get("velocity_threshold", 2.0))
                self.chk_instant_flip.setChecked(cfg.get("instant_flip", True))
                self.spin_max_orient_delta.setValue(cfg.get("max_orient_delta_stopped", 30.0))
                self.spin_W_POSITION.setValue(cfg.get("W_POSITION", 1.0))
                self.spin_W_ORIENTATION.setValue(cfg.get("W_ORIENTATION", 1.0))
                self.spin_W_AREA.setValue(cfg.get("W_AREA", 0.001))
                self.spin_W_ASPECT.setValue(cfg.get("W_ASPECT", 0.1))
                self.chk_use_mahal.setChecked(cfg.get("USE_MAHALANOBIS", True))
                self.chk_show_fg.setChecked(cfg.get("show_fg", True))
                self.chk_show_blob.setChecked(cfg.get("show_blob", True))
                self.chk_show_bg.setChecked(cfg.get("show_bg", True))
                self.spin_zoom.setValue(cfg.get("zoom_factor", 1.0))
                print("[INFO] Loaded config from", CONFIG_FILENAME)
            except Exception as e:
                print("[WARNING] Could not load config:", e)

    def save_config(self):
        cfg = {
            "file_path": self.file_path_line.text(),
            "csv_path": self.csv_path_line.text(),
            "max_targets": self.spin_max_targets.value(),
            "threshold_value": self.spin_threshold.value(),
            "morph_kernel_size": self.spin_morph_size.value(),
            "min_contour_area": self.spin_min_contour.value(),
            "max_dist_thresh": self.spin_max_dist.value(),
            "min_detect_counts": self.spin_min_detect.value(),
            "min_track_counts": self.spin_min_track.value(),
            "traj_history": self.spin_traj_hist.value(),
            "bg_prime_frames": self.spin_bg_prime.value(),
            "use_otsu_split": self.chk_otsu_split.isChecked(),
            "bg_split_value": self.spin_bg_split.value(),
            "regionA_thresh": self.spin_regionA_thresh.value(),
            "regionB_thresh": self.spin_regionB_thresh.value(),
            "kalman_noise": self.spin_kalman_noise.value(),
            "kalman_meas_noise": self.spin_kalman_meas_noise.value(),
            "resize_factor": self.spin_resize_factor.value(),
            "merge_area_thr": self.spin_merge_area_thr.value(),
            "conservative_kernel": self.spin_conservative_kernel.value(),
            "conservative_iter": self.spin_conservative_iter.value(),
            "brightness": self.spin_brightness.value(),
            "contrast": self.spin_contrast.value(),
            "gamma": self.spin_gamma.value(),
            "velocity_threshold": self.spin_velocity_threshold.value(),
            "instant_flip": self.chk_instant_flip.isChecked(),
            "max_orient_delta_stopped": self.spin_max_orient_delta.value(),
            "W_POSITION": self.spin_W_POSITION.value(),
            "W_ORIENTATION": self.spin_W_ORIENTATION.value(),
            "W_AREA": self.spin_W_AREA.value(),
            "W_ASPECT": self.spin_W_ASPECT.value(),
            "USE_MAHALANOBIS": self.chk_use_mahal.isChecked(),
            "show_fg": self.chk_show_fg.isChecked(),
            "show_blob": self.chk_show_blob.isChecked(),
            "show_bg": self.chk_show_bg.isChecked(),
            "zoom_factor": self.spin_zoom.value()
        }
        try:
            with open(CONFIG_FILENAME, "w") as f:
                json.dump(cfg, f, indent=2)
            print("[INFO] Saved config to", CONFIG_FILENAME)
        except Exception as e:
            print("[WARNING] Could not save config:", e)

    def select_file(self):
        fp, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov)")
        if fp:
            self.file_path_line.setText(fp)

    def select_csv(self):
        csv_fp, _ = QFileDialog.getSaveFileName(self, "Select CSV File to Save", "", "CSV Files (*.csv)")
        if csv_fp:
            self.csv_path_line.setText(csv_fp)

    def get_parameters_dict(self):
        max_targets = self.spin_max_targets.value()
        np.random.seed(42)
        color_array = [tuple(c.tolist()) for c in np.random.randint(0, 255, (max_targets, 3))]
        p = {
            "MAX_TARGETS": max_targets,
            "THRESHOLD_VALUE": self.spin_threshold.value(),
            "BG_SPLIT_VALUE": self.spin_bg_split.value(),
            "USE_OTSU_SPLIT": self.chk_otsu_split.isChecked(),
            "REGIONA_THRESHOLD": self.spin_regionA_thresh.value(),
            "REGIONB_THRESHOLD": self.spin_regionB_thresh.value(),
            "MORPH_KERNEL_SIZE": self.spin_morph_size.value(),
            "MIN_CONTOUR_AREA": self.spin_min_contour.value(),
            "MAX_DISTANCE_THRESHOLD": self.spin_max_dist.value(),
            "MIN_DETECTION_COUNTS": self.spin_min_detect.value(),
            "MIN_TRACKING_COUNTS": self.spin_min_track.value(),
            "TRAJECTORY_HISTORY_SECONDS": self.spin_traj_hist.value(),
            "BACKGROUND_PRIME_FRAMES": self.spin_bg_prime.value(),
            "KALMAN_NOISE_COVARIANCE": float(self.spin_kalman_noise.value()),
            "KALMAN_MEASUREMENT_NOISE_COVARIANCE": float(self.spin_kalman_meas_noise.value()),
            "RESIZE_FACTOR": float(self.spin_resize_factor.value()),
            "MERGE_AREA_THRESHOLD": self.spin_merge_area_thr.value(),
            "CONSERVATIVE_KERNEL_SIZE": self.spin_conservative_kernel.value(),
            "CONSERVATIVE_ERODE_ITER": self.spin_conservative_iter.value(),
            "BRIGHTNESS": float(self.spin_brightness.value()),
            "CONTRAST": float(self.spin_contrast.value()),
            "GAMMA": float(self.spin_gamma.value()),
            "VELOCITY_THRESHOLD": float(self.spin_velocity_threshold.value()),
            "INSTANT_FLIP_ORIENTATION": self.chk_instant_flip.isChecked(),
            "max_orient_delta_stopped": float(self.spin_max_orient_delta.value()),
            "W_POSITION": float(self.spin_W_POSITION.value()),
            "W_ORIENTATION": float(self.spin_W_ORIENTATION.value()),
            "W_AREA": float(self.spin_W_AREA.value()),
            "W_ASPECT": float(self.spin_W_ASPECT.value()),
            "USE_MAHALANOBIS": self.chk_use_mahal.isChecked(),
            "TRAJECTORY_COLORS": color_array,
            "SHOW_FG": self.chk_show_fg.isChecked(),
            "SHOW_BLOB": self.chk_show_blob.isChecked(),
            "SHOW_BG": self.chk_show_bg.isChecked(),
            "zoom_factor": self.spin_zoom.value()
        }
        return p

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
        self.save_config()
        video_fp = self.file_path_line.text()
        if not video_fp:
            QMessageBox.warning(self, "No file selected", "Please select a video file first!")
            if preview_mode:
                self.btn_preview.setChecked(False)
                self.btn_preview.setText("Preview")
            return
        if self.tracking_worker and self.tracking_worker.isRunning():
            QMessageBox.warning(self, "Worker busy", "A tracking thread is already running.")
            return
        csv_fp = self.csv_path_line.text()
        self.csv_writer_thread = None
        if csv_fp and (not preview_mode):
            self.csv_writer_thread = CSVWriterThread(csv_fp, header=["TargetID", "Index", "X", "Y", "Theta", "FrameID"])
            self.csv_writer_thread.start()
            print(f"[INFO] CSV Writer Thread started for {csv_fp}")
        self.tracking_worker = TrackingWorker(video_fp, csv_writer_thread=self.csv_writer_thread)
        self.tracking_worker.set_parameters(self.get_parameters_dict())
        self.tracking_worker.frame_signal.connect(self.on_new_frame)
        self.tracking_worker.finished_signal.connect(self.on_tracking_finished)
        self.tracking_worker.start()

    def stop_tracking(self):
        if self.tracking_worker and self.tracking_worker.isRunning():
            self.tracking_worker.stop()

    @Slot(np.ndarray)
    def on_new_frame(self, rgb_frame: np.ndarray):
        zoom_factor = float(self.spin_zoom.value())
        zoom_factor = max(zoom_factor, 0.1)
        h, w, c = rgb_frame.shape
        bytes_per_line = c * w
        qimg = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        scaled_w = int(w * zoom_factor)
        scaled_h = int(h * zoom_factor)
        scaled_qimg = qimg.scaled(scaled_w, scaled_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(QPixmap.fromImage(scaled_qimg))

    @Slot(bool, list, list)
    def on_tracking_finished(self, finished_normally: bool, fps_list: list, full_traj: list):
        if self.csv_writer_thread:
            self.csv_writer_thread.stop()
            self.csv_writer_thread.join()
            self.csv_writer_thread = None
            print("[INFO] CSV Writer Thread stopped.")
        if self.btn_preview.isChecked():
            self.btn_preview.setChecked(False)
            self.btn_preview.setText("Preview")
        if finished_normally and (not self.btn_preview.isChecked()):
            self.final_trajectories_full = full_traj
            QMessageBox.information(self, "Tracking Finished", "Full run completed. CSV was written if a path was provided.")
            self.plot_fps(fps_list)
        gc.collect()

    def plot_fps(self, fps_list):
        if len(fps_list) < 2:
            QMessageBox.information(self, "FPS Plot", "Not enough data to plot.")
            return
        plt.figure()
        plt.plot(fps_list, label="FPS")
        plt.xlabel("Frame Index")
        plt.ylabel("FPS")
        plt.title("Tracking FPS Over Time")
        plt.legend()
        plt.show()

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
