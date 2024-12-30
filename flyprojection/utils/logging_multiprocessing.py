import logging
import logging.handlers
import multiprocessing
import signal
import sys
import os
import numpy as np
import h5py
import hdf5plugin
import time
import subprocess

def _logging_process(queue, log_file, level):
    """
    Target function for the logging process.

    Continuously reads log records from the multiprocessing queue and writes them to the specified log file.

    Parameters
    ----------
    queue : multiprocessing.Queue
        Queue from which to read log records.
    log_file : str
        Path to the log file.
    level : int
        Logging level (e.g. logging.INFO).
    """
    # Set up a logger in this child process
    logger = logging.getLogger("async_logger_process")
    logger.setLevel(level)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    while True:
        record = queue.get()
        if record is None:
            break
        # Recreate the LogRecord from the tuple, as the record cannot be passed directly
        # The tuple should come from a `logging.handlers.QueueHandler` in the main process.
        if isinstance(record, logging.LogRecord):
            # If we directly passed log records (not recommended across processes), just emit
            logger.handle(record)
        else:
            # If we passed a tuple, we need to recreate a LogRecord. If desired, you can serialize differently.
            # Here, we assume the main process used standard QueueHandler which passes LogRecords directly.
            logger.handle(record)

    # Cleanup
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)


class AsyncLogger:
    """
    An asynchronous logger that writes log messages to a file using a separate process.

    This class:
    - Uses a named logger (not the root logger) to avoid interfering with global logging settings.
    - Sends log records through a multiprocessing-safe queue to a background process.
    - The background process writes logs to the file independently, bypassing the GIL for I/O.

    Usage:
        with AsyncLogger(log_file="app.log", logger_name="my_logger") as logger:
            logger.info("This is a log message.")
    """

    def __init__(self, log_file, logger_name='async_logger', level=logging.INFO):
        """
        Initialize the asynchronous logger.

        Parameters
        ----------
        log_file : str
            Path to the log file.
        logger_name : str
            Name for the logger (non-root).
        level : int
            Logging level (e.g., logging.INFO).
        """
        self.log_file = log_file
        self.logger_name = logger_name
        self.level = level
        self.log_queue = multiprocessing.Queue()
        self.logger = None
        self.log_process = None

    def __enter__(self):
        """
        Enter the runtime context for the asynchronous logger.

        Sets up the logger, a QueueHandler, and starts the background logging process.
        """
        self.logger = logging.getLogger(self.logger_name)
        self.logger.setLevel(self.level)

        queue_handler = logging.handlers.QueueHandler(self.log_queue)
        self.logger.addHandler(queue_handler)

        # Start the logging process
        self.log_process = multiprocessing.Process(
            target=_logging_process,
            args=(self.log_queue, self.log_file, self.level),
            daemon=True
        )
        self.log_process.start()
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the runtime context for the asynchronous logger.

        Signals the logging process to stop, waits for it, and closes all handlers.
        """
        # Signal the listener process to stop
        self.log_queue.put(None)
        self.log_process.join()

        # Remove all handlers and close them in the main logger
        handlers = self.logger.handlers[:]
        for h in handlers:
            self.logger.removeHandler(h)
            h.close()

        print("Successfully closed Logger. Exiting AsyncLogger Context.")


def _hdf5_writer_process(h5_filename, datasets_config, metadata_config, global_metadata, frame_queue, stop_event):
    """
    Worker process target function for asynchronous HDF5 saving.

    This process:
    - Opens the HDF5 file and creates datasets.
    - Continuously reads frames from the queue, processes them (e.g., background subtraction, difference coding),
      and writes to the HDF5 file.
    - Handles metadata if configured.
    - Shuts down gracefully when signaled.
    """
    h5_file = h5py.File(h5_filename, 'w')

    # Create datasets as per configuration
    datasets = {}
    for name, cfg in datasets_config.items():
        shape = cfg["shape"]
        dtype = cfg["dtype"]
        compression = cfg.get("compression", None)
        maxshape = (None,) + shape
        chunks = (1,) + shape
        dset = h5_file.create_dataset(
            name,
            shape=(0,) + shape,
            maxshape=maxshape,
            dtype=dtype,
            chunks=chunks,
            compression=compression
        )
        cfg["prev_frame"] = None
        datasets[name] = dset

    metadata_dset = None
    if metadata_config is not None:
        md_dtype = metadata_config["dtype"]
        compression = metadata_config.get("compression", None)
        metadata_dset = h5_file.create_dataset(
            "metadata",
            shape=(0,),
            maxshape=(None,),
            dtype=md_dtype,
            chunks=(1,),
            compression=compression
        )

    # Store global metadata as attributes
    for k, v in global_metadata.items():
        h5_file.attrs[k] = v

    frames_written = 0

    def process_frame_data(dataset_name, frame_data):
        """
        Apply background subtraction OR difference coding if enabled for this dataset.

        This function is defined inside the process for isolation and direct access to datasets_config.
        """
        cfg = datasets_config[dataset_name]
        dtype = cfg["dtype"]

        use_background = cfg.get("background", None)
        diff_coding = cfg.get("difference_coding", False)

        data = frame_data.astype(dtype, copy=False)

        # Background handling: might be bool True (use first frame) or an ndarray
        if isinstance(use_background, bool):
            if use_background:
                if frames_written == 0:
                    # First frame is stored as background
                    cfg["background_frame"] = data.astype(np.int16).copy()
                else:
                    bg = cfg["background_frame"]
                    diff = data.astype(np.int16) - bg
                    data = diff.astype(dtype)
            # If False, do nothing
        elif use_background is not None:
            bg = use_background.astype(np.int16, copy=False)
            diff = data.astype(np.int16) - bg
            data = diff.astype(dtype)

        # Difference coding
        if diff_coding:
            if frames_written == 0:
                cfg["prev_frame"] = data.copy()
            else:
                prev = cfg["prev_frame"].astype(np.int16)
                curr = data.astype(np.int16)
                diff = curr - prev
                data = diff.astype(dtype)
                cfg["prev_frame"] = (prev + diff).astype(dtype)

        return data

    # Main loop: read from the queue until stopped
    while True:
        if not frame_queue.empty():
            item = frame_queue.get()
            if item is None:
                # Signal to stop
                break

            metadata, data_dict = item

            # Process and write each dataset frame
            for name, dset in datasets.items():
                frame_data = data_dict[name]
                processed_data = process_frame_data(name, frame_data)
                new_size = frames_written + 1
                dset.resize((new_size,) + datasets_config[name]["shape"])
                dset[frames_written] = processed_data

            # Write metadata if available
            if metadata_dset is not None:
                new_size = frames_written + 1
                metadata_dset.resize((new_size,))
                md_entry = np.zeros(1, dtype=metadata_config["dtype"])
                for field in metadata_config["dtype"].names:
                    md_entry[field] = metadata.get(field, 0)
                metadata_dset[frames_written] = md_entry[0]

            frames_written += 1

        if stop_event.is_set() and frame_queue.empty():
            break

    # Finalize
    h5_file.flush()
    h5_file.close()


class AsyncHDF5Saver:
    """
    An asynchronous HDF5 saver that writes frames (and optional metadata) to an HDF5 file in a separate process.

    Features:
    - Multiple datasets with configurable shape, dtype, and compression.
    - Supports background subtraction or difference coding (mutually exclusive).
    - Optional per-frame metadata dataset.
    - Global metadata stored as file attributes.
    - Uses a multiprocessing queue and a dedicated process for non-blocking writes.

    Reconstruction Instructions:
    - If background subtraction was used:
        original_frame = stored_frame + background
    - If difference coding was used:
        original_frame_0 = stored_frame_0
        for i in range(1, n_frames):
            original_frame_i = original_frame_(i-1) + stored_frame_i
    """

    def __init__(
        self,
        h5_filename,
        datasets_config,
        metadata_config=None,
        global_metadata=None,
    ):
        """
        Initialize the asynchronous HDF5 saver.

        Parameters
        ----------
        h5_filename : str
            Path to the HDF5 file to be created.
        datasets_config : dict
            Configuration dict for each dataset with keys:
              - "shape": tuple (frame dimensions)
              - "dtype": numpy dtype or equivalent
              - "compression": hdf5plugin compression filter or None
              - "background": None, a boolean, or a background ndarray
              - "difference_coding": bool
        metadata_config : dict, optional
            Configuration for per-frame metadata dataset.
              - "dtype": numpy dtype describing the metadata fields
              - "compression": optional compression
        global_metadata : dict, optional
            Key-value pairs stored as HDF5 file attributes.
        """
        self.h5_filename = h5_filename
        os.makedirs(os.path.dirname(h5_filename), exist_ok=True)

        # Enforce exclusivity of background/diff coding
        for name, cfg in datasets_config.items():
            background = cfg.get("background", None)
            diff_coding = cfg.get("difference_coding", False)
            if background is not None and diff_coding:
                raise ValueError(
                    f"Dataset '{name}' cannot have both difference coding and background subtraction."
                )

        self.datasets_config = datasets_config
        self.metadata_config = metadata_config
        self.global_metadata = global_metadata or {}

        self.frame_queue = multiprocessing.Queue()
        self.stop_event = multiprocessing.Event()
        self.worker_process = None

    def __enter__(self):
        """
        Enter the runtime context, starting the worker process to handle HDF5 writes.
        """
        self.worker_process = multiprocessing.Process(
            target=_hdf5_writer_process,
            args=(self.h5_filename,
                  self.datasets_config,
                  self.metadata_config,
                  self.global_metadata,
                  self.frame_queue,
                  self.stop_event),
            daemon=True
        )
        self.worker_process.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the runtime context, triggering a graceful shutdown of the worker process.
        """
        self.shutdown()
        print("Successfully closed HDF5 Saver. Exiting AsyncHDF5Saver Context.")

    def save_frame_async(self, metadata, **data_dict):
        """
        Enqueue a frame for asynchronous saving.

        Parameters
        ----------
        metadata : dict
            Dictionary with per-frame metadata fields (if any).
        data_dict : dict
            Mapping dataset_name -> frame_array for all configured datasets.
        """
        # Validate that all configured datasets are present
        for name in self.datasets_config:
            if name not in data_dict:
                raise ValueError(f"Missing data for dataset '{name}' in save_frame_async call.")

        self.frame_queue.put((metadata, data_dict))

    def shutdown(self):
        """
        Gracefully shutdown the HDF5 saver:
        - Signal the worker process to stop by setting the event and sending None.
        - Wait for the worker process to join.
        """
        self.stop_event.set()
        self.frame_queue.put(None)
        if self.worker_process is not None:
            self.worker_process.join()


def _ffmpeg_writer_process(output_file, width, height, fps, codec, preset, pix_fmt_in, frame_queue, stop_event, error_queue):
    """
    Worker process function for the asynchronous FFmpeg video writer.

    This process:
    - Launches FFmpeg as a subprocess.
    - Reads frames from the queue and writes them to FFmpeg stdin.
    - Monitors FFmpeg stderr for errors.
    - Shuts down gracefully when signaled.
    """

    # Construct FFmpeg command
    cmd = [
        'ffmpeg',
        '-y',  # Overwrite output files without asking
        '-f', 'rawvideo',
        '-pix_fmt', pix_fmt_in,
        '-s', f"{width}x{height}",
        '-r', str(fps),
        '-i', 'pipe:0',  # Input from stdin
        '-c:v', codec,
        '-preset', preset,
        output_file
    ]

    # Start FFmpeg subprocess
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        bufsize=10**7
    )

    def read_stderr():
        for line in process.stderr:
            line_decoded = line.decode('utf-8', errors='replace').strip()
            # Log all FFmpeg output
            sys.stderr.write(f"[FFmpeg] {line_decoded}\n")
            # Forward any error lines
            if "Error" in line_decoded or "error" in line_decoded:
                error_queue.put(line_decoded)

    # Start reading stderr in a local loop
    # For multiprocessing, a separate thread in the worker process could be used if needed,
    # but here we can read stderr line by line non-blockingly.
    # We'll do a non-blocking read approach:
    import threading
    stderr_thread = threading.Thread(target=read_stderr, daemon=True)
    stderr_thread.start()

    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            if frame is None:
                # End of stream
                break

            # Validate frame
            if pix_fmt_in == 'rgb24':
                expected_shape = (height, width, 3)
                if frame.shape != expected_shape:
                    error_queue.put(
                        f"Frame shape mismatch: got {frame.shape}, expected {expected_shape}."
                    )
                    continue
            elif pix_fmt_in == 'gray':
                expected_shape = (height, width)
                if frame.shape != expected_shape:
                    error_queue.put(
                        f"Frame shape mismatch: got {frame.shape}, expected {expected_shape}."
                    )
                    continue
            else:
                error_queue.put(f"Unsupported pix_fmt_in: {pix_fmt_in}.")
                continue

            if frame.dtype != np.uint8:
                error_queue.put(f"Frame dtype mismatch: got {frame.dtype}, expected uint8.")
                continue

            # Write to ffmpeg stdin
            try:
                process.stdin.write(frame.tobytes())
            except BrokenPipeError:
                error_queue.put("BrokenPipeError: FFmpeg process may have crashed or ended.")
                break

        if stop_event.is_set() and frame_queue.empty():
            break

    # Close stdin to signal EOF
    if process and process.stdin:
        process.stdin.close()

    stderr_thread.join()

    # Wait for FFmpeg to finish
    process.wait()
    if process.returncode != 0:
        error_queue.put(f"FFmpeg non-zero exit code: {process.returncode}")


class AsyncFFmpegVideoWriter:
    """
    An asynchronous FFmpeg video writer that offloads encoding to a separate process, leveraging NVIDIA GPU acceleration if desired.

    Features:
    - Uses a multiprocessing queue to send frames to a worker process.
    - The worker process spawns FFmpeg and feeds frames, allowing non-blocking operation in the main process.
    - Stderr output is monitored, and errors are captured and can be raised in the main process.

    Usage:
        with AsyncFFmpegVideoWriter("out.mp4", width=1920, height=1080, fps=30) as writer:
            # Push frames in a loop
            writer.write_frame(frame)
    """

    def __init__(
        self,
        output_file,
        width,
        height,
        fps=30,
        codec='h264_nvenc',
        preset='p4',
        pix_fmt_in='rgb24'
    ):
        """
        Initialize an asynchronous FFmpeg video writer using multiprocessing.

        Parameters
        ----------
        output_file : str
            Output filename.
        width : int
            Frame width in pixels.
        height : int
            Frame height in pixels.
        fps : int
            Frames per second.
        codec : str
            NVENC codec (e.g., 'h264_nvenc', 'hevc_nvenc', 'av1_nvenc').
        preset : str
            NVENC preset (e.g., 'p4' for balanced quality/performance).
        pix_fmt_in : str
            Input pixel format (default 'rgb24'). Use 'gray' for grayscale frames.
        """
        self.output_file = output_file
        self.width = width
        self.height = height
        self.fps = fps
        self.codec = codec
        self.preset = preset
        self.pix_fmt_in = pix_fmt_in

        self.frame_queue = multiprocessing.Queue()
        self.error_queue = multiprocessing.Queue()
        self.stop_event = multiprocessing.Event()
        self.worker_process = None

    def __enter__(self):
        """
        Start the FFmpeg writer process.
        """
        self.worker_process = multiprocessing.Process(
            target=_ffmpeg_writer_process,
            args=(
                self.output_file, self.width, self.height,
                self.fps, self.codec, self.preset, self.pix_fmt_in,
                self.frame_queue, self.stop_event, self.error_queue
            ),
            daemon=True
        )
        self.worker_process.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Stop the FFmpeg writer process, wait for it, and raise any encountered errors.
        """
        self.stop_event.set()
        self.frame_queue.put(None)
        self.worker_process.join()

        # Check for errors
        errors_found = False
        while not self.error_queue.empty():
            err = self.error_queue.get()
            sys.stderr.write(f"[FFmpeg Error] {err}\n")
            errors_found = True

        if errors_found:
            raise RuntimeError(f"FFmpeg encountered errors during encoding of {self.output_file}.")
        else:
            print(f"Video '{self.output_file}' saved successfully.")

    def write_frame(self, frame):
        """
        Enqueue a frame for writing.

        Parameters
        ----------
        frame : np.ndarray
            Frame data. Shape should match (height, width, 3) for 'rgb24' or (height, width) for 'gray'.
            dtype should be uint8.
        """
        if not isinstance(frame, np.ndarray):
            raise TypeError("Frame must be a numpy.ndarray.")
        if frame.dtype != np.uint8:
            raise TypeError("Frame dtype must be uint8.")
        self.frame_queue.put(frame)


class MultiStreamManager:
    """
    Manages multiple asynchronous video streams, each handled by its own process-backed FFmpeg writer.

    Features:
    - Adds multiple streams (each with independent configuration).
    - Provides a unified context manager to start/stop all streams.
    - Allows writing frames to a specific stream or to all streams simultaneously.
    """

    def __init__(self):
        self.writers = []

    def add_stream(
        self,
        output_file,
        width,
        height,
        fps=30,
        codec='h264_nvenc',
        preset='p4',
        pix_fmt_in='rgb24'
    ):
        """
        Add a video output stream configuration.

        Parameters
        ----------
        output_file : str
            Output filename for this video.
        width : int
            Frame width in pixels.
        height : int
            Frame height in pixels.
        fps : int
            Frames per second.
        codec : str
            NVENC codec (e.g., 'h264_nvenc', 'hevc_nvenc', 'av1_nvenc').
        preset : str
            NVENC preset (e.g., 'p4' for balanced quality/performance).
        pix_fmt_in : str
            Input pixel format (default 'rgb24'). Use 'gray' for grayscale frames.
        """
        writer = AsyncFFmpegVideoWriter(
            output_file=output_file,
            width=width,
            height=height,
            fps=fps,
            codec=codec,
            preset=preset,
            pix_fmt_in=pix_fmt_in
        )
        self.writers.append(writer)

    def __enter__(self):
        for writer in self.writers:
            writer.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        errors = []
        for writer in reversed(self.writers):
            try:
                writer.__exit__(exc_type, exc_val, exc_tb)
            except Exception as e:
                errors.append(e)
        if errors:
            # Raise the first error encountered
            raise errors[0]

    def write_frame_to_stream(self, index, frame):
        """
        Write a frame to a specific stream by index.

        Parameters
        ----------
        index : int
            Index of the stream in the `writers` list.
        frame : np.ndarray
            Frame data matching the stream's expected format.
        """
        if index < 0 or index >= len(self.writers):
            raise IndexError("Stream index out of range.")
        self.writers[index].write_frame(frame)

    def write_frame_to_all(self, frames):
        """
        Write corresponding frames to all streams.

        Parameters
        ----------
        frames : list of np.ndarray
            List of frames, one for each stream, in the order they were added.
        """
        if len(frames) != len(self.writers):
            raise ValueError("Number of frames does not match number of writers.")
        for frame, writer in zip(frames, self.writers):
            writer.write_frame(frame)
