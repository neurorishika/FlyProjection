import logging
import logging.handlers
import queue
import threading
import signal
import sys
import os
import numpy as np
import h5py
import hdf5plugin
import time
import subprocess

class AsyncLogger:
    """
    An asynchronous logger that writes log messages to a file using a background thread.

    This class:
    - Uses a named logger (not the root logger) to avoid interfering with global logging settings.
    - Sends log records through a thread-safe queue to a background thread.
    - Writes logs to a file via a file handler from the background thread.

    Usage:
        with AsyncLogger(log_file="app.log", logger_name="my_logger") as logger:
            logger.info("This is a log message.")

    On exiting the context, the logger flushes and closes all handlers.
    """

    def __init__(self, log_file, logger_name='async_logger', level=logging.INFO, display=True):
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
        display : bool
            Whether to display log messages on the console.
        """
        self.log_file = log_file
        self.logger_name = logger_name
        self.level = level
        self.log_queue = queue.Queue()
        self.log_thread = None
        self.logger = None
        self.display = display

    def __enter__(self):
        """
        Enter the runtime context for the asynchronous logger.

        Sets up the logger, file handler, queue handler, and starts the background thread.
        """
        # Create and configure the named logger
        self.logger = logging.getLogger(self.logger_name)
        self.logger.setLevel(self.level)

        # Create a file handler to write logs to disk
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

        # Create a QueueHandler that sends records to the log queue
        queue_handler = logging.handlers.QueueHandler(self.log_queue)
        self.logger.addHandler(queue_handler)

        # Start the background thread that reads from the queue and writes to file
        self.log_thread = threading.Thread(
            target=self._queue_listener,
            args=(self.log_queue, file_handler),
            daemon=True
        )
        self.log_thread.start()

        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the runtime context for the asynchronous logger.

        Signals the background thread to stop, waits for it to finish, and cleans up handlers.
        """
        # Signal the listener thread to stop
        self.log_queue.put(None)
        self.log_thread.join()

        # Remove all handlers and close them
        handlers = self.logger.handlers[:]
        for h in handlers:
            self.logger.removeHandler(h)
            h.close()

        print("Successfully closed Logger. Exiting AsyncLogger Context.")

    def _queue_listener(self, log_queue, file_handler):
        """
        Continuously reads log records from the queue and writes them to the file handler.

        Parameters
        ----------
        log_queue : queue.Queue
            Queue from which log records are read.
        file_handler : logging.FileHandler
            Handler used to write records to the log file.
        """
        while True:
            try:
                record = log_queue.get()
                if record is None:
                    break
                file_handler.emit(record)
            except Exception as e:
                print(f"Error in log listener: {e}", file=sys.stderr)



class AsyncHDF5Saver:
    """
    An asynchronous HDF5 saver that writes frames (and optional metadata) to an HDF5 file using a background thread.

    Features:
    - Multiple datasets can be defined, each with its own shape, dtype, and compression.
    - Can store per-frame metadata in a separate dataset.
    - Stores global metadata as HDF5 file attributes.
    - Uses an unbounded queue and a background thread to avoid blocking the main thread.
    - Supports advanced compression methods via hdf5plugin.
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
            Dictionary of dataset configurations. Keys are dataset names, values are dicts:
                {
                    "shape": tuple,
                    "dtype": np.dtype or type,
                    "compression": hdf5plugin compression object or None
                }
        metadata_config : dict, optional
            Configuration for per-frame metadata dataset (or None if not used).
        global_metadata : dict, optional
            Dictionary of key-value pairs stored as file attributes.
        """
        self.h5_filename = h5_filename
        os.makedirs(os.path.dirname(h5_filename), exist_ok=True)

        self.datasets_config = datasets_config
        self.metadata_config = metadata_config
        self.global_metadata = global_metadata or {}

        self.h5_file = None
        self.datasets = {}
        self.metadata_dset = None
        self.frame_queue = queue.Queue()  # Unbounded queue
        self.stop_event = threading.Event()
        self.worker_thread = None
        self.frames_written = 0

    def __enter__(self):
        """
        Enter the runtime context, opening the HDF5 file, creating datasets, and starting the worker thread.
        """
        self.h5_file = h5py.File(self.h5_filename, 'w')

        # Create all datasets as unlimited and compressed as per configuration
        for name, cfg in self.datasets_config.items():
            shape = cfg["shape"]
            dtype = cfg["dtype"]
            compression = cfg.get("compression", None)

            maxshape = (None,) + shape
            chunks = (1,) + shape  # Multiple frames per chunk

            dset = self.h5_file.create_dataset(
                name,
                shape=(0,) + shape,
                maxshape=maxshape,
                dtype=dtype,
                chunks=chunks,
                compression=compression  # This is a hdf5plugin compression object
            )
            self.datasets[name] = dset

        # Create metadata dataset if needed
        if self.metadata_config is not None:
            md_dtype = self.metadata_config["dtype"]
            compression = self.metadata_config.get("compression", None)

            metadata_chunks = (1,)  # Assuming one metadata entry per frame

            self.metadata_dset = self.h5_file.create_dataset(
                "metadata",
                shape=(0,),
                maxshape=(None,),
                dtype=md_dtype,
                chunks=metadata_chunks,
                compression=compression  # This is a hdf5plugin compression object
            )

        # Store global metadata as attributes
        for k, v in self.global_metadata.items():
            self.h5_file.attrs[k] = v

        # Start the background worker thread
        self.worker_thread = threading.Thread(target=self._writer, daemon=True)
        self.worker_thread.start()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the runtime context, triggering a graceful shutdown of the worker thread and closing the HDF5 file.
        """
        self.shutdown()
        print("Successfully closed HDF5 Saver. Exiting AsyncHDF5Saver Context.")

    def _writer(self):
        """
        Worker thread function:
        - Continuously reads items (metadata, data_dict) from the queue.
        - Appends frames and metadata to the HDF5 file.
        - Stops when a None item is encountered or stop_event is set.
        """
        while not (self.stop_event.is_set() and self.frame_queue.empty()):
            try:
                item = self.frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if item is None:
                break

            metadata, data_dict = item

            # Write each dataset
            for name, dset in self.datasets.items():
                frame_data = data_dict[name]

                new_size = self.frames_written + 1
                # Resize dataset to accommodate the new frame
                dset.resize((new_size,) + self.datasets_config[name]["shape"])
                dset[self.frames_written] = frame_data

            # Write metadata if available
            if self.metadata_dset is not None:
                new_size = self.frames_written + 1
                self.metadata_dset.resize((new_size,))
                md_entry = np.zeros(1, dtype=self.metadata_config["dtype"])
                for field in self.metadata_config["dtype"].names:
                    md_entry[field] = metadata.get(field, 0)
                self.metadata_dset[self.frames_written] = md_entry[0]

            self.frames_written += 1
            self.frame_queue.task_done()

    def save_frame_async(self, metadata, **data_dict):
        """
        Enqueue a frame for asynchronous saving.

        Parameters
        ----------
        metadata : dict
            Dictionary with per-frame metadata fields (if any).
        data_dict : dict
            Keyword arguments mapping dataset_name -> frame_array.
            All datasets in datasets_config must be provided.
        """
        # Validate that all configured datasets are present
        for name in self.datasets_config:
            if name not in data_dict:
                raise ValueError(f"Missing data for dataset '{name}' in save_frame_async call.")

        # Enqueue the data without blocking
        self.frame_queue.put((metadata, data_dict))

    def shutdown(self):
        """
        Gracefully shutdown the HDF5 saver:
        - Signal the worker thread to stop by placing None in the queue.
        - Join the worker thread.
        - Flush and close the HDF5 file.
        """
        self.stop_event.set()
        self.frame_queue.put(None)
        if self.worker_thread is not None:
            self.worker_thread.join()
        if self.h5_file is not None:
            self.h5_file.flush()
            self.h5_file.close()
class AsyncFFmpegVideoWriter:
    def __init__(
        self,
        output_file,
        width,
        height,
        fps=30,
        codec='h264_nvenc',
        preset='p4',
        pix_fmt_in='rgb24',
        stop_event=None,
        log_file=None,
    ):
        """
        Initialize an asynchronous FFmpeg video writer using NVIDIA GPU acceleration.

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
        stop_event : threading.Event
            Optional stop event to signal the writer to stop.
        log_file : str
            Optional path to a file where FFmpeg logs will be written.
        """
        self.output_file = output_file
        self.width = width
        self.height = height
        self.fps = fps
        self.codec = codec
        self.preset = preset
        self.pix_fmt_in = pix_fmt_in
        self.stop_event = threading.Event() if stop_event is None else stop_event
        self.log_file = log_file

        self.frame_queue = queue.Queue()
        self.error_queue = queue.Queue()
        self.process = None
        self.writer_thread = None
        self.stderr_thread = None
        self.log_file_handle = None

    def __enter__(self):
        # Open log file if provided
        if self.log_file:
            self.log_file_handle = open(self.log_file, 'w')

        # Construct FFmpeg command
        cmd = [
            'ffmpeg',
            '-loglevel', 'debug',  # Suppress FFmpeg's stderr
            '-y',  # Overwrite output files without asking
            '-f', 'rawvideo',
            '-pix_fmt', self.pix_fmt_in,
            '-s', f"{self.width}x{self.height}",
            '-r', str(self.fps),
            '-i', 'pipe:0',  # Input comes from stdin
            '-c:v', self.codec,
            '-preset', self.preset,
            self.output_file
        ]

        # Start FFmpeg subprocess
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,  # Suppress FFmpeg's stdout
            stderr=self.log_file_handle or subprocess.PIPE,  # Log to file if provided, else capture stderr
            bufsize=10**7  # Large buffer to prevent blocking
        )

        # Start writer thread
        self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.writer_thread.start()

        # Start stderr reading thread if no log file is used
        if not self.log_file_handle:
            self.stderr_thread = threading.Thread(target=self._stderr_reader, daemon=True)
            self.stderr_thread.start()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Signal stop event
        self.stop_event.set()

        # Wait for writer thread to finish processing all frames
        self.frame_queue.put(None)  # Signal end of input
        if self.writer_thread:
            self.writer_thread.join()

        # Wait for stderr reader to finish
        if self.stderr_thread:
            self.stderr_thread.join()

        # Send 'q' to FFmpeg's stdin to request graceful termination
        if self.process and self.process.stdin:
            try:
                self.process.stdin.write(b'q')
                self.process.stdin.flush()
            except Exception as e:
                self.error_queue.put(f"Failed to send termination signal to FFmpeg: {str(e)}")

        # Wait for FFmpeg to finish encoding
        if self.process:
            try:
                self.process.wait(timeout=300)  # Wait up to 5 minutes
            except subprocess.TimeoutExpired:
                print("FFmpeg did not terminate in time. Attempting retries...")

            # Retry mechanism for FFmpeg finalization
            for _ in range(5):  # Retry up to 5 times
                if self.process.poll() is not None:  # Process has terminated
                    break
                print("Retrying FFmpeg finalization...")
                try:
                    self.process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    continue

        # Close log file if used
        if self.log_file_handle:
            self.log_file_handle.close()

        # Ensure FFmpeg terminated successfully
        success_detected = False
        if self.process and self.process.returncode == 0:
            success_detected = True

        # Check error queue for critical errors
        errors_found = False
        while not self.error_queue.empty():
            err = self.error_queue.get()
            if not self.log_file_handle:
                sys.stderr.write(f"[FFmpeg Error] {err}\n")
            errors_found = True
        
        if self.process and self.process.returncode not in [0, 255]:  # Treat return code 255 as a warning
            errors_found = True

        if errors_found and not success_detected:
            print(f"Video '{self.output_file}' saved with errors. Check logs at '{self.log_file}'.")
        elif not errors_found and success_detected:
            print(f"Video '{self.output_file}' saved successfully.")
        else:
            print(f"Video '{self.output_file}' completed with warnings. Check logs at '{self.log_file}'.")

    def _writer_loop(self):
        while True:
            try:
                frame = self.frame_queue.get(timeout=0.5)
            except queue.Empty:
                if self.stop_event.is_set() and self.frame_queue.empty():
                    break  # Exit loop if stop event is set and queue is empty
                continue

            if frame is None:
                break  # End of stream

            # Validate and write the frame
            if self.pix_fmt_in == 'rgb24':
                expected_shape = (self.height, self.width, 3)
            elif self.pix_fmt_in == 'gray':
                expected_shape = (self.height, self.width)
            else:
                self.error_queue.put(f"Unsupported pix_fmt_in: {self.pix_fmt_in}.")
                continue

            if frame.shape != expected_shape:
                self.error_queue.put(
                    f"Frame shape mismatch: got {frame.shape}, expected {expected_shape}."
                )
                continue

            if frame.dtype != np.uint8:
                self.error_queue.put(f"Frame dtype mismatch: got {frame.dtype}, expected uint8.")
                continue

            try:
                self.process.stdin.write(frame.tobytes())
            except BrokenPipeError:
                self.error_queue.put("BrokenPipeError: FFmpeg process may have crashed or ended.")
                break

            self.frame_queue.task_done()

        # Close stdin to signal EOF to FFmpeg
        if self.process and self.process.stdin:
            try:
                self.process.stdin.close()
            except Exception as e:
                self.error_queue.put(f"Error closing stdin: {e}")

    def _stderr_reader(self):
        for line in self.process.stderr:
            if self.stop_event.is_set():
                break
            line_decoded = line.decode('utf-8', errors='replace').strip()
            if "error" in line_decoded.lower() or "Error" in line_decoded:
                # Log only critical errors
                if "EOF while reading input" not in line_decoded and "PACKET SIZE" not in line_decoded:
                    self.error_queue.put(line_decoded)

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
    def __init__(self, stop_event = None):
        self.writers = []
        self.stop_event = stop_event

    def add_stream(
        self,
        output_file,
        width,
        height,
        fps=30,
        codec='h264_nvenc',
        preset='p4',
        pix_fmt_in='rgb24',
        log_file=None
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
        log_file : str
            Optional path to a file where FFmpeg logs will be written.
        """
        writer = AsyncFFmpegVideoWriter(
            output_file=output_file,
            width=width,
            height=height,
            fps=fps,
            codec=codec,
            preset=preset,
            pix_fmt_in=pix_fmt_in,
            stop_event=self.stop_event,
            log_file=log_file
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
                if self.stop_event.is_set():
                    print("Stopping MultiStreamManager due to stop event.")
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