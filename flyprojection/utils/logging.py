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
        self.log_queue = queue.Queue()
        self.log_thread = None
        self.logger = None

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
    - Exactly one of background subtraction OR difference coding may be enabled for a given dataset.
    - Can store per-frame metadata in a separate dataset.
    - Stores global metadata as HDF5 file attributes.
    - Uses a queue and a background thread to avoid blocking the main thread.

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
        global_metadata=None
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
                    "compression": str or None,
                    "compression_opts": int or None,
                    "background": ndarray or None,
                    "difference_coding": bool
                }
        metadata_config : dict, optional
            Configuration for per-frame metadata dataset (or None if not used).
        global_metadata : dict, optional
            Dictionary of key-value pairs stored as file attributes.
        """
        self.h5_filename = h5_filename
        os.makedirs(os.path.dirname(h5_filename), exist_ok=True)

        # Enforce exclusivity: cannot have both background subtraction and difference coding
        for name, cfg in datasets_config.items():
            background = cfg.get("background", None)
            diff_coding = cfg.get("difference_coding", False)
            if background is not None and diff_coding:
                raise ValueError(
                    f"Dataset '{name}' cannot have both difference coding and background subtraction enabled."
                )

        self.datasets_config = datasets_config
        self.metadata_config = metadata_config
        self.global_metadata = global_metadata or {}

        self.h5_file = None
        self.datasets = {}
        self.metadata_dset = None
        self.frame_queue = queue.Queue()
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
            comp = cfg.get("compression", None)
            comp_opts = cfg.get("compression_opts", None)

            maxshape = (None,) + shape
            chunks = (1,) + shape

            dset = self.h5_file.create_dataset(
                name,
                shape=(0,) + shape,
                maxshape=maxshape,
                dtype=dtype,
                chunks=chunks,
                compression=comp,
                compression_opts=comp_opts
            )
            cfg["prev_frame"] = None
            self.datasets[name] = dset

        # Create metadata dataset if needed
        if self.metadata_config is not None:
            md_dtype = self.metadata_config["dtype"]
            md_comp = self.metadata_config.get("compression", None)
            md_comp_opts = self.metadata_config.get("compression_opts", None)
            self.metadata_dset = self.h5_file.create_dataset(
                "metadata",
                shape=(0,),
                maxshape=(None,),
                dtype=md_dtype,
                chunks=(1,),
                compression=md_comp,
                compression_opts=md_comp_opts
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

    def _process_frame_data(self, dataset_name, frame_data):
        """
        Apply background subtraction OR difference coding if enabled for this dataset.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset being processed.
        frame_data : numpy.ndarray
            The raw frame data as a NumPy array.

        Returns
        -------
        numpy.ndarray
            Processed frame data ready for storage.
        """
        cfg = self.datasets_config[dataset_name]
        dtype = cfg["dtype"]
        data = frame_data.astype(dtype, copy=False)

        background = cfg.get("background", None)
        diff_coding = cfg.get("difference_coding", False)

        # If background is enabled, subtract it to store only differences from background
        if background is not None:
            bg = background.astype(np.int16, copy=False)
            diff = data.astype(np.int16) - bg
            diff = np.clip(diff, 0, np.iinfo(dtype).max).astype(dtype)
            data = diff
            # If this is the first frame, remember the processed frame
            if self.frames_written == 0:
                cfg["prev_frame"] = data.copy()

        # If difference coding is enabled, store frame differences relative to previous frame
        elif diff_coding:
            if self.frames_written == 0:
                # First frame stored as is
                cfg["prev_frame"] = data.copy()
            else:
                prev = cfg["prev_frame"].astype(np.int16)
                curr = data.astype(np.int16)
                diff = (curr - prev).astype(np.int16)
                data = diff
                # Update prev_frame as reconstructed current frame
                cfg["prev_frame"] = (prev + diff.astype(np.int16)).astype(dtype)

        # If neither background nor difference coding is used, do nothing special
        # Just store the frame as-is.

        return data

    def _writer(self):
        """
        Worker thread function:
        - Continuously reads items (metadata, data_dict) from the queue.
        - Processes each dataset's frame according to configured rules.
        - Appends frames and metadata to the HDF5 file.
        - Stops when a None item is encountered or stop_event is set.
        """
        while not self.stop_event.is_set():
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
                processed_data = self._process_frame_data(name, frame_data)

                new_size = self.frames_written + 1
                dset.resize((new_size,) + self.datasets_config[name]["shape"])
                dset[self.frames_written] = processed_data

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