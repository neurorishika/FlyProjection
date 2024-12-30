import subprocess
import threading
import queue
import sys
import numpy as np

class AsyncFFmpegVideoWriter:
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
        """

        self.output_file = output_file
        self.width = width
        self.height = height
        self.fps = fps
        self.codec = codec
        self.preset = preset
        self.pix_fmt_in = pix_fmt_in

        self.frame_queue = queue.Queue()
        self.error_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.process = None
        self.writer_thread = None
        self.stderr_thread = None

    def __enter__(self):
        # Construct FFmpeg command
        cmd = [
            'ffmpeg',
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
            stderr=subprocess.PIPE,    # Capture FFmpeg's stderr for error handling
            bufsize=10**7  # Large buffer to prevent blocking
        )

        # Start writer thread
        self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.writer_thread.start()

        # Start stderr reading thread
        self.stderr_thread = threading.Thread(target=self._stderr_reader, daemon=True)
        self.stderr_thread.start()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Signal stop event
        self.stop_event.set()
        # Put None in queue to signal end of stream
        self.frame_queue.put(None)

        # Wait for threads to finish
        if self.writer_thread:
            self.writer_thread.join()
        if self.stderr_thread:
            self.stderr_thread.join()

        # Wait for FFmpeg to finish
        if self.process:
            self.process.wait()

        # Check for errors
        errors_found = False
        while not self.error_queue.empty():
            err = self.error_queue.get()
            sys.stderr.write(f"[FFmpeg Error] {err}\n")
            errors_found = True

        if self.process and self.process.returncode != 0:
            sys.stderr.write(f"[FFmpeg] Non-zero exit code: {self.process.returncode}\n")
            errors_found = True

        if errors_found:
            raise RuntimeError(f"FFmpeg encountered errors during encoding of {self.output_file}.")
        else:
            print(f"Video '{self.output_file}' saved successfully.")

    def _writer_loop(self):
        while not (self.stop_event.is_set() and self.frame_queue.empty()):
            try:
                frame = self.frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if frame is None:
                # End of stream
                break

            # Validate frame shape and format
            if self.pix_fmt_in == 'rgb24':
                expected_shape = (self.height, self.width, 3)
                if frame.shape != expected_shape:
                    self.error_queue.put(
                        f"Frame shape mismatch: got {frame.shape}, expected {expected_shape}."
                    )
                    continue
            elif self.pix_fmt_in == 'gray':
                expected_shape = (self.height, self.width)
                if frame.shape != expected_shape:
                    self.error_queue.put(
                        f"Frame shape mismatch: got {frame.shape}, expected {expected_shape}."
                    )
                    continue
            else:
                self.error_queue.put(f"Unsupported pix_fmt_in: {self.pix_fmt_in}.")
                continue

            # Ensure frame data is in uint8 format
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
            self.process.stdin.close()

    def _stderr_reader(self):
        for line in self.process.stderr:
            line_decoded = line.decode('utf-8', errors='replace').strip()
            # Log all FFmpeg output for debugging
            sys.stderr.write(f"[FFmpeg] {line_decoded}\n")
            # Forward any line containing 'error' or 'Error'
            if "Error" in line_decoded or "error" in line_decoded:
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


if __name__ == "__main__":
    # Initialize the manager
    manager = MultiStreamManager()

    # Add a grayscale stream
    manager.add_stream(
        output_file='grayscale_output.mp4',
        width=640,
        height=480,
        fps=30,
        codec='h264_nvenc',
        preset='p4',
        pix_fmt_in='gray'  # No filters
    )

    # Add an RGB stream
    manager.add_stream(
        output_file='rgb_output.mp4',
        width=1280,
        height=720,
        fps=30,
        codec='h264_nvenc',
        preset='p4',
        pix_fmt_in='rgb24'
    )

    # Use the with block to ensure proper cleanup
    try:
        with manager:
            # Simulate writing some frames
            num_frames = 60  # 2 seconds at 30 fps
            for i in range(num_frames):
                # Generate dummy grayscale frame
                frame_gray = np.full((480, 640), 128, dtype=np.uint8)  # Gray level 128

                # Generate dummy RGB frame and convert to BGR
                # Example: Creating a red, green, and blue bar
                frame_rgb = np.zeros((720, 1280, 3), dtype=np.uint8)
                frame_rgb[:, :426, 0] = 255  # Red
                frame_rgb[:, 426:853, 1] = 255  # Green
                frame_rgb[:, 853:, 2] = 255  # Blue

                # Write frames to their respective streams
                manager.write_frame_to_stream(0, frame_gray)  # Grayscale stream
                manager.write_frame_to_stream(1, frame_rgb)  # RGB stream
    except RuntimeError as e:
        print(f"RuntimeError: {e}")
    except Exception as e:
        print(f"Unhandled exception: {e}")

    print("Both videos processing completed.")
