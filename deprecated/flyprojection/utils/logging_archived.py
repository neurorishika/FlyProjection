import logging
import logging.handlers
import queue
import threading
import signal
import sys

# Function to configure asynchronous logging
def setup_async_logger(log_file):
    # Create a thread-safe queue for logging messages
    log_queue = queue.Queue()

    # Create a handler that writes log messages to a file
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # Create a QueueHandler to send messages to the log queue
    queue_handler = logging.handlers.QueueHandler(log_queue)

    # Configure the root logger to use the queue handler
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Set the default log level
    logger.addHandler(queue_handler)

    # Create a thread that reads from the log queue and writes to the file
    log_thread = threading.Thread(target=queue_listener, args=(log_queue, file_handler), daemon=True)
    log_thread.start()

    # Store queue and thread for graceful shutdown
    logger.log_queue = log_queue
    logger.log_thread = log_thread

    return logger

# Function to listen for log messages in the queue
def queue_listener(log_queue, file_handler):
    while True:
        try:
            # Get log record from the queue
            record = log_queue.get()
            if record is None:
                break  # Exit the listener thread if a None record is received
            # Write the log record to the file
            file_handler.emit(record)
        except Exception as e:
            print(f"Error in log listener: {e}")

# Graceful shutdown function
def shutdown_logger(logger):
    logger.log_queue.put(None)  # Signal the listener thread to exit
    logger.log_thread.join()   # Wait for the thread to finish
    print("Logger shutdown gracefully.")

# Signal handler for graceful shutdown on interrupt
def signal_handler(sig, frame):
    print("Interrupt received. Shutting down logger...")
    shutdown_logger(logging.getLogger())
    sys.exit(0)