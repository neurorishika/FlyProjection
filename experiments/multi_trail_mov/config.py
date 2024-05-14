import numpy as np

# FUNDAMENTALS
ROI_TYPE = "circle" # "rectangle" or "circle"
BOUNDARY_COLOR = "#FFFFFF"
BOUNDARY_WIDTH = 1
BACKGROUND_COLOR = "#000000"

# TIME PARAMETERS
PHASE_DURATIONS = [1800, 1800, 1800] # in seconds
SPLIT_TIMES = np.cumsum(PHASE_DURATIONS)[: -1].tolist()

# TRAIL PARAMETERS
TRAIL_COLOR = ["#FF0000", "#0000FF"]
COLOR_START_INDEX = 0
TRAIL_WIDTH = 0.2 # in inches
TRAIL_MIN_RADIUS = 0.5 # in inches
TRAIL_MAX_RADIUS = 2 # in inches
TRAIL_STEP = 0.5 # in inches
TRAIL_CENTER = (0.5, 0.5)
VELOCITIES = [0.0, 0.1, -0.1] # in inches per second
