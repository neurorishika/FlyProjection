# experiment_logic.py

from PySide2 import QtWidgets, QtGui, QtCore
import pyqtgraph as pg
import time
import numpy as np
from flyprojection.utils import hex_to_rgb

def setup(view, config, rig_config):
    global TIMESERIES, ADJUSTED_FLASH_COLOR, ADJUSTED_CENTER, ADJUSTED_RADIUS, start_time, running, elapsed_time
    global PHASE_1_DURATION, PHASE_2_DURATION
    running = True
    elapsed_time = 0

    # Execute PRE_CODE in the global namespace
    compiled = compile(config.PRE_CODE, "<string>", "exec")
    exec(compiled, globals())

    # Setup timeseries as before
    PHASE_1_DURATION = config.PHASE_1_DURATION
    PHASE_2_DURATION = config.PHASE_2_DURATION
    TIMESERIES = np.zeros((int(rig_config["FPS"] * PHASE_2_DURATION),))

    time_index = 0
    current_state = 0
    while time_index < len(TIMESERIES):
        if current_state == 0:
            duration = max(1, int(eval(config.FLASH_ON_DURATION) * rig_config["FPS"]))
            TIMESERIES[time_index:time_index + duration] = 1
            current_state = 1
            time_index += duration
        else:
            duration = int(eval(config.FLASH_OFF_DURATION) * rig_config["FPS"])
            current_state = 0
            time_index += duration

    print(f"Timeseries length: {len(TIMESERIES)}")
    print(f"Timeseries: {TIMESERIES}")

    # Prepare colors and positions
    ADJUSTED_FLASH_COLOR = hex_to_rgb(config.FLASH_COLOR)
    ADJUSTED_FLASH_COLOR = QtGui.QColor(*ADJUSTED_FLASH_COLOR)

    # Assuming center at the middle of the screen
    screen_width = rig_config['projector_width']
    screen_height = rig_config['projector_height']
    ADJUSTED_CENTER = (screen_width / 2, screen_height / 2)
    SCALE_FACTOR = screen_width / rig_config['physical_x']
    ADJUSTED_RADIUS = int(3 * SCALE_FACTOR)

    start_time = time.time()

def updates(view, rig_config, elapsed_time):
    global TIMESERIES, running, ADJUSTED_FLASH_COLOR, ADJUSTED_CENTER, ADJUSTED_RADIUS

    # Draw stimuli based on the current state
    if elapsed_time < PHASE_1_DURATION:
        # Do nothing during phase 1
        pass
    elif elapsed_time < PHASE_1_DURATION + PHASE_2_DURATION:
        
        frame_time = int((elapsed_time - PHASE_1_DURATION) * rig_config["FPS"])
        frame_time = max(0, min(len(TIMESERIES) - 1, frame_time))
        if TIMESERIES[frame_time] == 1:
            # set background color to red
            view.setBackgroundColor('r')
            # Draw flash
            # circle = QtWidgets.QGraphicsEllipseItem(
            #     ADJUSTED_CENTER[0] - ADJUSTED_RADIUS,
            #     ADJUSTED_CENTER[1] - ADJUSTED_RADIUS,
            #     ADJUSTED_RADIUS * 2,
            #     ADJUSTED_RADIUS * 2
            # )
            # circle.setBrush(pg.mkBrush(ADJUSTED_FLASH_COLOR))
            # circle.setPen(pg.mkPen(None))
            # view.addItem(circle)
        else:
            # set background color to black
            view.setBackgroundColor('k')
            pass
    else:
        running = False