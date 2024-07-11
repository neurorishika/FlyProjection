import numpy as np

def setup():
    global TIMESERIES, ADJUSTED_FLASH_COLOR, ADJUSTED_CENTER, ADJUSTED_RADIUS

    TIMESERIES = np.zeros((int(rig_config["FPS"]*PHASE_2_DURATION),))

    # compile and execute the pre code in the global namespace
    compiled = compile(PRE_CODE, "<string>", "exec")
    exec(compiled, globals())
    print("Pre code executed.")

    time_index = 0
    current_state = 0
    while time_index < len(TIMESERIES):
        # if the current state is off, turn it on
        if current_state == 0:
            duration = max(1,int(eval(FLASH_ON_DURATION)*rig_config["FPS"]))
            TIMESERIES[time_index:time_index+duration] = 1
            current_state = 1
            time_index += duration
        # if the current state is on, turn it off
        else:
            duration = int(eval(FLASH_OFF_DURATION)*rig_config["FPS"])
            current_state = 0
            time_index += duration

    print("Timeseries setup complete.")

    ADJUSTED_FLASH_COLOR = hex_to_rgb(FLASH_COLOR) if type(FLASH_COLOR) == str else FLASH_COLOR if type(FLASH_COLOR) == tuple else (255, 0, 0)
    ADJUSTED_CENTER = relative_to_absolute(0.5, 0.5, rig_config)
    ADJUSTED_RADIUS = int(3*SCALE_FACTOR)    

    print("Setup complete.")

def constants():
    return

def updates():
    global running, elapsed_time
    # draw a circular trail
    if elapsed_time < PHASE_1_DURATION:
        pass # do nothing period
    elif elapsed_time < PHASE_1_DURATION + PHASE_2_DURATION:
        # round the time to the previous frame count
        frame_time = int((elapsed_time - PHASE_1_DURATION) * rig_config["FPS"])
        frame_time = max(0, frame_time)
        frame_time = min(len(TIMESERIES)-1, frame_time)
        # draw the flash
        if TIMESERIES[frame_time] == 1:
            pygame.draw.circle(screen, ADJUSTED_FLASH_COLOR, ADJUSTED_CENTER, ADJUSTED_RADIUS, 0)
        frame_metadata.append({"time":elapsed_time, "frame":frame_time, "flash":TIMESERIES[frame_time]})

    else:
        running = False
    return