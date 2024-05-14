
def setup():
    assert len(PHASE_DURATIONS) == len(VELOCITIES)
    global ADJUSTED_CENTER, ADJUSTED_TRAIL_MIN_RADIUS, ADJUSTED_TRAIL_MAX_RADIUS, ADJUSTED_TRAIL_STEP, ADJUSTED_TRAIL_WIDTH, ADJUSTED_TRAIL_COLOR, ADJUSTED_VELOCITIES

    ADJUSTED_CENTER = relative_to_absolute(TRAIL_CENTER[0], TRAIL_CENTER[1], rig_config)
    ADJUSTED_TRAIL_MIN_RADIUS = int(TRAIL_MIN_RADIUS * SCALE_FACTOR)
    ADJUSTED_TRAIL_MAX_RADIUS = int(TRAIL_MAX_RADIUS * SCALE_FACTOR)
    ADJUSTED_TRAIL_STEP = int(TRAIL_STEP * SCALE_FACTOR)
    ADJUSTED_TRAIL_WIDTH = int(TRAIL_WIDTH * SCALE_FACTOR)
    ADJUSTED_TRAIL_COLOR = [
        hex_to_rgb(TRAIL_COLOR[i]) if type(TRAIL_COLOR[i]) == str else TRAIL_COLOR[i] if type(TRAIL_COLOR[i]) == tuple else (255, 0, 0) for i in range(len(TRAIL_COLOR))
    ]
    ADJUSTED_VELOCITIES = [VELOCITIES[i] * SCALE_FACTOR for i in range(len(VELOCITIES))]
    print("Setup complete.")

def constants():
    return

def updates():
    global running, elapsed_time
    # check if elapsed time exceeds the total duration
    if elapsed_time > sum(PHASE_DURATIONS):
        running = False
        return
    
    index = np.argmax(np.cumsum(PHASE_DURATIONS) > elapsed_time)

    # draw a series of circles with increasing radii but alternating colors
    shift = (elapsed_time*ADJUSTED_VELOCITIES[index])%ADJUSTED_TRAIL_STEP
    color_shift = int(COLOR_START_INDEX + (elapsed_time*ADJUSTED_VELOCITIES[index])//ADJUSTED_TRAIL_STEP)%len(ADJUSTED_TRAIL_COLOR)

    for radius in range(ADJUSTED_TRAIL_MIN_RADIUS, ADJUSTED_TRAIL_MAX_RADIUS + 1, ADJUSTED_TRAIL_STEP):
        if radius + shift < ADJUSTED_TRAIL_MIN_RADIUS or radius + shift > ADJUSTED_TRAIL_MAX_RADIUS:
            continue
        pygame.draw.circle(screen, ADJUSTED_TRAIL_COLOR[color_shift], ADJUSTED_CENTER, radius + shift, ADJUSTED_TRAIL_WIDTH)
        color_shift = (color_shift + 1) % len(ADJUSTED_TRAIL_COLOR)
    return