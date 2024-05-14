
def setup():
    global ADJUSTED_CENTER, ADJUSTED_TRAIL_MIN_RADIUS, ADJUSTED_TRAIL_MAX_RADIUS, ADJUSTED_TRAIL_STEP, ADJUSTED_TRAIL_WIDTH, ADJUSTED_TRAIL_COLOR

    ADJUSTED_CENTER = relative_to_absolute(TRAIL_CENTER[0], TRAIL_CENTER[1], rig_config)
    ADJUSTED_TRAIL_MIN_RADIUS = int(TRAIL_MIN_RADIUS * SCALE_FACTOR)
    ADJUSTED_TRAIL_MAX_RADIUS = int(TRAIL_MAX_RADIUS * SCALE_FACTOR)
    ADJUSTED_TRAIL_STEP = int(TRAIL_STEP * SCALE_FACTOR)
    ADJUSTED_TRAIL_WIDTH = int(TRAIL_WIDTH * SCALE_FACTOR)
    ADJUSTED_TRAIL_COLOR = [
        hex_to_rgb(TRAIL_COLOR[i]) if type(TRAIL_COLOR[i]) == str else TRAIL_COLOR[i] if type(TRAIL_COLOR[i]) == tuple else (255, 0, 0) for i in range(len(TRAIL_COLOR))
    ]
    print("Setup complete.")

def constants():
    return

def updates():
    global running, elapsed_time
    # draw a series of circles with increasing radii
    if elapsed_time < PHASE_1_DURATION:
        for radius in range(ADJUSTED_TRAIL_MIN_RADIUS, ADJUSTED_TRAIL_MAX_RADIUS + 1, ADJUSTED_TRAIL_STEP):
            pygame.draw.circle(screen, ADJUSTED_TRAIL_COLOR[0], ADJUSTED_CENTER, radius, ADJUSTED_TRAIL_WIDTH)
    elif elapsed_time < PHASE_1_DURATION + PHASE_2_DURATION:
        for radius in range(ADJUSTED_TRAIL_MIN_RADIUS, ADJUSTED_TRAIL_MAX_RADIUS + 1, ADJUSTED_TRAIL_STEP):
            pygame.draw.circle(screen, ADJUSTED_TRAIL_COLOR[1], ADJUSTED_CENTER, radius, ADJUSTED_TRAIL_WIDTH)
    elif elapsed_time < PHASE_1_DURATION + PHASE_2_DURATION + PHASE_3_DURATION:
        for radius in range(ADJUSTED_TRAIL_MIN_RADIUS, ADJUSTED_TRAIL_MAX_RADIUS + 1, ADJUSTED_TRAIL_STEP):
            pygame.draw.circle(screen, ADJUSTED_TRAIL_COLOR[2], ADJUSTED_CENTER, radius, ADJUSTED_TRAIL_WIDTH)
    else:
        running = False
    return