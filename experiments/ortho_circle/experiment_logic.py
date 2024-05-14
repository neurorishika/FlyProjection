def setup():
    global ADJUSTED_TRAIL_COLOR, ADJUSTED_A_CENTER, ADJUSTED_B_CENTER, ADJUSTED_RADIUS, ADJUSTED_A_WIDTH, ADJUSTED_B_WIDTH

    ADJUSTED_TRAIL_COLOR = hex_to_rgb(TRAIL_COLOR) if type(TRAIL_COLOR) == str else TRAIL_COLOR if type(TRAIL_COLOR) == tuple else (255, 0, 0)
    # orthogonal distance between A and B centers
    ORTHO_DIST = int(np.sqrt(2*RADIUS**2)*SCALE_FACTOR)
    A_CENTER = (TRAIL_CENTER[0], TRAIL_CENTER[1])
    B_CENTER = (TRAIL_CENTER[0], TRAIL_CENTER[1])
    ADJUSTED_A_CENTER = relative_to_absolute(A_CENTER[0], A_CENTER[1], rig_config)
    ADJUSTED_A_CENTER = (ADJUSTED_A_CENTER[0] - ORTHO_DIST/2, ADJUSTED_A_CENTER[1])
    ADJUSTED_B_CENTER = relative_to_absolute(B_CENTER[0], B_CENTER[1], rig_config)
    ADJUSTED_B_CENTER = (ADJUSTED_B_CENTER[0] + ORTHO_DIST/2, ADJUSTED_B_CENTER[1])
    ADJUSTED_RADIUS = int(RADIUS * SCALE_FACTOR)
    ADJUSTED_A_WIDTH = int(A_WIDTH * SCALE_FACTOR)
    ADJUSTED_B_WIDTH = int(B_WIDTH * SCALE_FACTOR)
    print("Setup complete.")

def constants():
    return

def updates():
    global running, elapsed_time
    # draw a circular trail
    if elapsed_time < PRE_PERIOD_DURATION:
        pass
    elif elapsed_time < DURATION+PRE_PERIOD_DURATION:
        # DRAW A CIRCLE
        pygame.draw.circle(screen, ADJUSTED_TRAIL_COLOR, ADJUSTED_A_CENTER, ADJUSTED_RADIUS, ADJUSTED_A_WIDTH)
        # DRAW B CIRCLE
        pygame.draw.circle(screen, ADJUSTED_TRAIL_COLOR, ADJUSTED_B_CENTER, ADJUSTED_RADIUS, ADJUSTED_B_WIDTH)
    else:
        running = False
    return