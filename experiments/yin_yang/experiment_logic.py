def setup():
    global ADJUSTED_YING_CENTER, ADJUSTED_YANG_CENTER, ADJUSTED_RADIUS, ADJUSTED_YING_WIDTH, ADJUSTED_YANG_WIDTH, ADJUSTED_TRAIL_COLOR

    ADJUSTED_TRAIL_COLOR = hex_to_rgb(TRAIL_COLOR) if type(TRAIL_COLOR) == str else TRAIL_COLOR if type(TRAIL_COLOR) == tuple else (255, 0, 0)
    YING_CENTER = (TRAIL_CENTER[0], TRAIL_CENTER[1] - GAP_WIDTH/2)
    YANG_CENTER = (TRAIL_CENTER[0], TRAIL_CENTER[1] + GAP_WIDTH/2)
    ADJUSTED_YING_CENTER = relative_to_absolute(YING_CENTER[0], YING_CENTER[1], rig_config)
    ADJUSTED_YANG_CENTER = relative_to_absolute(YANG_CENTER[0], YANG_CENTER[1], rig_config)
    ADJUSTED_RADIUS = int(RADIUS * SCALE_FACTOR)
    ADJUSTED_YING_WIDTH = int(YING_WIDTH * SCALE_FACTOR)
    ADJUSTED_YANG_WIDTH = int(YANG_WIDTH * SCALE_FACTOR)
    print("Setup complete.")

def constants():
    return

def updates():
    global running, elapsed_time
    # draw a circular trail
    if elapsed_time < PRE_PERIOD_DURATION:
        pass
    elif elapsed_time < DURATION+PRE_PERIOD_DURATION:
        # DRAW YING ARCS
        pygame.draw.arc(
            screen, 
            ADJUSTED_TRAIL_COLOR, 
            (ADJUSTED_YING_CENTER[0] - ADJUSTED_RADIUS, ADJUSTED_YING_CENTER[1] - ADJUSTED_RADIUS, 2*ADJUSTED_RADIUS, 2*ADJUSTED_RADIUS),
            0, np.pi, ADJUSTED_YING_WIDTH
        )
        pygame.draw.arc(
            screen, 
            ADJUSTED_TRAIL_COLOR, 
            (ADJUSTED_YING_CENTER[0] - ADJUSTED_RADIUS, ADJUSTED_YING_CENTER[1] - ADJUSTED_RADIUS/2, ADJUSTED_RADIUS+ADJUSTED_YING_WIDTH/2, ADJUSTED_RADIUS),
            0,np.pi, ADJUSTED_YING_WIDTH
        )
        pygame.draw.arc(
            screen, 
            ADJUSTED_TRAIL_COLOR, 
            (ADJUSTED_YING_CENTER[0]-ADJUSTED_YING_WIDTH/2, ADJUSTED_YING_CENTER[1] - ADJUSTED_RADIUS/2, ADJUSTED_RADIUS+ADJUSTED_YING_WIDTH/2, ADJUSTED_RADIUS),
            np.pi, 2*np.pi, ADJUSTED_YING_WIDTH
        )
        # DRAW YANG ARCS
        pygame.draw.arc(
            screen, 
            ADJUSTED_TRAIL_COLOR, 
            (ADJUSTED_YANG_CENTER[0] - ADJUSTED_RADIUS, ADJUSTED_YANG_CENTER[1] - ADJUSTED_RADIUS, 2*ADJUSTED_RADIUS, 2*ADJUSTED_RADIUS),
            np.pi, 2*np.pi, ADJUSTED_YANG_WIDTH
        )
        pygame.draw.arc(
            screen, 
            ADJUSTED_TRAIL_COLOR, 
            (ADJUSTED_YANG_CENTER[0] - ADJUSTED_RADIUS, ADJUSTED_YANG_CENTER[1] - ADJUSTED_RADIUS/2, ADJUSTED_RADIUS + ADJUSTED_YANG_WIDTH/2, ADJUSTED_RADIUS),
            0, np.pi, ADJUSTED_YANG_WIDTH
        )
        pygame.draw.arc(
            screen, 
            ADJUSTED_TRAIL_COLOR, 
            (ADJUSTED_YANG_CENTER[0] - ADJUSTED_YANG_WIDTH/2, ADJUSTED_YANG_CENTER[1] - ADJUSTED_RADIUS/2, ADJUSTED_RADIUS + ADJUSTED_YANG_WIDTH/2, ADJUSTED_RADIUS),
            np.pi, 2*np.pi, ADJUSTED_YANG_WIDTH
        )
    else:
        running = False
    return