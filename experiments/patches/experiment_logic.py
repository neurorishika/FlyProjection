import numpy as np

def setup():
    global PATCHES

    PATCHES = []
    for i in range(PATCH_COUNT):
        radial_position = np.random.uniform(PATCH_POSITIONS[0], PATCH_POSITIONS[1])*SCALE_FACTOR
        angular_position = np.random.uniform(0, 2*np.pi)
        arena_center = relative_to_absolute(0.5, 0.5, rig_config)
        center_x = int(radial_position * np.cos(angular_position) + arena_center[0])
        center_y = int(radial_position * np.sin(angular_position) + arena_center[1])
        radius = int(np.random.uniform(PATCH_RADII[0], PATCH_RADII[1])*SCALE_FACTOR)
        print(f"Patch {i+1}: ({center_x}, {center_y}), {radius}")
        PATCHES.append((center_x, center_y, radius, hex_to_rgb(PATCH_COLOR) if type(PATCH_COLOR) == str else PATCH_COLOR if type(PATCH_COLOR) == tuple else (255, 0, 0)))

    print("Setup complete.")

def constants():
    return

def updates():
    global running, elapsed_time
    # draw a circular trail
    if elapsed_time < PHASE_1_DURATION:
        pass # do nothing period
    elif elapsed_time < PHASE_1_DURATION + PHASE_2_DURATION:
        for patch in PATCHES:
            pygame.draw.circle(screen, patch[3], patch[:2], patch[2], 0)
    else:
        running = False
    return