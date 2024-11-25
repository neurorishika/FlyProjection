import numpy as np
import time
import pygame  # Import pygame for event handling
from flyprojection.controllers.basler_camera import BaslerCamera
from flyprojection.projection.artist import Artist, Drawing

# Initialize the camera
camera = BaslerCamera(
    index=0,
    FPS=100,
    EXPOSURE_TIME=9000.0,
    GAIN=0.0,
    WIDTH=2048,
    HEIGHT=2048,
    OFFSETX=248,
    OFFSETY=0,
    TRIGGER_MODE="Continuous",
    CAMERA_FORMAT="Mono8",
    record_video=False,
    video_output_path=None,
    video_output_name=None,
    lossless=True,
    debug=False,
)

# Start the camera
camera.start()

rig_config_path = "/mnt/sda1/Rishika/FlyProjection/configs/rig_config.json"

# Initialize the Artist
artist = Artist(camera, rig_config_path, method="map")

# Define the pinwheel drawing function
def add_pinwheel(drawing, center, radius, num_spokes=20, color=(1, 0, 0), start_offset=0):
    for i in range(num_spokes):
        start_angle = (i * (2 * np.pi / num_spokes) + start_offset) % (2 * np.pi)
        end_angle = ((i + 1) * (2 * np.pi / num_spokes) + start_offset) % (2 * np.pi)
        drawing.add_arc(
            center,
            radius,
            start_angle,
            end_angle,
            color=color if i % 2 == 0 else (0, 0, 0),
            line_width=0,
            fill=True,
        )



pinwheel_dimension = min(artist.camera_width, artist.camera_height)//4
blueprint = Drawing()
add_pinwheel(blueprint, (pinwheel_dimension//2, pinwheel_dimension//2), pinwheel_dimension//2, num_spokes=20, color=(0, 0, 1), start_offset=0)

# Get or create the sprite
pinwheel_sprite = blueprint.create_sprite(
    pinwheel_dimension,
    pinwheel_dimension,
    blueprint.get_drawing_function(),
)


# Initialize variables for rotation
angle = 0  # Starting angle in radians
angular_speed = np.pi / 4  # Rotation speed in radians per second (e.g., 45 degrees per second)
last_time = time.time()

times = []

max_time = 10  # Run the loop for 10 seconds

try:
    while True:
        current_time = time.time()
        delta_time = current_time - last_time
        last_time = current_time
        times.append(delta_time)

        # Update the rotation angle
        angle += angular_speed * delta_time  # Update angle based on time elapsed

        start_time = time.time()

        # Create a new Drawing instance
        drawing = Drawing()

        # Add pinwheel sprite to the drawing
        drawing.add_sprite(
            pinwheel_sprite,
            center=(artist.camera_width / 2, artist.camera_height / 2),
            rotation=angle,
        )


        # Get the drawing function
        draw_fn = drawing.get_drawing_function()
        time_generating = time.time() - start_time

        # print(f"Time taken to generate drawing: {time_generating:0.6f} s")

        # Draw the geometry using Artist
        artist.draw_geometry(draw_fn, debug=False)

        if current_time - start_time > max_time:
            break

except Exception as e:
    print("Exiting due to exception: ", e)
finally:
    artist.close()
    camera.stop()
    print(f"Average frame rate: {1 / np.mean(times)} Hz")
