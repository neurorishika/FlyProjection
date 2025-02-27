import time
from flyprojection.controllers.led_server import WS2812B_LEDController
import matplotlib.pyplot as plt

if __name__ == "__main__":
    with WS2812B_LEDController(host='flyprojection-server', port=65432) as server:
        # Send a solid color image
        color_image = server.create_radial(
        center=(32, 3),
        ring_width=2,
        color1=(0, 0, 255),
        color2=(0, 0, 0),
        )
        # show image in matplotlib
        plt.imshow(color_image)
        plt.show()
        server.send_image(color_image)
        time.sleep(120)