import time
from flyprojection.controllers.led_server import LEDPanelController
import matplotlib.pyplot as plt

if __name__ == "__main__":
    with LEDPanelController(host='flyprojection-server', port=65432) as server:
        # Send a solid color image
        color_image = server.create_stripes(
        stripe_width=2,
        color1=(0, 0, 50),
        color2=(0, 0, 0),
        orientation="vertical",
        )
        # show image in matplotlib
        plt.imshow(color_image)
        plt.show()
        server.send_image(color_image)
        time.sleep(10)