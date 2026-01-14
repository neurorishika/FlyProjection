import time
from flyprojection.controllers.led_server import WS2812B_LEDController
import matplotlib.pyplot as plt
import cv2
import numpy as np

if __name__ == "__main__":

    with WS2812B_LEDController(host="flyprojection-server", port=65432) as server:
        try:
            # Send a solid color image
            color_image = server.create_radial(
                center=(32, 3),
                ring_width=2,
                color1=(0, 0, 50),
                color2=(0, 0, 0),
            )
            # create window to display the image using OpenCV
            # cv2.namedWindow("LED Pattern", cv2.WINDOW_NORMAL)
            # cv2.resizeWindow("LED Pattern", 800, 600)
            # cv2.imshow("LED Pattern", color_image.astype("uint8"))
            # cv2.waitKey(1)
            while True:
                # roll the image to create a moving effect
                color_image = np.roll(color_image, shift=1, axis=0)
                # change color
                color_image = np.roll(color_image, shift=1, axis=2)
                server.send_image(color_image)
                # cv2.imshow("LED Pattern", color_image.astype("uint8"))
                # cv2.waitKey(1)
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Interrupted by user. Exiting...")
        finally:
            # cv2.destroyAllWindows()
            server.close_connection()
            print(
                "Successfully closed LED controller connection. Exiting WS2812B_LEDController Context."
            )
