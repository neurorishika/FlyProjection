import socket
import struct
import numpy as np
import os
import subprocess

class WS2812B_LEDController:
    """A server class to send image or test pattern commands to an LED matrix client using binary transmission."""
    
    def __init__(self, host='flyprojection-server', port=65432, rows=64, cols=8):
        """Initialize the server with the target IP, port, and matrix size."""
        self.host = host
        self.port = port
        self.rows = rows
        self.cols = cols
        self.connection = None

        # spawn a new terminal to run the LED client
        # os.system(f"gnome-terminal -- bash -c 'cd /mnt/sda1/Rishika/FlyProjection/flyprojection/controllers/;./start_LED_client.sh'")

    def __enter__(self):
        """Handle entering the 'with' block by establishing the connection."""
        self.connect_to_client()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Handle exiting the 'with' block by closing the connection."""
        self.close_connection()
        print("Successfully closed LED controller connection. Exiting WS2812B_LEDController Context.")

    def connect_to_client(self):
        """Establish a persistent connection to the LED client."""
        try:
            self.connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.connection.connect((self.host, self.port))
            # Send matrix dimensions on connection
            header = struct.pack('!III', 0, self.rows, self.cols)  # Message type 0 for setup
            self.connection.sendall(header)
            print(f"Connected to LED client at {self.host}:{self.port} with matrix size {self.rows}x{self.cols}")
        except ConnectionRefusedError:
            print("Error: Could not connect to the LED client.")
            self.connection = None
        except Exception as e:
            print(f"Unexpected error: {e}")
            self.connection = None

    def send_image(self, image_array):
        """Send an (rows, cols, 3) image array to the LED client as binary data."""
        if not self.connection:
            print("Error: No active connection to LED client.")
            return

        try:
            flat_image_data = image_array.flatten().astype(np.uint8).tobytes()
            header = struct.pack('!II', 1, len(flat_image_data))
            self.connection.sendall(header + flat_image_data)
            print("Sent image to LED client.")
        except Exception as e:
            print(f"Error sending image data: {e}")

    def send_test_pattern(self, pattern_name):
        """Send a test pattern command to the LED client as binary data."""
        if not self.connection:
            print("Error: No active connection to LED client.")
            return

        try:
            patterns = {"rgb_sweep": 1, "rainbow_sweep": 2}
            pattern_code = patterns.get(pattern_name, 0)
            header = struct.pack('!II', 2, pattern_code)
            self.connection.sendall(header)
            print(f"Sent test pattern '{pattern_name}' to LED client.")
        except Exception as e:
            print(f"Error sending test pattern command: {e}")

    def close_connection(self):
        """Close the connection to the LED client."""
        if self.connection:
            self.connection.close()

    def create_image(self, color):
        """Create a simple (rows, cols, 3) image with a uniform color."""
        image_array = np.full((self.rows, self.cols, 3), color, dtype=int)
        return image_array

    def create_stripes(self, stripe_width, color1, color2, orientation="vertical"):
        """Create horizontal or vertical stripe pattern."""
        image = np.zeros((self.rows, self.cols, 3), dtype=int)
        if orientation == "vertical":
            for row in range(self.rows):
                color = color1 if (row // stripe_width) % 2 == 0 else color2
                image[row, :, :] = color
        elif orientation == "vertical":
            for col in range(self.cols):
                color = color1 if (col // stripe_width) % 2 == 0 else color2
                image[:, col, :] = color
        return image

    def create_checkerboard(self, block_size, color1, color2):
        """Create a checkerboard pattern."""
        image = np.zeros((self.rows, self.cols, 3), dtype=int)
        for row in range(self.rows):
            for col in range(self.cols):
                if ((row // block_size) + (col // block_size)) % 2 == 0:
                    image[row, col, :] = color1
                else:
                    image[row, col, :] = color2
        return image

    def create_gradient(self, color_start, color_end, direction="vertical"):
        """Create a gradient pattern."""
        image = np.zeros((self.rows, self.cols, 3), dtype=int)
        for row in range(self.rows):
            for col in range(self.cols):
                t = col / (self.cols - 1) if direction == "vertical" else row / (self.rows - 1)
                color = [
                    int(color_start[i] + t * (color_end[i] - color_start[i])) for i in range(3)
                ]
                image[row, col, :] = color
        return image

    def create_radial(self, center, ring_width, color1, color2):
        """Create a radial pattern."""
        image = np.zeros((self.rows, self.cols, 3), dtype=int)
        for row in range(self.rows):
            for col in range(self.cols):
                distance = int(np.hypot(row - center[0], col - center[1]))
                color = color1 if (distance // ring_width) % 2 == 0 else color2
                image[row, col, :] = color
        return image

    def create_dots(self, dot_radius, spacing, dot_color, bg_color):
        """Create a dot grid pattern."""
        image = np.full((self.rows, self.cols, 3), bg_color, dtype=int)
        for row in range(dot_radius, self.rows, spacing):
            for col in range(dot_radius, self.cols, spacing):
                rr, cc = np.ogrid[:self.rows, :self.cols]
                mask = (rr - row)**2 + (cc - col)**2 <= dot_radius**2
                image[mask] = dot_color
        return image

    def create_waves(self, amplitude, frequency, color1, color2):
        """Create a sine-wave pattern."""
        image = np.zeros((self.rows, self.cols, 3), dtype=int)
        for row in range(self.rows):
            for col in range(self.cols):
                wave = int(amplitude * np.sin(2 * np.pi * frequency * col / self.cols))
                if abs(row - self.rows // 2) <= wave:
                    image[row, col, :] = color1
                else:
                    image[row, col, :] = color2
        return image

    def create_diagonal_stripes(self, stripe_width, color1, color2):
        """Create diagonal stripe pattern."""
        image = np.zeros((self.rows, self.cols, 3), dtype=int)
        for row in range(self.rows):
            for col in range(self.cols):
                if (row + col) // stripe_width % 2 == 0:
                    image[row, col, :] = color1
                else:
                    image[row, col, :] = color2
        return image

    def create_concentric_squares(self, square_width, color1, color2):
        """Create concentric squares pattern."""
        image = np.zeros((self.rows, self.cols, 3), dtype=int)
        for row in range(self.rows):
            for col in range(self.cols):
                layer = min(row, col, self.rows - row - 1, self.cols - col - 1) // square_width
                image[row, col, :] = color1 if layer % 2 == 0 else color2
        return image

    def create_random_noise(self, palette):
        """Create random noise with a specified palette."""
        image = np.zeros((self.rows, self.cols, 3), dtype=int)
        for row in range(self.rows):
            for col in range(self.cols):
                image[row, col, :] = palette[np.random.randint(len(palette))]
        return image

class KasaPowerController:
    """A server class to send power commands to a Kasa Smart Plug client using python-kasa."""

    def __init__(self, ip, default_state="off"):
        """Initialize the server with the target IP."""
        self.ip = ip
        self.default_state = default_state
    
    def turn_on(self):
        """Turn on the Kasa Smart Plug."""
        try:
            os.system(f"kasa --host {self.ip} on")
            print(f"Turned on Kasa Smart Plug at {self.ip}")
        except Exception as e:
            print(f"Error turning on Kasa Smart Plug: {e}")

    def turn_off(self):
        """Turn off the Kasa Smart Plug."""
        try:
            os.system(f"kasa --host {self.ip} off")
            print(f"Turned off Kasa Smart Plug at {self.ip}")
        except Exception as e:
            print(f"Error turning off Kasa Smart Plug: {e}")

    def __enter__(self):
        """Handle entering the 'with' block by establishing the connection."""
        if self.default_state == "on":
            self.turn_on()
        else:
            self.turn_off()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Handle exiting the 'with' block by turning off the plug."""
        self.turn_off()
        print("Successfully turned off Kasa Smart Plug. Exiting KasaPowerController Context.")