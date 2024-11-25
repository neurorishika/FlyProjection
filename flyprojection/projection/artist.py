import os
import sys
import json
import numpy as np
import pygame
import cairo
import cv2
from flyprojection.utils import *
import time

# Import torch and kornia if available
try:
    import torch
    import kornia
except ImportError:
    torch = None
    kornia = None
    print("Warning: PyTorch and Kornia are not installed. 'kornia' method will not be available.")

class Artist:
    def __init__(self, camera, rig_config, method='classic', fps=200):
        """
        Initializes the Artist class.

        Parameters:
        - camera: The camera object.
        - rig_config: Path to the rig_config.json file or the rig_config dictionary.
        - method: The method to use for transformations ('classic', 'kornia', 'map').
        - fps: The frame rate for the display window.
        """
        self.camera = camera
        self.method = method.lower()
        self.fps = fps

        # Load rig configuration
        if isinstance(rig_config, dict):
            self.rig_config = rig_config
        elif isinstance(rig_config, str):
            with open(rig_config_path, 'r') as f:
                self.rig_config = json.load(f)
        else:
            raise ValueError("Invalid rig_config. Provide a path to the rig_config.json file or the rig_config dictionary.")

        # Extract necessary parameters from rig_config
        self.projector_width = self.rig_config['projector_width']
        self.projector_height = self.rig_config['projector_height']
        self.camera_width = self.camera.WIDTH
        self.camera_height = self.camera.HEIGHT

        # Load the camera to projector transformation matrices
        self.H_refined = np.array(self.rig_config['H_refined'])
        self.H_projector_distortion_corrected = np.array(self.rig_config['H_projector_distortion_corrected'])
        self.projector_correction_method = self.rig_config.get('projector_correction_method', 'distortion')
        self.camera_correction_method = self.rig_config.get('camera_correction_method', 'none')

        # Load interpolated markers for distortion correction if needed
        if self.projector_correction_method == 'distortion':
            self.interpolated_projector_space_detail_markers = np.array(
                self.rig_config['interpolated_projector_space_detail_markers']
            )
            self.distortion_corrected_projection_space_detail_markers = np.array(
                self.rig_config['distortion_corrected_projection_space_detail_markers']
            )

        # Initialize Pygame
        os.environ['SDL_VIDEO_WINDOW_POS'] = "0,0"
        pygame.init()

        # Set up the display window
        self.screen = pygame.display.set_mode(
            (self.projector_width, self.projector_height),
            pygame.NOFRAME | pygame.HWSURFACE | pygame.DOUBLEBUF
        )

        # Store the last image displayed
        self.last_image = None

        # Prepare a Cairo surface for drawing in camera space
        self.camera_surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.camera_width, self.camera_height)
        self.camera_context = cairo.Context(self.camera_surface)

        # Prepare the clock for frame rate control
        self.clock = pygame.time.Clock()

        # Set up method-specific configurations
        if self.method == 'kornia':
            if torch is None or kornia is None:
                raise ImportError("PyTorch and Kornia are required for the 'kornia' method.")
            # Set up device for PyTorch
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # Precompute transformation matrices for Kornia
            self.H_refined_tensor = torch.tensor(self.H_refined, dtype=torch.float32, device=self.device)
            self.H_projector_distortion_corrected_tensor = torch.tensor(
                self.H_projector_distortion_corrected, dtype=torch.float32, device=self.device
            )
            if self.projector_correction_method == 'distortion':
                # Convert control points to tensors
                self.X = torch.tensor(
                    self.distortion_corrected_projection_space_detail_markers, dtype=torch.float32, device=self.device
                )
                self.Y = torch.tensor(
                    self.interpolated_projector_space_detail_markers, dtype=torch.float32, device=self.device
                )
        elif self.method == 'map':
            # Load the precomputed remap maps
            self.H_refined_mapx = np.array(self.rig_config['H_refined_mapx'], dtype=np.float32)
            self.H_refined_mapy = np.array(self.rig_config['H_refined_mapy'], dtype=np.float32)
            if self.projector_correction_method == 'homography':
                self.H_projector_distortion_corrected_mapx = np.array(
                    self.rig_config['H_projector_distortion_corrected_homography_mapx'], dtype=np.float32
                )
                self.H_projector_distortion_corrected_mapy = np.array(
                    self.rig_config['H_projector_distortion_corrected_homography_mapy'], dtype=np.float32
                )
            elif self.projector_correction_method == 'distortion':
                self.H_projector_distortion_corrected_mapx = np.array(
                    self.rig_config['H_projector_distortion_corrected_distortion_mapx'], dtype=np.float32
                )
                self.H_projector_distortion_corrected_mapy = np.array(
                    self.rig_config['H_projector_distortion_corrected_distortion_mapy'], dtype=np.float32
                )
        elif self.method == 'classic':
            # No additional setup needed for classic method
            pass
        else:
            raise ValueError("Invalid method. Choose from 'classic', 'kornia', or 'map'.")

    def draw_geometry(self, drawing_function, debug=False):
        """
        Draws geometry in camera space using the provided drawing function.

        Parameters:
        - drawing_function: A function that takes a Cairo context as input and draws on it.
        """
        if debug:
            time_cairo_drawing = 0
            time_data_conversion = 0
            time_transform = 0
            time_display_update = 0
            start = time.time()

        # Clear the camera surface
        self.camera_context.save()
        self.camera_context.set_operator(cairo.OPERATOR_CLEAR)
        self.camera_context.paint()
        self.camera_context.restore()

        # Call the drawing function
        drawing_function(self.camera_context)
        if debug:
            time_cairo_drawing += time.time() - start

        # Get the image data from the camera surface
        if debug:
            start_conversion = time.time()
        camera_image = np.frombuffer(self.camera_surface.get_data(), np.uint8)
        camera_image.shape = (self.camera_height, self.camera_surface.get_stride() // 4, 4)
        camera_image_rgb = camera_image[:, :self.camera_width, 2::-1]
        if debug:
            time_data_conversion += time.time() - start_conversion

        # Apply the camera to projector transformation
        if debug:
            start_transform = time.time()
        if self.method == 'kornia':
            projector_image = self.transform_camera_to_projector_kornia(camera_image_rgb)
        elif self.method == 'map':
            projector_image = self.transform_camera_to_projector_map(camera_image_rgb)
        else:  # 'classic'
            projector_image = self.transform_camera_to_projector_classic(camera_image_rgb)
        if debug:
            time_transform += time.time() - start_transform

        # Update the pygame display
        if debug:
            start_display = time.time()
        self.update_display(projector_image)
        if debug:
            time_display_update += time.time() - start_display

        # Store the last image
        self.last_image = projector_image

        if debug:
            print(f"Time Cairo drawing: {time_cairo_drawing:.6f} s")
            print(f"Time data conversion: {time_data_conversion:.6f} s")
            print(f"Time transform: {time_transform:.6f} s")
            print(f"Time display update: {time_display_update:.6f} s")

    def update_display(self, image=None):
        """
        Updates the pygame display with the provided image.

        Parameters:
        - image: The image to display. If None, uses the last image.
        """
        if image is None and self.last_image is not None:
            image = self.last_image
        elif image is None:
            # Nothing to display
            return

        # Convert image to pygame surface
        pygame_surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))

        # Blit the surface onto the screen
        self.screen.blit(pygame_surface, (0, 0))

        # Update the display
        pygame.display.flip()

        # Handle events to keep the window responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Limit the frame rate
        self.clock.tick(self.fps)

    def transform_camera_to_projector_classic(self, camera_image):
        """
        Transforms an image from camera space to projector space using OpenCV (classic method).

        Parameters:
        - camera_image: The image in camera space.

        Returns:
        - projector_image: The transformed image in projector space.
        """
        # Apply the refined homography to get the image in projection space
        projection_space_image = cv2.warpPerspective(
            camera_image,
            self.H_refined,
            (self.projector_width, self.projector_height)
        )

        if self.projector_correction_method == 'homography':
            # Apply the projector distortion correction homography
            corrected_projector_image = cv2.warpPerspective(
                projection_space_image,
                self.H_projector_distortion_corrected,
                (self.projector_width, self.projector_height)
            )
        elif self.projector_correction_method == 'distortion':
            # Apply the distortion correction mapping
            corrected_projector_image = remap_image_with_interpolation(
                projection_space_image,
                self.distortion_corrected_projection_space_detail_markers,
                self.interpolated_projector_space_detail_markers,
                (self.projector_height, self.projector_width)
            )
        else:
            # No correction
            corrected_projector_image = projection_space_image

        return corrected_projector_image

    def transform_camera_to_projector_kornia(self, camera_image):
        """
        Transforms an image from camera space to projector space using Kornia (HWAccel).

        Parameters:
        - camera_image: The image in camera space.

        Returns:
        - projector_image: The transformed image in projector space.
        """
        # Convert camera_image to tensor
        camera_image_tensor = numpy_to_tensor(camera_image)

        # Apply the refined homography to get the image in projection space
        projection_space_image_tensor = kornia.geometry.transform.warp_perspective(
            camera_image_tensor,
            self.H_refined_tensor.unsqueeze(0),
            dsize=(self.projector_height, self.projector_width),
            align_corners=True
        )

        if self.projector_correction_method == 'homography':
            # Apply the projector distortion correction homography
            corrected_projector_image_tensor = kornia.geometry.transform.warp_perspective(
                projection_space_image_tensor,
                self.H_projector_distortion_corrected_tensor.unsqueeze(0),
                dsize=(self.projector_height, self.projector_width)
            )
            corrected_projector_image = tensor_to_numpy(corrected_projector_image_tensor)
        elif self.projector_correction_method == 'distortion':
            # Convert to numpy array
            projection_space_image = tensor_to_numpy(projection_space_image_tensor)
            # Apply the distortion correction mapping
            corrected_projector_image = remap_image_with_interpolation(
                projection_space_image,
                self.distortion_corrected_projection_space_detail_markers,
                self.interpolated_projector_space_detail_markers,
                (self.projector_height, self.projector_width)
            )
        else:
            # No correction
            corrected_projector_image = tensor_to_numpy(projection_space_image_tensor)

        return corrected_projector_image

    def transform_camera_to_projector_map(self, camera_image):
        """
        Transforms an image from camera space to projector space using map remap maps.

        Parameters:
        - camera_image: The image in camera space.

        Returns:
        - projector_image: The transformed image in projector space.
        """
        # Apply the map remap for H_refined
        projection_space_image = cv2.remap(
            camera_image,
            self.H_refined_mapx,
            self.H_refined_mapy,
            interpolation=cv2.INTER_LINEAR
        )

        # Apply projector correction method
        corrected_projector_image = cv2.remap(
            projection_space_image,
            self.H_projector_distortion_corrected_mapx,
            self.H_projector_distortion_corrected_mapy,
            interpolation=cv2.INTER_LINEAR
        )

        return corrected_projector_image

    def keep_last_image(self):
        """
        Keeps the last image displayed without any updates.
        """
        self.update_display()

    def close(self):
        """
        Closes the pygame window and releases resources.
        """
        pygame.quit()



class Drawing:
    def __init__(self):
        self.instructions = []

    def add_circle(self, center, radius, color=(0, 0, 0), line_width=1, fill=False, rotation=0):
        """
        Adds a circle to the drawing.

        Parameters:
        - center: Tuple (x, y) specifying the center of the circle.
        - radius: Radius of the circle.
        - color: Tuple (r, g, b) with values between 0 and 1.
        - line_width: Width of the circle's outline.
        - fill: Boolean indicating whether to fill the circle.
        - rotation: Rotation angle in radians around the center.
        """
        def draw(context):
            context.save()
            context.translate(center[0], center[1])  # Move to center
            context.rotate(rotation)
            context.set_line_width(line_width)
            context.set_source_rgb(*color)
            context.arc(0, 0, radius, 0, 2 * np.pi)
            if fill:
                context.fill_preserve()
            context.stroke()
            context.restore()
        self.instructions.append(draw)

    def add_ellipse(self, center, radius_x, radius_y, color=(0, 0, 0), line_width=1, fill=False, rotation=0):
        """
        Adds an ellipse to the drawing.

        Parameters:
        - center: Tuple (x, y) specifying the center of the ellipse.
        - radius_x: Horizontal radius.
        - radius_y: Vertical radius.
        - color: Tuple (r, g, b) with values between 0 and 1.
        - line_width: Width of the ellipse's outline.
        - fill: Boolean indicating whether to fill the ellipse.
        - rotation: Rotation angle in radians around the center.
        """
        def draw(context):
            context.save()
            context.translate(center[0], center[1])
            context.rotate(rotation)
            context.scale(radius_x / max(radius_x, radius_y), radius_y / max(radius_x, radius_y))
            context.set_line_width(line_width / max(radius_x, radius_y))
            context.set_source_rgb(*color)
            context.arc(0, 0, max(radius_x, radius_y), 0, 2 * np.pi)
            if fill:
                context.fill_preserve()
            context.stroke()
            context.restore()
        self.instructions.append(draw)

    def add_line(self, start_point, end_point, color=(0, 0, 0), line_width=1, rotation=0):
        """
        Adds a line to the drawing.

        Parameters:
        - start_point: Tuple (x, y) specifying the start point.
        - end_point: Tuple (x, y) specifying the end point.
        - color: Tuple (r, g, b) with values between 0 and 1.
        - line_width: Width of the line.
        - rotation: Rotation angle in radians around the midpoint of the line.
        """
        def draw(context):
            context.save()
            # Calculate midpoint for rotation
            mid_x = (start_point[0] + end_point[0]) / 2
            mid_y = (start_point[1] + end_point[1]) / 2
            context.translate(mid_x, mid_y)
            context.rotate(rotation)
            context.translate(-mid_x, -mid_y)
            context.set_line_width(line_width)
            context.set_source_rgb(*color)
            context.move_to(*start_point)
            context.line_to(*end_point)
            context.stroke()
            context.restore()
        self.instructions.append(draw)

    def add_rectangle(self, top_left, width, height, color=(0, 0, 0), line_width=1, fill=False, rotation=0):
        """
        Adds a rectangle to the drawing.

        Parameters:
        - top_left: Tuple (x, y) specifying the top-left corner.
        - width: Width of the rectangle.
        - height: Height of the rectangle.
        - color: Tuple (r, g, b) with values between 0 and 1.
        - line_width: Width of the rectangle's outline.
        - fill: Boolean indicating whether to fill the rectangle.
        - rotation: Rotation angle in radians around the rectangle's center.
        """
        def draw(context):
            context.save()
            # Calculate center of the rectangle
            center_x = top_left[0] + width / 2
            center_y = top_left[1] + height / 2
            context.translate(center_x, center_y)
            context.rotate(rotation)
            context.translate(-width / 2, -height / 2)
            context.set_line_width(line_width)
            context.set_source_rgb(*color)
            context.rectangle(0, 0, width, height)
            if fill:
                context.fill_preserve()
            context.stroke()
            context.restore()
        self.instructions.append(draw)

    def add_text(self, position, text, font_size=12, color=(0, 0, 0), rotation=0, font_family="Sans", font_weight=cairo.FONT_WEIGHT_NORMAL):
        """
        Adds text to the drawing.

        Parameters:
        - position: Tuple (x, y) specifying the position.
        - text: The text string to display.
        - font_size: Size of the font.
        - color: Tuple (r, g, b) with values between 0 and 1.
        - rotation: Rotation angle in radians around the text position.
        - font_family: Name of the font family.
        - font_weight: Weight of the font (e.g., cairo.FONT_WEIGHT_BOLD).
        """
        def draw(context):
            context.save()
            context.translate(position[0], position[1])
            context.rotate(rotation)
            context.set_source_rgb(*color)
            context.select_font_face(font_family, cairo.FONT_SLANT_NORMAL, font_weight)
            context.set_font_size(font_size)
            context.move_to(0, 0)
            context.show_text(text)
            context.restore()
        self.instructions.append(draw)

    def add_polygon(self, points, center=(0, 0), color=(0, 0, 0), line_width=1, fill=False, rotation=0):
        """
        Adds a polygon to the drawing.

        Parameters:
        - points: List of tuples [(x0, y0), (x1, y1), ..., (xn, yn)] specifying vertices relative to the center.
        - center: Tuple (x, y) specifying the center of the polygon.
        - color: Tuple (r, g, b) with values between 0 and 1.
        - line_width: Width of the polygon's outline.
        - fill: Boolean indicating whether to fill the polygon.
        - rotation: Rotation angle in radians around the center.
        """
        def draw(context):
            if len(points) < 3:
                return  # Need at least 3 points for a polygon
            context.save()
            context.translate(center[0], center[1])  # Translate to center
            context.rotate(rotation)
            context.set_line_width(line_width)
            context.set_source_rgb(*color)
            context.move_to(*points[0])
            for point in points[1:]:
                context.line_to(*point)
            context.close_path()
            if fill:
                context.fill_preserve()
            context.stroke()
            context.restore()
        self.instructions.append(draw)

    def add_arc(self, center, radius, start_angle, end_angle, color=(0, 0, 0), line_width=1, fill=False, rotation=0):
        """
        Adds an arc to the drawing.

        Parameters:
        - center: Tuple (x, y) specifying the center of the arc.
        - radius: Radius of the arc.
        - start_angle: Starting angle in radians.
        - end_angle: Ending angle in radians.
        - color: Tuple (r, g, b) with values between 0 and 1.
        - line_width: Width of the arc's outline.
        - fill: Boolean indicating whether to fill the arc sector.
        - rotation: Rotation angle in radians around the center.
        """
        def draw(context):
            context.save()
            context.translate(center[0], center[1])  # Move to center
            context.rotate(rotation)
            context.set_line_width(line_width)
            context.set_source_rgb(*color)
            context.move_to(0, 0)
            context.arc(0, 0, radius, start_angle, end_angle)
            if fill:
                context.line_to(0, 0)
                context.close_path()
                context.fill_preserve()
            context.stroke()
            context.restore()
        self.instructions.append(draw)

    def add_image(self, image_path, position, rotation=0, scale=(1, 1)):
        """
        Adds an image to the drawing.

        Parameters:
        - image_path: Path to the image file.
        - position: Tuple (x, y) specifying the top-left corner.
        - rotation: Rotation angle in radians around the image center.
        - scale: Tuple (sx, sy) specifying scaling factors.
        """
        def draw(context):
            context.save()
            # Load the image
            image_surface = cairo.ImageSurface.create_from_png(image_path)
            width = image_surface.get_width()
            height = image_surface.get_height()
            # Calculate center
            center_x = position[0] + width / 2
            center_y = position[1] + height / 2
            context.translate(center_x, center_y)
            context.rotate(rotation)
            context.scale(*scale)
            context.translate(-width / 2, -height / 2)
            context.set_source_surface(image_surface, 0, 0)
            context.paint()
            context.restore()
        self.instructions.append(draw)

    def add_path(self, points, center=(0, 0), color=(0, 0, 0), line_width=1, close=False, fill=False, rotation=0):
        """
        Adds a path (polyline) to the drawing.

        Parameters:
        - points: List of tuples [(x0, y0), (x1, y1), ..., (xn, yn)] specifying the path relative to the center.
        - center: Tuple (x, y) specifying the center of the path.
        - color: Tuple (r, g, b) with values between 0 and 1.
        - line_width: Width of the path's line.
        - close: Boolean indicating whether to close the path.
        - fill: Boolean indicating whether to fill the path if closed.
        - rotation: Rotation angle in radians around the center.
        """
        def draw(context):
            if len(points) < 2:
                return  # Need at least 2 points for a path
            context.save()
            context.translate(center[0], center[1])  # Translate to center
            context.rotate(rotation)
            context.set_line_width(line_width)
            context.set_source_rgb(*color)
            context.move_to(*points[0])
            for point in points[1:]:
                context.line_to(*point)
            if close:
                context.close_path()
                if fill:
                    context.fill_preserve()
            context.stroke()
            context.restore()
        self.instructions.append(draw)
    
    def add_bezier_curve(self, points, center=(0, 0), color=(0, 0, 0), line_width=1, fill=False, rotation=0):
        """
        Adds a Bezier curve to the drawing.

        Parameters:
        - points: List of tuples [(x0, y0), (x1, y1), ..., (xn, yn)] specifying control points relative to the center.
        - center: Tuple (x, y) specifying the center of the curve.
        - color: Tuple (r, g, b) with values between 0 and 1.
        - line_width: Width of the curve's line.
        - fill: Boolean indicating whether to fill the curve.
        - rotation: Rotation angle in radians around the center.
        """
        def draw(context):
            if len(points) < 4:
                return # Need at least 4 points for a Bezier curve
            context.save()
            context.translate(center[0], center[1])  # Translate to center
            context.rotate(rotation)
            context.set_line_width(line_width)
            context.set_source_rgb(*color)
            context.move_to(*points[0])
            context.curve_to(*points[1], *points[2], *points[3])
            if fill:
                context.fill_preserve()
            context.stroke()
            context.restore()
        self.instructions.append(draw)


    def add_text(self, position, text, font_size=12, color=(0, 0, 0), rotation=0, font_family="Sans", font_weight=cairo.FONT_WEIGHT_NORMAL):
        """
        Adds text to the drawing.

        Parameters:
        - position: Tuple (x, y) specifying the position.
        - text: The text string to display.
        - font_size: Size of the font.
        - color: Tuple (r, g, b) with values between 0 and 1.
        - rotation: Rotation angle in radians around the text position.
        - font_family: Name of the font family.
        - font_weight: Weight of the font (e.g., cairo.FONT_WEIGHT_BOLD).
        """
        def draw(context):
            context.save()
            context.translate(position[0], position[1])
            context.rotate(rotation)
            context.set_source_rgb(*color)
            context.select_font_face(font_family, cairo.FONT_SLANT_NORMAL, font_weight)
            context.set_font_size(font_size)
            context.move_to(0, 0)
            context.show_text(text)
            context.restore()
        self.instructions.append(draw)

    def add_surface(self, surface, position=(0, 0), rotation=0):
        """
        Adds a pre-rendered surface to the drawing.

        Parameters:
        - surface: A Cairo ImageSurface to draw.
        - position: Tuple (x, y) specifying the top-left corner.
        - rotation: Rotation angle in radians around the surface's center.
        """
        def draw(context):
            context.save()
            # Get surface dimensions
            width = surface.get_width()
            height = surface.get_height()
            # Calculate center
            center_x = position[0] + width / 2
            center_y = position[1] + height / 2
            context.translate(center_x, center_y)
            context.rotate(rotation)
            context.translate(-width / 2, -height / 2)
            context.set_source_surface(surface, 0, 0)
            context.paint()
            context.restore()
        self.instructions.append(draw)

    def create_sprite(self, width, height, drawing_instructions):
        """
        Creates a pre-rendered sprite from given drawing instructions.

        Parameters:
        - width: Width of the sprite surface.
        - height: Height of the sprite surface.
        - drawing_instructions: A function that takes a Cairo context and draws on it.

        Returns:
        - A Cairo ImageSurface containing the pre-rendered sprite.
        """
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        context = cairo.Context(surface)
        drawing_instructions(context)
        return surface

    def add_sprite(self, sprite_surface, center=(0, 0), rotation=0, scale=(1, 1)):
        """
        Adds a pre-rendered sprite to the drawing with optional transformations.

        Parameters:
        - sprite_surface: The Cairo ImageSurface of the sprite.
        - center: Tuple (x, y) specifying the position to place the sprite.
        - rotation: Rotation angle in radians around the sprite's center.
        - scale: Tuple (sx, sy) specifying scaling factors.
        """
        def draw(context):
            context.save()
            width = sprite_surface.get_width()
            height = sprite_surface.get_height()
            # Move to the sprite's center
            context.translate(center[0], center[1])
            # Apply transformations
            context.rotate(rotation)
            context.scale(scale[0], scale[1])
            # Draw the sprite centered at (0, 0)
            context.set_source_surface(sprite_surface, -width / 2, -height / 2)
            context.paint()
            context.restore()
        self.instructions.append(draw)

    def get_drawing_function(self):
        """
        Compiles all drawing instructions into a single function.

        Returns:
        - A function that takes a Cairo context and executes all drawing instructions.
        """
        def draw_all(context):
            for instruction in self.instructions:
                instruction(context)
        return draw_all

