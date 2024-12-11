import os
import sys
import json
import numpy as np
import pygame
import cairo
import cv2
from flyprojection.utils.transforms import remap_image_with_interpolation, remap_coords_with_interpolation, remap_image_with_map, remap_coords_with_map
from flyprojection.utils.utils import numpy_to_tensor, tensor_to_numpy
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
        Initialize the Artist class responsible for rendering geometry defined in camera space 
        onto a projector window after applying a series of transformations and corrections.
        
        The class supports multiple transformation methods ('classic', 'kornia', 'map') for 
        converting from camera coordinates to projector coordinates and applies corrections 
        such as distortion or homography transformations.

        Parameters
        ----------
        camera : object
            The camera object with attributes WIDTH and HEIGHT representing the camera resolution.
        rig_config : str or dict
            The path to the rig_config.json file or a dictionary representing the rig configuration.
            The rig_config should contain:
             - 'projector_width': int
             - 'projector_height': int
             - 'H_refined': list (2D)
             - 'H_projector_distortion_corrected': list (2D)
             - 'projector_correction_method': str ('distortion', 'homography', 'none')
             - 'camera_correction_method': str ('none')
             - If using 'distortion' correction method:
                 'interpolated_projector_space_detail_markers': list (2D)
                 'distortion_corrected_projection_space_detail_markers': list (2D)
             - If using 'map':
                 'H_refined_mapx': list (2D)
                 'H_refined_mapy': list (2D)
                 'H_refined_inv_mapx': list (2D)
                 'H_refined_inv_mapy': list (2D)
                 And similar sets for 'H_projector_distortion_corrected_*_mapx',
                 'H_projector_distortion_corrected_*_mapy',
                 'H_projector_distortion_corrected_*_inv_mapx', 
                 'H_projector_distortion_corrected_*_inv_mapy' depending on the correction method.
        method : str, optional
            The method to use for transformations. One of ('classic', 'kornia', 'map').
            - 'classic': Uses OpenCV warpPerspective for transformations.
            - 'kornia': Uses Kornia + PyTorch for GPU-accelerated transformations.
            - 'map': Uses precomputed remap maps (forward and inverse).
        fps : int, optional
            The frame rate for the display window updates.

        Raises
        ------
        ValueError
            If an invalid rig_config is provided or if the method is not recognized.
        ImportError
            If 'kornia' method is chosen but PyTorch and Kornia are not available.
        """
        self.camera = camera
        self.method = method.lower()
        self.fps = fps

        # Load rig configuration
        if isinstance(rig_config, dict):
            self.rig_config = rig_config
        elif isinstance(rig_config, str):
            with open(rig_config, 'r') as f:
                self.rig_config = json.load(f)
        else:
            raise ValueError("Invalid rig_config. Provide a path to rig_config.json or a dictionary.")

        # Extract necessary parameters from rig_config
        self.projector_width = self.rig_config['projector_width']
        self.projector_height = self.rig_config['projector_height']
        self.camera_width = self.camera.WIDTH
        self.camera_height = self.camera.HEIGHT

        # Load the camera-to-projector transformation matrices
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

        # Initialize Pygame and set up the display window
        os.environ['SDL_VIDEO_WINDOW_POS'] = "0,0"
        pygame.init()
        self.screen = pygame.display.set_mode(
            (self.projector_width, self.projector_height),
            pygame.NOFRAME | pygame.HWSURFACE | pygame.DOUBLEBUF
        )

        # Store the last displayed image
        self.last_image = None

        # Prepare a Cairo surface and context for drawing in camera space
        self.camera_surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.camera_width, self.camera_height)
        self.camera_context = cairo.Context(self.camera_surface)

        # Prepare a clock to control the display's frame rate
        self.clock = pygame.time.Clock()

        # Set up method-specific configurations
        if self.method == 'kornia':
            if torch is None or kornia is None:
                raise ImportError("PyTorch and Kornia are required for the 'kornia' method.")
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # Precompute tensors for transformation
            self.H_refined_tensor = torch.tensor(self.H_refined, dtype=torch.float32, device=self.device)
            self.H_projector_distortion_corrected_tensor = torch.tensor(
                self.H_projector_distortion_corrected, dtype=torch.float32, device=self.device
            )
            if self.projector_correction_method == 'distortion':
                self.X = torch.tensor(
                    self.distortion_corrected_projection_space_detail_markers,
                    dtype=torch.float32, device=self.device
                )
                self.Y = torch.tensor(
                    self.interpolated_projector_space_detail_markers,
                    dtype=torch.float32, device=self.device
                )
        elif self.method == 'map':
            # Load the precomputed forward and inverse remap maps for camera-to-projector transformation
            self.H_refined_mapx = np.array(self.rig_config['H_refined_mapx'], dtype=np.float32)
            self.H_refined_mapy = np.array(self.rig_config['H_refined_mapy'], dtype=np.float32)
            self.H_refined_inv_mapx = np.array(self.rig_config['H_refined_inv_mapx'], dtype=np.float32)
            self.H_refined_inv_mapy = np.array(self.rig_config['H_refined_inv_mapy'], dtype=np.float32)

            # Load projector correction maps depending on the chosen correction method
            if self.projector_correction_method == 'homography':
                self.H_projector_distortion_corrected_mapx = np.array(
                    self.rig_config['H_projector_distortion_corrected_homography_mapx'], dtype=np.float32
                )
                self.H_projector_distortion_corrected_mapy = np.array(
                    self.rig_config['H_projector_distortion_corrected_homography_mapy'], dtype=np.float32
                )
                self.H_projector_distortion_corrected_inv_mapx = np.array(
                    self.rig_config['H_projector_distortion_corrected_homography_inv_mapx'], dtype=np.float32
                )
                self.H_projector_distortion_corrected_inv_mapy = np.array(
                    self.rig_config['H_projector_distortion_corrected_homography_inv_mapy'], dtype=np.float32
                )
            elif self.projector_correction_method == 'distortion':
                self.H_projector_distortion_corrected_mapx = np.array(
                    self.rig_config['H_projector_distortion_corrected_distortion_mapx'], dtype=np.float32
                )
                self.H_projector_distortion_corrected_mapy = np.array(
                    self.rig_config['H_projector_distortion_corrected_distortion_mapy'], dtype=np.float32
                )
                self.H_projector_distortion_corrected_inv_mapx = np.array(
                    self.rig_config['H_projector_distortion_corrected_distortion_inv_mapx'], dtype=np.float32
                )
                self.H_projector_distortion_corrected_inv_mapy = np.array(
                    self.rig_config['H_projector_distortion_corrected_distortion_inv_mapy'], dtype=np.float32
                )
        elif self.method == 'classic':
            # No additional setup needed for the classic method
            pass
        else:
            raise ValueError("Invalid method. Choose from 'classic', 'kornia', or 'map'.")

    def draw_geometry(self, drawing_function, debug=False, return_camera_image=False, return_projector_image=False):
        """
        Draw geometry onto the camera surface, transform it into the projector space, 
        and display it on the projector window.
        
        This method:
        1. Clears the camera surface.
        2. Calls the provided `drawing_function` to draw onto the camera_context (in camera space).
        3. Extracts the drawn image as a numpy array (camera_image_rgb).
        4. Transforms the camera image into projector space using the specified method.
        5. Updates the projector display window with the transformed image.

        Parameters
        ----------
        drawing_function : callable
            A function that accepts a Cairo context and draws geometry onto it.
        debug : bool, optional
            If True, prints out timing information for each step.
        return_camera_image : bool, optional
            If True, returns the camera space image as a numpy array.
        return_projector_image : bool, optional
            If True, returns the projector space image as a numpy array.

        Returns
        -------
        camera_image_rgb : numpy.ndarray, optional
            The camera space image if `return_camera_image` is True.
        projector_image : numpy.ndarray, optional
            The projector space image if `return_projector_image` is True.
            Returns both if both are True.
        None
            If neither return_camera_image nor return_projector_image is True.
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

        # Draw using the provided drawing_function
        drawing_function(self.camera_context)
        if debug:
            time_cairo_drawing += time.time() - start

        # Convert the drawn Cairo surface to a numpy array (camera space image)
        if debug:
            start_conversion = time.time()
        camera_image = np.frombuffer(self.camera_surface.get_data(), np.uint8)
        camera_image.shape = (self.camera_height, self.camera_surface.get_stride() // 4, 4)
        # Extract RGB channels and flip BGR to RGB
        camera_image_rgb = camera_image[:, :self.camera_width, 2::-1]
        if debug:
            time_data_conversion += time.time() - start_conversion

        # Transform the camera image into projector space
        if debug:
            start_transform = time.time()
        if self.method == 'kornia':
            projector_image = self.transform_camera_to_projector_kornia(camera_image_rgb)
        elif self.method == 'map':
            projector_image = self.transform_camera_to_projector_map(camera_image_rgb)
        else:  # classic method
            projector_image = self.transform_camera_to_projector_classic(camera_image_rgb)
        if debug:
            time_transform += time.time() - start_transform

        # Update the pygame display with the transformed (projector space) image
        if debug:
            start_display = time.time()
        self.update_display(projector_image)
        if debug:
            time_display_update += time.time() - start_display

        # Store the last image for future reference
        self.last_image = projector_image

        # Print debugging information if requested
        if debug:
            print(f"Time Cairo drawing: {time_cairo_drawing:.6f} s")
            print(f"Time data conversion: {time_data_conversion:.6f} s")
            print(f"Time transform: {time_transform:.6f} s")
            print(f"Time display update: {time_display_update:.6f} s")

        # Return requested images
        if return_camera_image and return_projector_image:
            return camera_image_rgb, projector_image
        elif return_camera_image:
            return camera_image_rgb
        elif return_projector_image:
            return projector_image
        else:
            return None

    def update_display(self, image=None):
        """
        Update the pygame display with the provided image or the last displayed image.

        Parameters
        ----------
        image : numpy.ndarray, optional
            The image to display on the projector window. If None, the last displayed image 
            will be reused. The array shape should match (projector_height, projector_width, 3).
        """
        if image is None and self.last_image is not None:
            image = self.last_image
        elif image is None:
            # No image to display
            return

        # Convert the numpy image to a Pygame surface
        pygame_surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))

        # Blit the surface onto the screen
        self.screen.blit(pygame_surface, (0, 0))

        # Update the display
        pygame.display.flip()

        # Process Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Limit the frame rate
        self.clock.tick(self.fps)

    def get_screen_value_at(self, x, y):
        """
        Retrieve the value (color) of the projector screen at a given pixel coordinate.

        Parameters
        ----------
        x : int or float
            The x-coordinate on the projector screen.
        y : int or float
            The y-coordinate on the projector screen.

        Returns
        -------
        tuple
            A tuple (R, G, B, A) representing the color at the specified screen coordinates.
        """
        return self.screen.get_at((int(x), int(y)))

    def get_values_at_camera_coords(self, coords):
        """
        Given a list of coordinates in camera space, transform them into projector space 
        and retrieve the screen values at those positions.

        Parameters
        ----------
        coords : list of tuples
            A list of (x, y) coordinates in camera space.

        Returns
        -------
        values : list of tuples
            A list of (R, G, B, A) values for each corresponding transformed coordinate in projector space.
        """
        # Transform camera coords to projector coords depending on the chosen method
        if self.method == 'kornia':
            projector_coords = self.transform_coords_to_projector_kornia(coords)
        elif self.method == 'map':
            projector_coords = self.transform_coords_to_projector_map(coords)
        else:  # classic method
            projector_coords = self.transform_coords_to_projector_classic(coords)

        # Retrieve the screen values at the transformed projector coordinates
        values = []
        for coord in projector_coords:
            values.append(self.get_screen_value_at(*coord))

        return values

    def transform_camera_to_projector_classic(self, camera_image):
        """
        Transform the camera image to projector space using OpenCV warpPerspective and 
        apply the specified projector correction method (homography or distortion).

        Parameters
        ----------
        camera_image : numpy.ndarray
            The camera space image as a numpy array (H x W x 3).

        Returns
        -------
        corrected_projector_image : numpy.ndarray
            The transformed projector space image after all corrections.
        """
        # Warp from camera to projection space
        projection_space_image = cv2.warpPerspective(
            camera_image,
            self.H_refined,
            (self.projector_width, self.projector_height)
        )

        # Apply projector corrections
        if self.projector_correction_method == 'homography':
            corrected_projector_image = cv2.warpPerspective(
                projection_space_image,
                self.H_projector_distortion_corrected,
                (self.projector_width, self.projector_height)
            )
        elif self.projector_correction_method == 'distortion':
            corrected_projector_image = remap_image_with_interpolation(
                projection_space_image,
                self.distortion_corrected_projection_space_detail_markers,
                self.interpolated_projector_space_detail_markers,
                (self.projector_height, self.projector_width)
            )
        else:
            corrected_projector_image = projection_space_image

        return corrected_projector_image

    def transform_coords_to_projector_classic(self, coords):
        """
        Transform a list of coordinates from camera space to projector space using OpenCV perspectiveTransform.
        Applies any required correction (homography or distortion) to the coordinates.

        Parameters
        ----------
        coords : list of tuples
            List of (x, y) coordinates in camera space.

        Returns
        -------
        corrected_projector_coords : numpy.ndarray
            The coordinates transformed into projector space after correction.
        """
        coords = np.array(coords, dtype=np.float32)

        # Transform coordinates to projection space
        projection_space_coords = cv2.perspectiveTransform(
            coords.reshape(1, -1, 2),
            self.H_refined
        ).reshape(-1, 2)

        # Apply projector corrections
        if self.projector_correction_method == 'homography':
            corrected_projector_coords = cv2.perspectiveTransform(
                projection_space_coords.reshape(1, -1, 2),
                self.H_projector_distortion_corrected
            ).reshape(-1, 2)
        elif self.projector_correction_method == 'distortion':
            corrected_projector_coords = remap_coords_with_interpolation(
                projection_space_coords,
                self.distortion_corrected_projection_space_detail_markers,
                self.interpolated_projector_space_detail_markers
            )
        else:
            corrected_projector_coords = projection_space_coords

        return corrected_projector_coords

    def transform_camera_to_projector_kornia(self, camera_image):
        """
        Transform the camera image to projector space using Kornia and PyTorch. 
        If distortion correction is used, a post-processing step is applied.

        Parameters
        ----------
        camera_image : numpy.ndarray
            The camera space image (H x W x 3).

        Returns
        -------
        corrected_projector_image : numpy.ndarray
            The transformed projector space image after Kornia transformations and corrections.
        """
        # Convert camera image to torch tensor
        camera_image_tensor = numpy_to_tensor(camera_image)

        # Warp from camera to projection space using Kornia
        projection_space_image_tensor = kornia.geometry.transform.warp_perspective(
            camera_image_tensor,
            self.H_refined_tensor.unsqueeze(0),
            dsize=(self.projector_height, self.projector_width),
            align_corners=True
        )

        # Apply projector corrections
        if self.projector_correction_method == 'homography':
            corrected_projector_image_tensor = kornia.geometry.transform.warp_perspective(
                projection_space_image_tensor,
                self.H_projector_distortion_corrected_tensor.unsqueeze(0),
                dsize=(self.projector_height, self.projector_width)
            )
            corrected_projector_image = tensor_to_numpy(corrected_projector_image_tensor)
        elif self.projector_correction_method == 'distortion':
            projection_space_image = tensor_to_numpy(projection_space_image_tensor)
            corrected_projector_image = remap_image_with_interpolation(
                projection_space_image,
                self.distortion_corrected_projection_space_detail_markers,
                self.interpolated_projector_space_detail_markers,
                (self.projector_height, self.projector_width)
            )
        else:
            corrected_projector_image = tensor_to_numpy(projection_space_image_tensor)

        return corrected_projector_image

    def transform_coords_to_projector_kornia(self, coords):
        """
        Transform coordinates from camera space to projector space using Kornia transformations.
        Applies projector corrections (homography or distortion) if required.

        Parameters
        ----------
        coords : list of tuples
            List of (x, y) coordinates in camera space.

        Returns
        -------
        corrected_projector_coords : numpy.ndarray
            The transformed coordinates in projector space after corrections.
        """
        coords_tensor = numpy_to_tensor(np.array(coords, dtype=np.float32))

        # Perspective transform of coordinates from camera to projection space
        projection_space_coords_tensor = kornia.geometry.transform.perspective_transform(
            coords_tensor.unsqueeze(0),
            self.H_refined_tensor.unsqueeze(0)
        ).squeeze(0)

        # Apply corrections
        if self.projector_correction_method == 'homography':
            corrected_projector_coords_tensor = kornia.geometry.transform.perspective_transform(
                projection_space_coords_tensor.unsqueeze(0),
                self.H_projector_distortion_corrected_tensor.unsqueeze(0)
            ).squeeze(0)
            corrected_projector_coords = tensor_to_numpy(corrected_projector_coords_tensor)
        elif self.projector_correction_method == 'distortion':
            projection_space_coords = tensor_to_numpy(projection_space_coords_tensor)
            corrected_projector_coords = remap_coords_with_interpolation(
                projection_space_coords,
                self.distortion_corrected_projection_space_detail_markers,
                self.interpolated_projector_space_detail_markers
            )
        else:
            corrected_projector_coords = tensor_to_numpy(projection_space_coords_tensor)

        return corrected_projector_coords

    def transform_camera_to_projector_map(self, camera_image):
        """
        Transform the camera image to projector space using precomputed remap maps.

        Parameters
        ----------
        camera_image : numpy.ndarray
            The camera space image (H x W x 3).

        Returns
        -------
        corrected_projector_image : numpy.ndarray
            The projector space image after applying the forward map and then the projector correction map.
        """
        # First apply H_refined remap
        projection_space_image = remap_image_with_map(
            camera_image,
            self.H_refined_mapx,
            self.H_refined_mapy,
        )

        # Apply projector correction remap
        corrected_projector_image = remap_image_with_map(
            projection_space_image,
            self.H_projector_distortion_corrected_mapx,
            self.H_projector_distortion_corrected_mapy,
        )

        return corrected_projector_image

    def transform_coords_to_projector_map(self, coords):
        """
        Transform a list of coordinates from camera space to projector space using precomputed remap maps.
        The inverse maps are used here to find the corresponding projector coordinates.

        Parameters
        ----------
        coords : list of tuples
            List of (x, y) coordinates in camera space.

        Returns
        -------
        corrected_projector_coords : numpy.ndarray
            The transformed coordinates in projector space after corrections.
        """
        coords = np.array(coords, dtype=np.float32)

        # Remap coordinates through inverse maps for the camera-to-projector transformation
        projection_space_coords = remap_coords_with_map(
            coords,
            self.H_refined_inv_mapx,
            self.H_refined_inv_mapy,
        )

        # Apply projector correction inverse maps
        corrected_projector_coords = remap_coords_with_map(
            projection_space_coords,
            self.H_projector_distortion_corrected_inv_mapx,
            self.H_projector_distortion_corrected_inv_mapy,
        )

        return corrected_projector_coords

    def keep_last_image(self):
        """
        Redisplay the last transformed image without performing any new drawing or transformations.
        This can be useful to hold the current frame on screen.
        """
        self.update_display()

    def close(self):
        """
        Close the pygame window and clean up resources.
        """
        pygame.quit()

    def __enter__(self):
        """
        Return the instance so it can be used as a context variable
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Close the resources (like the Pygame window) when exiting the context
        """
        self.close()
        print("Successfully closed pygame. Exiting Artist Context.")


class Drawing:
    def __init__(self):
        self.instructions = []
        self.mask = None

    def add_mask(self, mask_drawing):
        """
        Adds a mask to the drawing.

        Parameters:
        - mask_drawing: A Drawing object representing the mask.
        """
        self.mask = mask_drawing

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

    # def get_drawing_function(self):
    #     """
    #     Compiles all drawing instructions into a single function.

    #     Returns:
    #     - A function that takes a Cairo context and executes all drawing instructions.
    #     """
    #     def draw_all(context):
    #         for instruction in self.instructions:
    #             instruction(context)
    #     return draw_all
    
    def get_drawing_function(self):
        """
        Compiles all drawing instructions into a single function, applying mask if present.

        Returns:
        - A function that takes a Cairo context and executes all drawing instructions.
        """
        def draw_all(context):
            width = context.get_target().get_width()
            height = context.get_target().get_height()

            # Create a surface for the main drawing
            main_surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
            main_context = cairo.Context(main_surface)

            # Draw all instructions onto the main surface
            for instruction in self.instructions:
                instruction(main_context)

            if self.mask is not None:
                # Create a surface for the mask
                mask_surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
                mask_context = cairo.Context(mask_surface)

                # Render the mask drawing onto the mask surface
                mask_drawing_function = self.mask.get_drawing_function()
                mask_drawing_function(mask_context)

                # Apply the mask to the main surface
                context.save()
                context.set_source_surface(main_surface, 0, 0)
                context.mask_surface(mask_surface, 0, 0)
                context.restore()
            else:
                # No mask; paint the main surface directly
                context.save()
                context.set_source_surface(main_surface, 0, 0)
                context.paint()
                context.restore()
        return draw_all

    def clear(self):
        """
        Clears all drawing instructions.
        """
        self.instructions = []

