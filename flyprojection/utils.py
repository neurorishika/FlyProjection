import importlib.util
import os
from scipy.optimize import minimize, minimize_scalar
from scipy.optimize import shgo
import numpy as np
import cv2
import torch
import kornia

import logging
import logging.handlers
import queue
import threading
import signal
import sys

# Function to configure asynchronous logging
def setup_async_logger(log_file):
    # Create a thread-safe queue for logging messages
    log_queue = queue.Queue()

    # Create a handler that writes log messages to a file
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # Create a QueueHandler to send messages to the log queue
    queue_handler = logging.handlers.QueueHandler(log_queue)

    # Configure the root logger to use the queue handler
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Set the default log level
    logger.addHandler(queue_handler)

    # Create a thread that reads from the log queue and writes to the file
    log_thread = threading.Thread(target=queue_listener, args=(log_queue, file_handler), daemon=True)
    log_thread.start()

    # Store queue and thread for graceful shutdown
    logger.log_queue = log_queue
    logger.log_thread = log_thread

    return logger

# Function to listen for log messages in the queue
def queue_listener(log_queue, file_handler):
    while True:
        try:
            # Get log record from the queue
            record = log_queue.get()
            if record is None:
                break  # Exit the listener thread if a None record is received
            # Write the log record to the file
            file_handler.emit(record)
        except Exception as e:
            print(f"Error in log listener: {e}")

# Graceful shutdown function
def shutdown_logger(logger):
    logger.log_queue.put(None)  # Signal the listener thread to exit
    logger.log_thread.join()   # Wait for the thread to finish
    print("Logger shutdown gracefully.")

# Signal handler for graceful shutdown on interrupt
def signal_handler(sig, frame):
    print("Interrupt received. Shutting down logger...")
    shutdown_logger(logging.getLogger())
    sys.exit(0)


def hex_to_rgb(value):
    """Convert hex color to RGB."""
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

def relative_to_absolute(x, y, rig_config):
    """Convert relative coordinates to absolute coordinates."""
    region_x = rig_config['exp_region_x']
    region_y = rig_config['exp_region_y']
    width = rig_config['projector_width']
    height = rig_config['projector_height']
    x = int(width*(region_x[0] + x * (region_x[1] - region_x[0])))
    y = int(height*(region_y[0] + y * (region_y[1] - region_y[0])))
    return x, y

def get_boolean_answer(prompt, default=None, pygame = None):
    """Get a boolean answer from the user. Defaults if no answer is given."""
    while True:
        answer = input(prompt).lower()
        if answer == '':
            return default
        elif answer in ['y', 'yes']:
            return True
        elif answer in ['n', 'no']:
            return False
        else:
            print("Invalid answer. Please try again.")
            continue

def get_predefined_answer(prompt, options, default=None):
    """Get a predefined answer from the user. Defaults if no answer is given."""
    while True:
        answer = input(prompt).lower()
        if answer == '':
            return default
        elif answer in options:
            return answer
        else:
            print("Invalid answer. Please try again.")
            continue


def load_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"Could not load spec for '{module_name}' at '{file_path}'")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def tensor_to_numpy(tensor):
    """
    Convert a torch tensor to a NumPy array for display.

    Parameters:
        tensor (torch.Tensor): Tensor to convert. Shape [1, C, H, W].

    Returns:
        numpy.ndarray: Converted array. Shape [H, W, C], uint8.
    """
    tensor = tensor.detach()
    if tensor.is_cuda:
        tensor = tensor.cpu()
    tensor = tensor.squeeze(0).permute(1, 2, 0).numpy()  # [H, W, C]
    array = (tensor * 255).astype(np.uint8)
    return array

def numpy_to_tensor(array):
    """
    Convert a NumPy array to a torch tensor.

    Parameters:
        array (numpy.ndarray): Array to convert. Shape [H, W, C] or [H, W].

    Returns:
        torch.Tensor: Converted tensor. Shape [1, C, H, W], float32.
    """
    tensor = torch.from_numpy(array.copy()).float() / 255.0  # Normalize to [0,1]
    if len(tensor.shape) == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.shape[2] == 3:
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    # move to GPU if available
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor



from scipy.interpolate import griddata

def remap_image_with_interpolation(camera_image, X, Y, image_size, method='linear'):
    """
    Remap an image based on the interpolation from points X to points Y.

    Parameters:
    - camera_image: The input image to be remapped (numpy array).
    - X: Source points for the mapping (numpy array of shape (N, 2)).
    - Y: Target points corresponding to X (numpy array of shape (N, 2)).
    - image_size: Size of the output image (tuple of (height, width)).

    Returns:
    - remapped_image: The remapped image (numpy array).
    """
    
    # Create a mesh grid for the original image dimensions
    xx, yy = np.meshgrid(np.arange(image_size[1]), np.arange(image_size[0]))

    # Flatten the grid for interpolation
    points_grid = np.column_stack((xx.flatten(), yy.flatten()))

    # Perform grid data interpolation
    mapped_points = griddata(X, Y, points_grid, method='linear')

    # Split the mapped points back into x and y components
    mapx = mapped_points[:, 0].reshape(image_size)
    mapy = mapped_points[:, 1].reshape(image_size)

    # Ensure that mapped points are within the image boundaries
    mapx = np.clip(mapx, 0, camera_image.shape[1] - 1)
    mapy = np.clip(mapy, 0, camera_image.shape[0] - 1)

    # Remap the image
    remapped_image = cv2.remap(camera_image, mapx.astype(np.float32), mapy.astype(np.float32), interpolation=cv2.INTER_LINEAR)

    return remapped_image

def generate_grid_map(image_size, from_points, to_points, input_size, method='cubic'):
    """
    Generate a grid map for remapping an image based on the interpolation from points X to points Y.

    Parameters:
    - image_size: Size of the output image (tuple of (height, width)).
    - from_points: Source points for the mapping (numpy array of shape (N, 2)).
    - to_points: Target points corresponding to X (numpy array of shape (N, 2)).
    - input_size: Size of the input image (tuple of (height, width)).
    - method: Interpolation method to use (string, default 'cubic').

    Returns:
    - (mapx, mapy): The grid map for remapping the image.
    """
    
    # Create a mesh grid for the original image dimensions
    xx, yy = np.meshgrid(np.arange(image_size[1]), np.arange(image_size[0]))

    # Flatten the grid for interpolation
    points_grid = np.column_stack((xx.flatten(), yy.flatten()))

    # Perform grid data interpolation
    mapped_points = griddata(from_points, to_points, points_grid, method=method)

    # Split the mapped points back into x and y components
    mapx = mapped_points[:, 0].reshape(image_size)
    mapy = mapped_points[:, 1].reshape(image_size)

    # Ensure that mapped points are within the image boundaries
    mapx = np.clip(mapx, 0, input_size[1] - 1)
    mapy = np.clip(mapy, 0, input_size[0] - 1)

    return (mapx.astype(np.float32), mapy.astype(np.float32))

def remap_image_with_map(camera_image, interpolation_mapx, interpolation_mapy):
    """
    Remap an image based on a precomputed interpolation map.

    Parameters:
    - camera_image: The input image to be remapped (numpy array).
    - interpolation_mapx: The interpolation map for remapping the image in the x-direction (numpy array).
    - interpolation_mapy: The interpolation map for remapping the image in the y-direction (numpy array).
    - output_size: Size of the output image (tuple of (height, width)).

    Returns:
    - remapped_image: The remapped image (numpy array).
    """
    
    # Remap the image
    remapped_image = cv2.remap(camera_image, interpolation_mapx, interpolation_mapy, interpolation=cv2.INTER_LINEAR)

    return remapped_image

def fit_quadratic_curve(x, y):
    """
    Fit a quadratic curve y = a x^2 + b x + c to the data.
    Returns the coefficients [a, b, c].
    """
    coeffs = np.polyfit(x, y, deg=2)
    return coeffs  # [a, b, c]

def project_to_quadratic(points, coeffs):
    """
    Project points onto the quadratic curve defined by y = a x^2 + b x + c.
    points: array of shape (n_points, 2)
    coeffs: [a, b, c]
    Returns an array of projected points.
    """
    a, b, c = coeffs
    projected_points = []
    for xi, yi in points:
        # Define the distance squared function D^2(xp)
        def distance_squared(xp):
            yp = a * xp**2 + b * xp + c
            return (xp - xi)**2 + (yp - yi)**2
        res = shgo(distance_squared, bounds=[(0,3000)], n=100, iters=10)
        xp = res.x
        yp = a * xp**2 + b * xp + c
        projected_points.append((xp, yp))
    return np.array(projected_points)

def fit_linear_curve(x, y):
    """
    Fit a linear curve y = m x + c to the data.
    Returns the coefficients [m, c].
    """
    coeffs = np.polyfit(x, y, deg=1)
    return coeffs  # [m, c]

def project_to_linear(points, coeffs):
    """
    Project points onto the linear curve defined by y = m x + c.
    points: array of shape (n_points, 2)
    coeffs: [m, c]
    Returns an array of projected points.
    """
    m, c = coeffs
    projected_points = []
    for xi, yi in points:
        xp = (xi + m*yi - m*c) / (1 + m**2)
        yp = m * xp + c
        projected_points.append((xp, yp))
    return np.array(projected_points)

def subdivide_on_linear(points, coeffs, subdivisions=10):
    """
    Subdivide the points on the linear curve defined by y = m x + c.
    points: array of shape (n_points, 2)
    coeffs: [m, c]
    subdivisions: number of subdivisions between each pair of points
    Returns an array of subdivided points.
    """
    m, c = coeffs
    subdivided_points = []
    for i in range(len(points)-1):
        x1, y1 = points[i]
        x2, y2 = points[i+1]
        for j in range(subdivisions):
            t = j / subdivisions
            x = x1 + t*(x2 - x1)
            y = y1 + t*(y2 - y1)
            xp = (x + m*y - m*c) / (1 + m**2)
            yp = m * xp + c
            subdivided_points.append((xp, yp))
    return np.array(subdivided_points)


def ellipse_cart_to_pol(coeffs):
    """

    Convert the cartesian conic coefficients, (a, b, c, d, e, f), to the
    ellipse parameters, where F(x, y) = ax^2 + bxy + cy^2 + dx + ey + f = 0.
    The returned parameters are x0, y0, ap, bp, e, phi, where (x0, y0) is the
    ellipse centre; (ap, bp) are the semi-major and semi-minor axes,
    respectively; e is the eccentricity; and phi is the rotation of the semi-
    major axis from the x-axis.

    """

    # We use the formulas from https://mathworld.wolfram.com/Ellipse.html
    # which assumes a cartesian form ax^2 + 2bxy + cy^2 + 2dx + 2fy + g = 0.
    # Therefore, rename and scale b, d and f appropriately.
    a = coeffs[0]
    b = coeffs[1] / 2
    c = coeffs[2]
    d = coeffs[3] / 2
    f = coeffs[4] / 2
    g = coeffs[5]

    den = b**2 - a*c
    if den > 0:
        raise ValueError('coeffs do not represent an ellipse: b^2 - 4ac must'
                         ' be negative!')

    # The location of the ellipse centre.
    x0, y0 = (c*d - b*f) / den, (a*f - b*d) / den

    num = 2 * (a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g)
    fac = np.sqrt((a - c)**2 + 4*b**2)
    # The semi-major and semi-minor axis lengths (these are not sorted).
    ap = np.sqrt(num / den / (fac - a - c))
    bp = np.sqrt(num / den / (-fac - a - c))

    # Sort the semi-major and semi-minor axis lengths but keep track of
    # the original relative magnitudes of width and height.
    width_gt_height = True
    if ap < bp:
        width_gt_height = False
        ap, bp = bp, ap

    # The eccentricity.
    r = (bp/ap)**2
    if r > 1:
        r = 1/r
    e = np.sqrt(1 - r)

    # The angle of anticlockwise rotation of the major-axis from x-axis.
    if b == 0:
        phi = 0 if a < c else np.pi/2
    else:
        phi = np.arctan((2.*b) / (a - c)) / 2
        if a > c:
            phi += np.pi/2
    if not width_gt_height:
        # Ensure that phi is the angle to rotate to the semi-major axis.
        phi += np.pi/2
    phi = phi % np.pi

    return (x0, y0, ap, bp, e, phi)

def fit_ellipse(x, y):
    """

    Fit the coefficients a,b,c,d,e,f, representing an ellipse described by
    the formula F(x,y) = ax^2 + bxy + cy^2 + dx + ey + f = 0 to the provided
    arrays of data points x=[x1, x2, ..., xn] and y=[y1, y2, ..., yn].

    Based on the algorithm of Halir and Flusser, "Numerically stable direct
    least squares fitting of ellipses'.


    """

    D1 = np.vstack([x**2, x*y, y**2]).T
    D2 = np.vstack([x, y, np.ones(len(x))]).T
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2
    T = -np.linalg.inv(S3) @ S2.T
    M = S1 + S2 @ T
    C = np.array(((0, 0, 2), (0, -1, 0), (2, 0, 0)), dtype=float)
    M = np.linalg.inv(C) @ M
    eigval, eigvec = np.linalg.eig(M)
    con = 4 * eigvec[0]* eigvec[2] - eigvec[1]**2
    ak = eigvec[:, np.nonzero(con > 0)[0]]
    return ellipse_cart_to_pol(np.concatenate((ak, T @ ak)).ravel())


def project_to_ellipse(points, ellipse_params):
    # loop through all points and project them to the ellipse
    x0, y0, ap, bp, e, phi = ellipse_params
    for i, point in enumerate(points):
        # find the nearest point on the ellipse
        x, y = point
        def distance(t):
            x_position = x0 + ap * np.cos(t) * np.cos(phi) - bp * np.sin(t) * np.sin(phi)
            y_position = y0 + ap * np.cos(t) * np.sin(phi) + bp * np.sin(t) * np.cos(phi)
            return (x_position - x)**2 + (y_position - y)**2
        result = minimize(distance, 0)
        t = result.x[0]
        x_position = x0 + ap * np.cos(t) * np.cos(phi) - bp * np.sin(t) * np.sin(phi)
        y_position = y0 + ap * np.cos(t) * np.sin(phi) + bp * np.sin(t) * np.cos(phi)
        points[i] = (x_position, y_position)
    return points

def interpolate_on_ellipse(start_point, ellipse_params, num_points=100):
    x0, y0, ap, bp, e, phi = ellipse_params
    x, y = start_point
    def distance(t):
        x_position = x0 + ap * np.cos(t) * np.cos(phi) - bp * np.sin(t) * np.sin(phi)
        y_position = y0 + ap * np.cos(t) * np.sin(phi) + bp * np.sin(t) * np.cos(phi)
        return (x_position - x)**2 + (y_position - y)**2
    result = minimize(distance, 0)
    t_start = result.x[0]
    t_end = t_start + 2*np.pi
    ts = np.linspace(t_start, t_end, num_points)
    new_points = []
    for t in ts:
        x_position = x0 + ap * np.cos(t) * np.cos(phi) - bp * np.sin(t) * np.sin(phi)
        y_position = y0 + ap * np.cos(t) * np.sin(phi) + bp * np.sin(t) * np.cos(phi)
        new_points.append((x_position, y_position))
    return new_points


def subdivide_on_ellipse(points, ellipse_params, subdivisions=10):
    x0, y0, ap, bp, e, phi = ellipse_params
    ts = []
    for point in points:
        x, y = point
        def distance(t):
            x_position = x0 + ap * np.cos(t) * np.cos(phi) - bp * np.sin(t) * np.sin(phi)
            y_position = y0 + ap * np.cos(t) * np.sin(phi) + bp * np.sin(t) * np.cos(phi)
            return (x_position - x)**2 + (y_position - y)**2
        # find the nearest point on the ellipse using shgo
        result = shgo(distance, bounds=[(0, 2*np.pi)], n=100, iters=10)
        ts.append(result.x[0])
    ts = np.array(ts)
    # add 2*pi to the end to make sure the last point is included
    ts = np.concatenate([ts, [ts[0] + 2*np.pi]])
    ts = np.unwrap(ts)
    new_points = []
    for i in range(len(ts)-1):
        # x_position = x0 + ap * np.cos(ts[i]) * np.cos(phi) - bp * np.sin(ts[i]) * np.sin(phi)
        # y_position = y0 + ap * np.cos(ts[i]) * np.sin(phi) + bp * np.sin(ts[i]) * np.cos(phi)
        # new_points.append((x_position, y_position))
        t_start = ts[i]
        t_end = ts[i+1]
        ts_sub = np.linspace(t_start, t_end, subdivisions+2)[:-1]
        for t in ts_sub:
            x_position = x0 + ap * np.cos(t) * np.cos(phi) - bp * np.sin(t) * np.sin(phi)
            y_position = y0 + ap * np.cos(t) * np.sin(phi) + bp * np.sin(t) * np.cos(phi)
            new_points.append((x_position, y_position))
    return new_points
        

def get_rectangle_points(x, y, width, height, rotation):
    """Get the 4 corner points of a rectangle."""
    x = np.array([x - width/2, x + width/2, x + width/2, x - width/2])
    y = np.array([y - height/2, y - height/2, y + height/2, y + height/2])
    x_rot = x*np.cos(rotation) - y*np.sin(rotation)
    y_rot = x*np.sin(rotation) + y*np.cos(rotation)
    return list(zip(x_rot, y_rot))

# a function to fit a rectangle to a 4 corner points
def rect_fit(points,force_rotation=None):
    """
    Fit a rectangle to 4 corner points.
    
    Variables:
        points : 4 corner points of the rectangle (list of tuples)
        scale : scale factor for the initial guess (float)
    
    Returns:
        rectangle : 4 corner points of the rectangle (list of tuples)
    """
    # initial gueses
    x_pos = np.mean([point[0] for point in points])
    y_pos = np.mean([point[1] for point in points])
    width = np.max([point[0] for point in points]) - np.min([point[0] for point in points])
    height = np.max([point[1] for point in points]) - np.min([point[1] for point in points])
    rotation = 0

    def loss(params):
        """ Loss function to minimize the distance between the rectangle and the points. """
        x_pos, y_pos, width, height, rotation = params
        rect_points = get_rectangle_points(x_pos, y_pos, width, height, rotation)
        return np.sum(np.linalg.norm(np.array(rect_points) - np.array(points), axis=1))
    
    # set bounds
    if force_rotation is not None:
        bounds = [(0, None), (0, None), (0, None), (0, None), (force_rotation, force_rotation)]
    else:
        bounds = [(0, None), (0, None), (0, None), (0, None), (-np.pi, np.pi)]
    result = minimize(loss, [x_pos, y_pos, width, height, rotation], bounds=bounds)
    x_pos, y_pos, width, height, rotation = result.x
    rect_points = get_rectangle_points(x_pos, y_pos, width, height, rotation)
    return rect_points, result.x

def draw_line(pygame, screen, color, start, end, width):
    """Draw a line on the screen."""
    pygame.draw.line(screen, color, start, end, width)

def draw_rectangle(pygame, screen, color, rect, width):
    """Draw a rectangle on the screen."""
    pygame.draw.polygon(screen, color, rect, width)

def draw_circle(pygame, screen, color, center, radius, width):
    """Draw a circle on the screen."""
    pygame.draw.circle(screen, color, center, radius, width)

def draw_arc(pygame, screen, color, rect, start_angle, stop_angle, trail_width, resolution=1/(20*np.pi)):
    """
    Draw an arc on the screen.
    (CUSTOM FUNCTION AS PYGAME DOES NOT SUPPORT DRAWING ARCS WITH SPECIFIC WIDTH)
    """
    x, y, width, height = rect
    start_angle = (start_angle+np.pi)%(2*np.pi)
    stop_angle = (stop_angle+np.pi)%(2*np.pi)
    if start_angle > stop_angle:
        stop_angle += 2*np.pi
    center_x = x + width/2
    center_y = y + height/2
    start_point_x = int(center_x + width/2*np.cos(start_angle))
    start_point_y = int(center_y + height/2*np.sin(start_angle))
    for angle in np.arange(start_angle, stop_angle, resolution):
        end_point_x = int(center_x + width/2*np.cos(angle + resolution))
        end_point_y = int(center_y + height/2*np.sin(angle + resolution))
        # draw line joining the points
        draw_line(pygame, screen, color, (start_point_x, start_point_y), (end_point_x, end_point_y), trail_width)
        start_point_x, start_point_y = end_point_x, end_point_y

def draw_cubic_bezier(pygame, screen, color, points, width, resolution=0.01):
    """
    Draw a cubic bezier curve on the screen.
    (CUSTOM FUNCTION AS PYGAME DOES NOT SUPPORT DRAWING CUBIC BEZIER CURVES)
    """
    for t in np.arange(0, 1, resolution):
        x = (1-t)**3*points[0][0] + 3*t*(1-t)**2*points[1][0] + 3*t**2*(1-t)*points[2][0] + t**3*points[3][0]
        y = (1-t)**3*points[0][1] + 3*t*(1-t)**2*points[1][1] + 3*t**2*(1-t)*points[2][1] + t**3*points[3][1]
        next_x = (1-(t+resolution))**3*points[0][0] + 3*(t+resolution)*(1-(t+resolution))**2*points[1][0] + 3*(t+resolution)**2*(1-(t+resolution))*points[2][0] + (t+resolution)**3*points[3][0]
        next_y = (1-(t+resolution))**3*points[0][1] + 3*(t+resolution)*(1-(t+resolution))**2*points[1][1] + 3*(t+resolution)**2*(1-(t+resolution))*points[2][1] + (t+resolution)**3*points[3][1]
        draw_line(pygame, screen, color, (x, y), (next_x, next_y), width)