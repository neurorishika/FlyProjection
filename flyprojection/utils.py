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

def get_boolean_answer(prompt, default=None):
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

from scipy.optimize import minimize
import numpy as np

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
    