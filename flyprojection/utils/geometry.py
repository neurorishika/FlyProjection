import numpy as np
from scipy.optimize import minimize, shgo
from numba import njit

def bounding_boxes_intersect(p1, p2):
    """
    Check if the bounding boxes of two sets of points overlap.

    Parameters
    ----------
    p1 : array_like of shape (N, 2)
        The first set of points, where each point is (x, y).
    p2 : array_like of shape (M, 2)
        The second set of points, where each point is (x, y).

    Returns
    -------
    bool
        True if their bounding boxes overlap, False otherwise.

    Notes
    -----
    This function finds the minimum and maximum x and y coordinates for each set 
    of points, forming two bounding boxes. It then checks if these boxes overlap.
    Using Numba (njit) here provides a slight performance benefit if called frequently.
    """
    p1 = np.array(p1)
    p2 = np.array(p2)

    # Compute bounding boxes
    min_x_1, max_x_1 = np.min(p1[:, 0]), np.max(p1[:, 0])
    min_y_1, max_y_1 = np.min(p1[:, 1]), np.max(p1[:, 1])

    min_x_2, max_x_2 = np.min(p2[:, 0]), np.max(p2[:, 0])
    min_y_2, max_y_2 = np.min(p2[:, 1]), np.max(p2[:, 1])

    # Check overlap along x-axis and y-axis
    overlap_x = not (max_x_1 < min_x_2 or max_x_2 < min_x_1)
    overlap_y = not (max_y_1 < min_y_2 or max_y_2 < min_y_1)

    return overlap_x and overlap_y

@njit
def cartesian_to_polar(state, center):
    """
    Convert Cartesian coordinates to polar coordinates.

    Parameters
    ----------
    state : array_like of shape (4,)
        The state [x, y, x_dot, y_dot] in Cartesian coordinates.
    center : tuple of floats (x_center, y_center)
        The center of the polar coordinate system.

    Returns
    -------
    polar_state : ndarray of shape (4,)
        The state [r, theta, r_dot, theta_dot] in polar coordinates.

    Notes
    -----
    The polar coordinates are defined as:
    r = sqrt((x - x_center)^2 + (y - y_center)^2)
    theta = arctan2(y - y_center, x - x_center)
    r_dot = (x - x_center) * x_dot + (y - y_center) * y_dot) / r
    theta_dot = ((x - x_center) * y_dot - (y - y_center) * x_dot) / r^2
    """

    x, y, x_dot, y_dot = state
    x_center, y_center = center
    x_rel = x - x_center
    y_rel = y - y_center
    r = np.sqrt(x_rel**2 + y_rel**2)

    # Avoid division by zero
    if r == 0.0:
        return np.array([0.0, 0.0, 0.0, 0.0])

    # Compute polar coordinates
    r_inv = 1.0 / r
    theta = np.arctan2(y_rel, x_rel)
    r_dot = (x_rel * x_dot + y_rel * y_dot) * r_inv
    theta_dot = (x_rel * y_dot - y_rel * x_dot) * (r_inv**2)
    return np.array([r, theta, r_dot, theta_dot])

@njit
def polar_to_cartesian(state, center):
    """
    Convert polar coordinates to Cartesian coordinates.

    Parameters
    ----------
    state : array_like of shape (4,)
        The state [r, theta, r_dot, theta_dot] in polar coordinates.
    center : tuple of floats (x_center, y_center)
        The center of the polar coordinate system.

    Returns
    -------
    cartesian_state : ndarray of shape (4,)
        The state [x, y, x_dot, y_dot] in Cartesian coordinates.

    Notes
    -----
    The Cartesian coordinates are defined as:
    x = x_center + r * cos(theta)
    y = y_center + r * sin(theta)
    x_dot = r_dot * cos(theta) - r * theta_dot * sin(theta)
    y_dot = r_dot * sin(theta) + r * theta_dot * cos(theta)
    """
    r, theta, r_dot, theta_dot = state
    x_center, y_center = center
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    x = x_center + r * cos_theta
    y = y_center + r * sin_theta
    x_dot = r_dot * cos_theta - r * theta_dot * sin_theta
    y_dot = r_dot * sin_theta + r * theta_dot * cos_theta
    return np.array([x, y, x_dot, y_dot])

def fit_linear_curve(x, y):
    """
    Fit a linear curve y = m x + c to given data.

    Parameters
    ----------
    x : array_like of shape (N,)
        The x-coordinates of the data points.
    y : array_like of shape (N,)
        The y-coordinates of the data points.

    Returns
    -------
    coeffs : ndarray of shape (2,)
        The linear coefficients [m, c].
    """
    coeffs = np.polyfit(x, y, deg=1)
    return coeffs


def project_to_linear(points, coeffs):
    """
    Project points onto a linear curve y = m x + c.

    Parameters
    ----------
    points : array_like of shape (N, 2)
        The points to project.
    coeffs : array_like of shape (2,)
        The line coefficients [m, c].

    Returns
    -------
    projected_points : ndarray of shape (N, 2)
        The projected points lying on the line.
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
    Subdivide the line between consecutive points along a linear curve y = m x + c.

    Parameters
    ----------
    points : array_like of shape (N, 2)
        The points along which to subdivide.
    coeffs : array_like of shape (2,)
        The line coefficients [m, c].
    subdivisions : int, optional
        Number of subdivisions between each pair of points.

    Returns
    -------
    subdivided_points : ndarray of shape ((N-1)*subdivisions, 2)
        The subdivided points lying on the line.
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

def fit_quadratic_curve(x, y):
    """
    Fit a quadratic curve y = a x^2 + b x + c to given data.

    Parameters
    ----------
    x : array_like of shape (N,)
        The x-coordinates of the data points.
    y : array_like of shape (N,)
        The y-coordinates of the data points.

    Returns
    -------
    coeffs : ndarray of shape (3,)
        The quadratic coefficients [a, b, c].
    """
    coeffs = np.polyfit(x, y, deg=2)
    return coeffs


def project_to_quadratic(points, coeffs):
    """
    Project points onto a quadratic curve defined by y = a x^2 + b x + c.

    Parameters
    ----------
    points : array_like of shape (N, 2)
        The points to be projected onto the quadratic curve.
    coeffs : array_like of shape (3,)
        Coefficients [a, b, c] of the quadratic curve.

    Returns
    -------
    projected_points : ndarray of shape (N, 2)
        The projected points lying on the quadratic curve.

    Notes
    -----
    Each point is projected by minimizing the squared distance 
    to a point on the curve. The optimization uses the SHGO 
    global optimization method to find the best x on [0,3000].
    """
    a, b, c = coeffs
    projected_points = []
    for xi, yi in points:
        def distance_squared(xp):
            yp = a * xp**2 + b * xp + c
            return (xp - xi)**2 + (yp - yi)**2

        # Using SHGO for global optimization over xp in [0,3000]
        res = shgo(distance_squared, bounds=[(0, 3000)], n=100, iters=10)
        xp = res.x
        yp = a * xp**2 + b * xp + c
        projected_points.append((xp, yp))
    return np.array(projected_points)



def ellipse_cartesian_to_polar(coeffs):
    """
    Convert cartesian conic coefficients to ellipse parameters.

    Parameters
    ----------
    coeffs : array_like of shape (6,)
        Ellipse coefficients (a, b, c, d, e, f) defining the conic:
        F(x, y) = a x² + b x y + c y² + d x + e y + f = 0

    Returns
    -------
    (x0, y0, ap, bp, e, phi) : tuple
        x0, y0 : float
            Ellipse center coordinates.
        ap, bp : float
            Semi-major and semi-minor axes lengths.
        e : float
            Eccentricity of the ellipse.
        phi : float
            Rotation angle of the ellipse in radians.

    Raises
    ------
    ValueError
        If the provided coefficients do not represent a valid ellipse.

    Notes
    -----
    Based on formulas from: https://mathworld.wolfram.com/Ellipse.html
    """
    a = coeffs[0]
    b = coeffs[1] / 2.0
    c = coeffs[2]
    d = coeffs[3] / 2.0
    f = coeffs[4] / 2.0
    g = coeffs[5]

    den = b**2 - a*c
    if den > 0:
        raise ValueError("Provided coefficients do not represent an ellipse (b² - 4ac must be negative).")

    x0 = (c*d - b*f) / den
    y0 = (a*f - b*d) / den

    num = 2 * (a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g)
    fac = np.sqrt((a - c)**2 + 4*b**2)
    ap = np.sqrt(num / den / (fac - a - c))
    bp = np.sqrt(num / den / (-fac - a - c))

    width_gt_height = True
    if ap < bp:
        width_gt_height = False
        ap, bp = bp, ap

    r = (bp/ap)**2
    if r > 1:
        r = 1/r
    e = np.sqrt(1 - r)

    if b == 0:
        if a < c:
            phi = 0
        else:
            phi = np.pi/2
    else:
        phi = np.arctan((2.*b) / (a - c)) / 2
        if a > c:
            phi += np.pi/2
    if not width_gt_height:
        phi += np.pi/2
    phi = phi % np.pi

    return (x0, y0, ap, bp, e, phi)


def fit_ellipse(x, y):
    """
    Fit an ellipse to given data points using a direct least squares method.

    Parameters
    ----------
    x : array_like of shape (N,)
        The x-coordinates of the data points.
    y : array_like of shape (N,)
        The y-coordinates of the data points.

    Returns
    -------
    (x0, y0, ap, bp, e, phi) : tuple
        Ellipse parameters. See `ellipse_cartesian_to_polar` for details.

    Notes
    -----
    Based on Halir and Flusser's method:
    "Numerically Stable Direct Least Squares Fitting of Ellipses".
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
    return ellipse_cartesian_to_polar(np.concatenate((ak, T @ ak)).ravel())


def project_to_ellipse(points, ellipse_params):
    """
    Project points onto an ellipse defined by its parameters.

    Parameters
    ----------
    points : array_like of shape (N, 2)
        Points to be projected.
    ellipse_params : tuple
        (x0, y0, ap, bp, e, phi) defining the ellipse.

    Returns
    -------
    projected_points : ndarray of shape (N, 2)
        The projected points on the ellipse.

    Notes
    -----
    Each point is projected by finding the closest point on the ellipse 
    (parametrized by t) that minimizes the squared distance. Uses `scipy.optimize.minimize`.
    """
    x0, y0, ap, bp, e, phi = ellipse_params
    projected_points = np.array(points, dtype=float)
    for i, (x, y) in enumerate(points):
        def distance(t):
            x_pos = x0 + ap * np.cos(t) * np.cos(phi) - bp * np.sin(t) * np.sin(phi)
            y_pos = y0 + ap * np.cos(t) * np.sin(phi) + bp * np.sin(t) * np.cos(phi)
            return (x_pos - x)**2 + (y_pos - y)**2

        result = minimize(distance, 0)
        t = result.x[0]
        x_pos = x0 + ap * np.cos(t) * np.cos(phi) - bp * np.sin(t) * np.sin(phi)
        y_pos = y0 + ap * np.cos(t) * np.sin(phi) + bp * np.sin(t) * np.cos(phi)
        projected_points[i] = (x_pos, y_pos)
    return projected_points


def interpolate_on_ellipse(start_point, ellipse_params, num_points=100):
    """
    Generate a set of points on the ellipse, starting at the closest point to a given start_point 
    and continuing one full revolution (2π) around the ellipse.

    Parameters
    ----------
    start_point : tuple (x, y)
        The point from which to start the interpolation (closest point on the ellipse).
    ellipse_params : tuple
        (x0, y0, ap, bp, e, phi) defining the ellipse.
    num_points : int, optional
        Number of points to generate along the ellipse.

    Returns
    -------
    new_points : list of tuples
        The interpolated points along the ellipse.
    """
    x0, y0, ap, bp, e, phi = ellipse_params
    x, y = start_point

    def distance(t):
        x_pos = x0 + ap * np.cos(t) * np.cos(phi) - bp * np.sin(t) * np.sin(phi)
        y_pos = y0 + ap * np.cos(t) * np.sin(phi) + bp * np.sin(t) * np.cos(phi)
        return (x_pos - x)**2 + (y_pos - y)**2

    result = minimize(distance, 0)
    t_start = result.x[0]
    t_end = t_start + 2*np.pi
    ts = np.linspace(t_start, t_end, num_points)
    new_points = []
    for t in ts:
        x_pos = x0 + ap * np.cos(t) * np.cos(phi) - bp * np.sin(t) * np.sin(phi)
        y_pos = y0 + ap * np.cos(t) * np.sin(phi) + bp * np.sin(t) * np.cos(phi)
        new_points.append((x_pos, y_pos))
    return new_points


def subdivide_on_ellipse(points, ellipse_params, subdivisions=10):
    """
    Subdivide the ellipse between given points that lie on it or close to it.

    Parameters
    ----------
    points : array_like of shape (N, 2)
        Points on or near the ellipse.
    ellipse_params : tuple
        (x0, y0, ap, bp, e, phi) defining the ellipse.
    subdivisions : int, optional
        Number of subdivisions between each pair of successive points.

    Returns
    -------
    new_points : list of tuples
        The subdivided points on the ellipse.
    """
    x0, y0, ap, bp, e, phi = ellipse_params
    ts = []
    for point in points:
        x, y = point
        def distance(t):
            x_pos = x0 + ap * np.cos(t) * np.cos(phi) - bp * np.sin(t) * np.sin(phi)
            y_pos = y0 + ap * np.cos(t) * np.sin(phi) + bp * np.sin(t) * np.cos(phi)
            return (x_pos - x)**2 + (y_pos - y)**2
        result = shgo(distance, bounds=[(0, 2*np.pi)], n=100, iters=10)
        ts.append(result.x[0])
    ts = np.array(ts)
    # Ensure a full wrap-around
    ts = np.concatenate([ts, [ts[0] + 2*np.pi]])
    ts = np.unwrap(ts)
    new_points = []
    for i in range(len(ts)-1):
        t_start = ts[i]
        t_end = ts[i+1]
        ts_sub = np.linspace(t_start, t_end, subdivisions+2)[:-1]
        for t in ts_sub:
            x_pos = x0 + ap * np.cos(t) * np.cos(phi) - bp * np.sin(t) * np.sin(phi)
            y_pos = y0 + ap * np.cos(t) * np.sin(phi) + bp * np.sin(t) * np.cos(phi)
            new_points.append((x_pos, y_pos))
    return new_points


@njit
def get_rectangle_points(x, y, width, height, rotation):
    """
    Compute the 4 corner points of a rectangle given its center, width, height, and rotation.

    Parameters
    ----------
    x : float
        The x-coordinate of the rectangle center.
    y : float
        The y-coordinate of the rectangle center.
    width : float
        The width of the rectangle.
    height : float
        The height of the rectangle.
    rotation : float
        The rotation angle of the rectangle in radians.

    Returns
    -------
    corners : list of tuples
        The 4 corner points of the rotated rectangle.

    Notes
    -----
    With Numba's njit, this calculation is significantly faster for repeated calls.
    """
    xs = np.array([x - width/2, x + width/2, x + width/2, x - width/2])
    ys = np.array([y - height/2, y - height/2, y + height/2, y + height/2])

    cos_r = np.cos(rotation)
    sin_r = np.sin(rotation)

    x_rot = xs*cos_r - ys*sin_r
    y_rot = xs*sin_r + ys*cos_r

    corners = [(x_rot[i], y_rot[i]) for i in range(4)]
    return corners


def rect_fit(points, force_rotation=None):
    """
    Fit a rectangle to 4 given corner points by adjusting position, width, height, and rotation.

    Parameters
    ----------
    points : array_like of shape (4, 2)
        Four corner points of the rectangle (not necessarily a perfect rectangle).
    force_rotation : float, optional
        If provided, the rotation angle is fixed to this value.

    Returns
    -------
    rect_points : list of tuples
        The 4 corner points of the fitted rectangle.
    params : ndarray of shape (5,)
        The fitted parameters [x_pos, y_pos, width, height, rotation].

    Notes
    -----
    The fitting is done by minimizing the sum of distances between the given points and the 
    proposed rectangle corners. If force_rotation is given, the rotation is fixed and not optimized.
    """
    # Initial guesses
    x_pos = np.mean([p[0] for p in points])
    y_pos = np.mean([p[1] for p in points])
    width = np.max([p[0] for p in points]) - np.min([p[0] for p in points])
    height = np.max([p[1] for p in points]) - np.min([p[1] for p in points])
    rotation = 0

    def loss(params):
        x_pos_, y_pos_, width_, height_, rotation_ = params
        rect_points = get_rectangle_points(x_pos_, y_pos_, width_, height_, rotation_)
        return np.sum(np.linalg.norm(np.array(rect_points) - np.array(points), axis=1))

    if force_rotation is not None:
        bounds = [(None, None), (None, None), (0, None), (0, None), (force_rotation, force_rotation)]
    else:
        bounds = [(None, None), (None, None), (0, None), (0, None), (-np.pi, np.pi)]

    result = minimize(loss, [x_pos, y_pos, width, height, rotation], bounds=bounds)
    x_pos, y_pos, width, height, rotation = result.x
    rect_points = get_rectangle_points(x_pos, y_pos, width, height, rotation)
    return rect_points, result.x
