import cv2
import torch
import kornia
import numpy as np
from scipy.interpolate import griddata
from flyprojection.utils.utils import tensor_to_numpy, numpy_to_tensor

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

def remap_coords_with_interpolation(coords, X, Y, method='linear'): 
    """
    Remap coordinates based on the interpolation from points X to points Y.

    Parameters:
    - coords: The input coordinates to be remapped (numpy array of shape (N, 2)).
    - X: Source points for the mapping (numpy array of shape (N, 2)).
    - Y: Target points corresponding to X (numpy array of shape (N, 2)).
    - method: Interpolation method to use (string, default 'linear').

    Returns:
    - remapped_coords: The remapped coordinates (numpy array).
    """

    # Perform grid data interpolation
    remapped_coords = griddata(X, Y, coords, method=method)

    return remapped_coords

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

def remap_coords_with_map(coords, interpolation_mapx, interpolation_mapy):
    """
    Remap coordinates based on a precomputed interpolation map.

    Parameters:
    - coords: The input coordinates to be remapped (numpy array of shape (N, 2)).
    - interpolation_mapx: The interpolation map for remapping the coordinates in the x-direction (numpy array).
    - interpolation_mapy: The interpolation map for remapping the coordinates in the y-direction (numpy array).

    Returns:
    - remapped_coords: The remapped coordinates (numpy array).
    """

    # use the interpolation map to find the new coordinates
    new_coords = []
    for coord in coords:
        x, y = coord
        new_x = interpolation_mapx[int(y), int(x)]
        new_y = interpolation_mapy[int(y), int(x)]
        new_coords.append([new_x, new_y])
    remapped_coords = np.array(new_coords)

    return remapped_coords