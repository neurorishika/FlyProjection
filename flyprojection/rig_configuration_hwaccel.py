import os
import sys
import argparse
import datetime
import time
import json
import shutil
import threading
import multiprocessing
from itertools import product
import numpy as np
import pygame
import torch
import kornia
import matplotlib.pyplot as plt
import cv2
import apriltag

from flyprojection.utils import *
from flyprojection.controllers.basler_camera import BaslerCamera, list_basler_cameras

# Function to continuously display images from the queue in a separate process
def display_images(queue):
    cv2.namedWindow("Projection Feed", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Projection Feed", 1300, 1300)
    
    while True:
        if not queue.empty():
            image = queue.get()  # Get the latest image from the queue
            if image is None:  # Check for exit signal
                break
            # Rotate the image 180 degrees
            image = cv2.rotate(image, cv2.ROTATE_180)
            cv2.imshow("Projection Feed", image)
            cv2.waitKey(1)

    cv2.destroyWindow("Projection Feed")

# Function for mouse callback in OpenCV
def mouse_points_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point stored: {points[-1]}")
    elif event == cv2.EVENT_RBUTTONDOWN:
        if points:
            points.pop()
            print("Point removed.")

# Terminology:
# camera_space: the (pixel x pixel) space of the camera
# projector_space: the (pixel x pixel) space of the projector
# world_space: the (mm x mm) space of the physical world
# projection_space: the (pixel x pixel) space of the projected image on the camera

if __name__ == "__main__":

    # Set the device to be used
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create a multiprocessing queue
    queue = multiprocessing.Manager().Queue()

    # Set up the argument parser
    parser = argparse.ArgumentParser(description='Open/Closed Loop Fly Projection System Rig Configuration')
    parser.add_argument('--repo_dir', type=str, default='/mnt/sda1/Rishika/FlyProjection/', help='Path to the repository directory')
    parser.add_argument('--display_wait_time', type=int, default=500, help='Time to wait before closing the display window (in ms)')
    parser.add_argument('--homography_iterations', type=int, default=10, help='Number of iterations to find the homography between the camera and projector detections')
    args = parser.parse_args()
    repo_dir = args.repo_dir

    # Load the rig configuration
    assert os.path.isdir(os.path.join(repo_dir, 'configs')), "Invalid configs directory"
    assert os.path.isfile(os.path.join(repo_dir, 'configs', 'rig_config_hwaccel.json')), "rig_config_hwaccel.json file not found"
    with open(os.path.join(repo_dir, 'configs', 'rig_config_hwaccel.json'), 'r') as f:
        rig_config = json.load(f)

    # Backup the rig_config_hwaccel.json file
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.copy(os.path.join(repo_dir, 'configs', 'rig_config_hwaccel.json'),
                os.path.join(repo_dir, 'configs', 'archived_configs', f'rig_config_until_{current_time}.json'))

    # Setup projector dimensions
    if get_boolean_answer(f"Use current projector dimensions ({rig_config['projector_width']}x{rig_config['projector_height']})? [Y/n] ", default=True):
        projector_width = int(rig_config['projector_width'])
        projector_height = int(rig_config['projector_height'])
    else:
        projector_width = int(input("Enter the projector width: "))
        projector_height = int(input("Enter the projector height: "))

    rig_config['projector_width'] = projector_width
    rig_config['projector_height'] = projector_height

    # List and select camera
    cameras = list_basler_cameras()
    if not cameras:
        print("No cameras found. Exiting.")
        sys.exit()
    else:
        print(f"Found {len(cameras)} camera(s).")
        if len(cameras) > 1:
            if get_boolean_answer(f"Use current camera index ({rig_config['camera_index']})? [Y/n] ", default=True):
                camera_index = int(rig_config['camera_index'])
            else:
                camera_index = int(input("Enter the index of the camera you would like to use: "))
            assert camera_index < len(cameras), "Invalid camera index."
        else:
            camera_index = 0
    rig_config['camera_index'] = camera_index

    # Setup FPS
    if get_boolean_answer(f"Use current FPS ({rig_config['FPS']})? [Y/n] ", default=True):
        FPS = int(rig_config['FPS'])
    else:
        FPS = int(input("Enter the FPS: "))
    rig_config['FPS'] = FPS

    # Initialize Pygame
    os.environ['SDL_VIDEO_WINDOW_POS'] = f"0,0"
    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((projector_width, projector_height), pygame.NOFRAME | pygame.HWSURFACE | pygame.DOUBLEBUF)


    # Start the camera
    with BaslerCamera(index=camera_index, FPS=FPS, record_video=False, EXPOSURE_TIME=2000) as cam:
        cam.start()

        #### CALIBRATION STEP 1: FIND THE ARENA IN THE PROJECTOR SPACE ####

        # Check if projector_space_arena_center and projector_space_arena_radius are present in the rig_config
        if 'projector_space_arena_center' in rig_config and 'projector_space_arena_radius' in rig_config:
            # Ask the user if the current projector_space_arena_center and projector_space_arena_radius should be used
            if get_boolean_answer(f"Do you want to use the current projector_space_arena_center ({rig_config['projector_space_arena_center']}) and projector_space_arena_radius ({rig_config['projector_space_arena_radius']})? [Y/n] ", default=True):
                projector_space_arena_center = np.array(rig_config['projector_space_arena_center'])
                projector_space_arena_radius = rig_config['projector_space_arena_radius']
            else:
                projector_space_arena_center = None
                projector_space_arena_radius = None
        else:
            projector_space_arena_center = None
            projector_space_arena_radius = None

        if projector_space_arena_center is None or projector_space_arena_radius is None:

            # Ask user if live feed should be displayed (might cause lag)
            if get_boolean_answer("Do you want to display the live feed? [y/N] ", default=False):
                display_live_feed = True
            else:
                display_live_feed = False

            if display_live_feed:
                # Start the OpenCV display process
                display_process = multiprocessing.Process(target=display_images, args=(queue,))
                display_process.start()

            # Set up font for displaying text
            font = pygame.font.Font(None, 20)

            # List to store points
            points = []

            # Change the cursor to a crosshair
            crosshair_cursor = pygame.Surface((10, 10), pygame.SRCALPHA)
            pygame.draw.line(crosshair_cursor, (255, 255, 255), (5, 0), (5, 10), 2)
            pygame.draw.line(crosshair_cursor, (255, 255, 255), (0, 5), (10, 5), 2)
            pygame.mouse.set_visible(False)  # Hide the default cursor

            # Main loop
            collecting_points = True
            while collecting_points:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    
                    if display_live_feed:
                        # Capture image from the camera
                        image = cam.get_array(dont_save=True)  # Capture a frame
                        image = np.clip(image, 0, 255).astype(np.uint8)  # Format the image
                        queue.put(image)  # Put the image in the queue

                    # Check for mouse button down event
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        # Store the current mouse position in the points list if the left mouse button is pressed
                        if event.button == 1:
                            points.append(pygame.mouse.get_pos())
                            print(f"Point stored: {points[-1]}")  # Print the stored point to the console
                            
                            # Exit the loop if we have 4 points
                            if len(points) >= 4:
                                # Clear the screen
                                screen.fill((0, 0, 0))
                                
                                if display_live_feed:
                                    # Send exit signal to the display process
                                    queue.put(None)
                                    display_process.join()

                                # Stop the loop
                                collecting_points = False
                                break
                        # Remove the last point if the right mouse button is pressed
                        elif event.button == 3:
                            points.pop()
                            print("Point removed.")

                # Get the current mouse position
                mouse_x, mouse_y = pygame.mouse.get_pos()

                # Fill the screen with black
                screen.fill((0, 0, 0))

                # Create text surface with cursor coordinates
                text_surface = font.render(f'({mouse_x}, {mouse_y})', True, (255, 255, 255))

                # Blit the text next to the cursor
                screen.blit(text_surface, (mouse_x + 10, mouse_y))

                # Draw the stored points and mark them
                for point in points:
                    # Draw a crosshair at the point with the coordinates
                    pygame.draw.line(screen, (255, 255, 255), (point[0] - 5, point[1]), (point[0] + 5, point[1]), 2)
                    pygame.draw.line(screen, (255, 255, 255), (point[0], point[1] - 5), (point[0], point[1] + 5), 2)

                    text_surface = font.render(f'({point[0]}, {point[1]})', True, (255, 255, 255))
                    screen.blit(text_surface, (point[0] + 10, point[1]))

                # Draw the custom cursor
                screen.blit(crosshair_cursor, (mouse_x - 5, mouse_y - 5))  # Center the cursor on the mouse position

                # Update the display
                pygame.display.flip()

                # Limit the frame rate
                clock.tick(30)

            # Release the cursor
            pygame.mouse.set_visible(True)

            # Fit a circle to the marked points
            points = np.array(points)
            projector_space_radius, projector_space_center = nsphere_fit(points)
            print(f"Center: {projector_space_center}, Radius: {projector_space_radius}")

            # Draw the circle on the screen
            screen.fill((0, 0, 0))
            pygame.draw.circle(screen, (255, 255, 255), projector_space_center.astype(int), int(projector_space_radius), 2)
            pygame.display.flip()

            # Ask the user if the circle is correct
            if get_boolean_answer("Is the circle correct? [Y/n] ", default=True):
                pass
            else:
                pygame.quit()
                sys.exit()

            # Save the projector_space_arena_center and projector_space_arena_radius
            rig_config['projector_space_arena_center'] = projector_space_center.tolist()
            rig_config['projector_space_arena_radius'] = projector_space_radius

            # Save the rig configuration
            with open(os.path.join(repo_dir, 'configs', 'rig_config_hwaccel.json'), 'w') as f:
                json.dump(rig_config, f, indent=4)

        projector_space_radius = rig_config['projector_space_arena_radius']
        projector_space_center = np.array(rig_config['projector_space_arena_center'])

        #### CALIBRATION STEP 2: FIND THE DISTORTION IN THE PROJECTION ####

        # Check if camera_space_arena_center and camera_space_arena_radius are present in the rig_config
        if 'camera_space_arena_center' in rig_config and 'camera_space_arena_radius' in rig_config:
            # Ask the user if the current camera_space_arena_center and camera_space_arena_radius should be used
            if get_boolean_answer(f"Do you want to use the current camera_space_arena_center ({rig_config['camera_space_arena_center']}) and camera_space_arena_radius ({rig_config['camera_space_arena_radius']})? [Y/n] ", default=True):
                camera_space_arena_center = np.array(rig_config['camera_space_arena_center'])
                camera_space_arena_radius = rig_config['camera_space_arena_radius']
            else:
                camera_space_arena_center = None
                camera_space_arena_radius = None
        else:
            camera_space_arena_center = None
            camera_space_arena_radius = None

        if camera_space_arena_center is None or camera_space_arena_radius is None:
            # Draw the arena on the screen
            screen.fill((0, 0, 0))
            pygame.draw.circle(screen, (255, 255, 255), projector_space_center.astype(int), int(projector_space_radius), 2)
            pygame.display.flip()

            # Wait a few seconds and then take a picture
            time.sleep(2)

            # Get a camera image as a tensor
            camera_image = cam.get_array(dont_save=True)  # Capture a frame

            # Display the camera image and mark points on it
            cv2.namedWindow("Mark Arena on the Camera Image", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Mark Arena on the Camera Image", 1300, 1300)

            # Set up the mouse callback
            points = []

            # Remove any existing mouse callback
            cv2.setMouseCallback("Mark Arena on the Camera Image", lambda *args: None)

            # Set the mouse callback
            cv2.setMouseCallback("Mark Arena on the Camera Image", mouse_points_callback)

            # Make a copy of the camera image
            camera_image_with_points = camera_image.copy()

            # Show the camera image and mark the points
            while True:
                # Keep the pygame event loop running
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

                camera_image_with_points = camera_image.copy()
                for point in points:
                    # Draw a crosshair at the point with the coordinates
                    cv2.line(camera_image_with_points, (point[0] - 20, point[1]), (point[0] + 20, point[1]), (255, 255, 255), 5)
                    cv2.line(camera_image_with_points, (point[0], point[1] - 20), (point[0], point[1] + 20), (255, 255, 255), 5)

                # Fit and draw a circle to the points if there are at least 4 points
                if len(points) >= 4:
                    camera_space_arena_radius, camera_space_arena_center = nsphere_fit(np.array(points))
                    cv2.circle(camera_image_with_points, tuple(camera_space_arena_center.astype(int)), int(camera_space_arena_radius), (255, 255, 255), 2)
                
                # Write instructions on the image
                cv2.putText(camera_image_with_points, "Click on points on the circle. Press 'q' and Alt+Tab back to VSCode", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                cv2.imshow("Mark Arena on the Camera Image", camera_image_with_points)

                # Break if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q') and len(points) >= 4:
                    break

            # Stop the mouse callback
            cv2.setMouseCallback("Mark Arena on the Camera Image", lambda *args: None)

            # Convert the points to numpy array
            points = np.array(points)

            # Fit a circle to the points
            camera_space_arena_radius, camera_space_arena_center = nsphere_fit(points)

            # Draw the circle on the camera image
            cv2.circle(camera_image_with_points, tuple(camera_space_arena_center.astype(int)), int(camera_space_arena_radius), (255, 255, 255), 2)

            # Display the camera image with the circle
            cv2.imshow("Mark Arena on the Camera Image", camera_image_with_points)
            cv2.waitKey(0)
            cv2.destroyWindow("Mark Arena on the Camera Image")

        # Save the camera_space_arena_center and camera_space_arena_radius
        rig_config['camera_space_arena_center'] = camera_space_arena_center.tolist()
        rig_config['camera_space_arena_radius'] = camera_space_arena_radius

        # Save the rig configuration
        with open(os.path.join(repo_dir, 'configs', 'rig_config_hwaccel.json'), 'w') as f:
            json.dump(rig_config, f, indent=4)

        # Ask the user for the degree of detail in this step of the calibration process (4-20)
        if 'calibration_detail' in rig_config:
            default_detail = rig_config['calibration_detail']
        else:
            default_detail = 12

        detail = get_predefined_answer(f"Calibration Degree of Detail [3-12] (Default: {default_detail}) ", [str(i) for i in range(3, 13)], default=str(default_detail))
        detail = int(detail)

        flag_for_calibration = False
        if default_detail != detail:
            flag_for_calibration = True

        # Define detail radial grid points in a uniform manner
        radial = np.linspace(0, 1, detail//3+2)[1:]
        angular = np.linspace(0, 2*np.pi, detail+1)[:-1]

        detail_markers = list(product(radial, angular))

        # Save the degree of detail in the rig_config
        rig_config['calibration_detail'] = detail

        # Save the rig configuration
        with open(os.path.join(repo_dir, 'configs', 'rig_config_hwaccel.json'), 'w') as f:
            json.dump(rig_config, f, indent=4)

        # Check if projection_space_detail_markers and projector_space_detail_markers are present in the rig_config and flag_for_calibration is False
        if 'projection_space_detail_markers' in rig_config and 'projector_space_detail_markers' in rig_config and not flag_for_calibration:
            # Ask the user if the current projection_space_detail_markers and projector_space_detail_markers should be used
            if get_boolean_answer(f"Do you want to use the current projection_space_detail_markers and projector_space_detail_markers? [Y/n] ", default=True):
                projection_space_detail_markers = np.array(rig_config['projection_space_detail_markers'])
                projector_space_detail_markers = np.array(rig_config['projector_space_detail_markers'])
            else:
                projection_space_detail_markers = None
                projector_space_detail_markers = None
        else:
            projection_space_detail_markers = None
            projector_space_detail_markers = None

        if projection_space_detail_markers is None or projector_space_detail_markers is None:
            # Find the corresponding points in the projector space
            projector_space_detail_markers = [(projector_space_center[0] + marker[0]*projector_space_radius*np.cos(marker[1]),
                                               projector_space_center[1] + marker[0]*projector_space_radius*np.sin(marker[1])) for marker in detail_markers]

            # Draw the markers on the screen
            screen.fill((0, 0, 0))
            font = pygame.font.Font(None, 20)
            for idx, marker in enumerate(projector_space_detail_markers):
                pygame.draw.circle(screen, (255, 255, 255), (int(marker[0]), int(marker[1])), 2)
                # Put the number of the marker next to it
                text = font.render(str(idx), True, (255, 255, 255))
                screen.blit(text, (int(marker[0]) + 5, int(marker[1]) + 5))
            pygame.display.flip()

            # Wait a few seconds and then take a picture
            time.sleep(2)

            # Get a camera image
            camera_image_tensor = cam.get_tensor(dont_save=True)  # Shape: [1, 1, H, W]
            camera_image = camera_image_tensor.squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)

            # Display the camera image and mark points on it
            cv2.namedWindow("Mark Markers on the Camera Image", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Mark Markers on the Camera Image", 1300, 1300)

            cv2.setMouseCallback("Mark Markers on the Camera Image", mouse_points_callback)

            points = []

            # Make a copy of the camera image
            camera_image_with_points = camera_image.copy()

            num_markers = (detail//3+1) * detail

            # Show the camera image and mark the points
            while len(points) < num_markers:
                camera_image_with_points = camera_image.copy()
                for point in points:
                    # Draw a crosshair at the point with the coordinates
                    cv2.line(camera_image_with_points, (point[0] - 20, point[1]), (point[0] + 20, point[1]), (255, 255, 255), 5)
                    cv2.line(camera_image_with_points, (point[0], point[1] - 20), (point[0], point[1] + 20), (255, 255, 255), 5)
                cv2.imshow("Mark Markers on the Camera Image", camera_image_with_points)
                cv2.waitKey(1)

            # Stop the mouse callback
            cv2.setMouseCallback("Mark Markers on the Camera Image", lambda *args: None)

            # Destroy the window
            cv2.destroyWindow("Mark Markers on the Camera Image")

            # Convert the points to numpy array
            projection_space_detail_markers = np.array(points)
            projector_space_detail_markers = np.array(projector_space_detail_markers)

        # Save the projection_space_detail_markers and projector_space_detail_markers
        rig_config['projection_space_detail_markers'] = projection_space_detail_markers.tolist()
        rig_config['projector_space_detail_markers'] = projector_space_detail_markers.tolist()

        # Save the rig configuration
        with open(os.path.join(repo_dir, 'configs', 'rig_config_hwaccel.json'), 'w') as f:
            json.dump(rig_config, f, indent=4)

        # Get a camera image
        camera_image = cam.get_array(dont_save=True)  # Capture a frame

        fitted_projection_space_detail_markers = np.zeros_like(projection_space_detail_markers)
        fitted_projector_space_detail_markers = np.zeros_like(projector_space_detail_markers)

        # For each outward radial line, fit a linear curve to the points and project onto the curve
        for i in range(detail):
            # Get the points for the radial line
            projection_space_radial_line_points = projection_space_detail_markers[i::detail]

            # Fit a linear curve to the points
            projection_space_radial_line_params = fit_linear_curve(projection_space_radial_line_points[:, 0], projection_space_radial_line_points[:, 1])

            # Get projected points on the curve
            projected_points = project_to_linear(projection_space_radial_line_points, projection_space_radial_line_params)

            # Add the points to the fitted_projection_space_detail_markers
            fitted_projection_space_detail_markers[i::detail] = projected_points[:, :]

            # Get the points for the radial line
            projector_space_radial_line_points = projector_space_detail_markers[i::detail]

            # Fit a linear curve to the points
            projector_space_radial_line_params = fit_linear_curve(projector_space_radial_line_points[:, 0], projector_space_radial_line_points[:, 1])

            # Get projected points on the curve
            projected_points = project_to_linear(projector_space_radial_line_points, projector_space_radial_line_params)

            # Add the points to the fitted_projector_space_detail_markers
            fitted_projector_space_detail_markers[i::detail] = projected_points[:, :]

        # Save the fitted_projection_space_detail_markers and fitted_projector_space_detail_markers
        rig_config['fitted_projection_space_detail_markers'] = fitted_projection_space_detail_markers.tolist()
        rig_config['fitted_projector_space_detail_markers'] = fitted_projector_space_detail_markers.tolist()

        # Save the rig configuration
        with open(os.path.join(repo_dir, 'configs', 'rig_config_hwaccel.json'), 'w') as f:
            json.dump(rig_config, f, indent=4)

        interpolated_projection_space_detail_markers = []
        interpolated_projector_space_detail_markers = []

        # For each concentric ellipse, fit an ellipse to the points and interpolate the points on the ellipse
        for i in range(detail//3+1):
            # Get the points for the ellipse
            projection_space_ellipse_points = projection_space_detail_markers[i*detail:(i+1)*detail]

            # Fit an ellipse to the points
            projection_space_ellipse_params = fit_ellipse(projection_space_ellipse_points[:, 0], projection_space_ellipse_points[:, 1])

            # Interpolate the points on the ellipse
            projection_space_ellipse_points = subdivide_on_ellipse(projection_space_ellipse_points, projection_space_ellipse_params, 10)

            # Add the points to the interpolated_projection_space_detail_markers
            interpolated_projection_space_detail_markers.extend(projection_space_ellipse_points)

            # Get the points for the ellipse
            projector_space_ellipse_points = projector_space_detail_markers[i*detail:(i+1)*detail]

            # Fit an ellipse to the points
            projector_space_ellipse_params = fit_ellipse(projector_space_ellipse_points[:, 0], projector_space_ellipse_points[:, 1])

            # Interpolate the points on the ellipse
            projector_space_ellipse_points = subdivide_on_ellipse(projector_space_ellipse_points, projector_space_ellipse_params, 10)

            # Add the points to the interpolated_projector_space_detail_markers
            interpolated_projector_space_detail_markers.extend(projector_space_ellipse_points)

        # Convert the lists to numpy arrays
        interpolated_projection_space_detail_markers = np.array(interpolated_projection_space_detail_markers)
        interpolated_projector_space_detail_markers = np.array(interpolated_projector_space_detail_markers)

        # Save the interpolated_projection_space_detail_markers and interpolated_projector_space_detail_markers
        rig_config['interpolated_projection_space_detail_markers'] = interpolated_projection_space_detail_markers.tolist()
        rig_config['interpolated_projector_space_detail_markers'] = interpolated_projector_space_detail_markers.tolist()

        # Draw the markers on the camera image
        cv2.namedWindow("Interpolated Markers on the Camera Image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Interpolated Markers on the Camera Image", 1300, 1300)
        camera_image_with_markers = camera_image.copy()
        for marker in interpolated_projection_space_detail_markers:
            cv2.circle(camera_image_with_markers, tuple(np.int32(marker)), 2, (255, 255, 255), -1)
        cv2.imshow("Interpolated Markers on the Camera Image", camera_image_with_markers)
        cv2.waitKey(args.display_wait_time)
        cv2.destroyWindow("Interpolated Markers on the Camera Image")

        # Place a calibration tag on the screen just within the circle
        screen.fill((0, 0, 0))

        # Calculate inscribed square half-side
        insquare_halfside = projector_space_radius / np.sqrt(2)

        # Get the apriltag image and place it within the circle
        april_tag = pygame.image.load(os.path.join(repo_dir, 'assets', 'apriltag.png'))
        april_tag = pygame.transform.scale(april_tag, (int(2*insquare_halfside), int(2*insquare_halfside)))

        # Place the apriltag on the screen
        screen.blit(april_tag, (projector_space_center[0] - insquare_halfside, projector_space_center[1] - insquare_halfside))

        # Update the display
        pygame.display.flip()

        # Find Apriltag in camera image
        finding_apriltag = True
        while finding_apriltag:

            # Keep the pygame event loop running
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # Wait for the camera to adjust
            time.sleep(0.1)

            # Get the camera image
            image= cam.get_array(dont_save=True)  # Capture a frame

            # Increase the contrast of the image using CLAHE (cv2)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            image = clahe.apply(image)

            # Apply Otsu thresholding
            _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Show the image
            cv2.imshow("Camera Image", image)
            cv2.waitKey(args.display_wait_time)
            cv2.destroyWindow("Camera Image")

            # Initialize the AprilTag detector
            detector = apriltag.Detector()

            # Detect AprilTags
            detections = detector.detect(image)

            # Check if any AprilTag is detected
            if len(detections) == 0:
                print("Apriltag not detected. Conducting the process again.")
                continue
            else:
                finding_apriltag = False
                assert len(detections) == 1, f"Multiple AprilTags detected: {len(detections)}"
                camera_detection = detections[0]
                print(f"Apriltag detected in camera image. ID: {camera_detection.tag_id}")

        # Find Apriltag in projected image
        finding_apriltag = True
        while finding_apriltag:

            # Keep the pygame event loop running
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # Get the screen as an array
            projected_image = pygame.surfarray.array3d(screen).copy().transpose((1, 0, 2))

            # convert to grayscale
            projected_image = cv2.cvtColor(projected_image, cv2.COLOR_RGB2GRAY)

            # Initialize the AprilTag detector
            detector = apriltag.Detector()

            # Detect AprilTags
            detections = detector.detect(projected_image)

            # Check if any AprilTag is detected
            if len(detections) == 0:
                print("Apriltag not detected. Conducting the process again.")
                continue
            else:
                finding_apriltag = False
                assert len(detections) == 1, f"Multiple AprilTags detected: {len(detections)}"
                projector_detection = detections[0]
                print(f"Apriltag detected in projected image. ID: {projector_detection.tag_id}")

        # Convert src and dst points to tensors
        src = np.array([point for point in camera_detection.corners])  # Shape: [4, 2]
        dst = np.array([point for point in projector_detection.corners])  # Shape: [4, 2]

        src_tensor = torch.tensor(src, dtype=torch.float32).unsqueeze(0).to(device)  # Shape: [1, 4, 2]
        dst_tensor = torch.tensor(dst, dtype=torch.float32).unsqueeze(0).to(device)  # Shape: [1, 4, 2]

        # Estimate homography using Kornia
        Hs = []
        for i in range(args.homography_iterations):
            H = kornia.geometry.homography.find_homography_dlt(src_tensor, dst_tensor)  # Shape: [1, 3, 3]
            Hs.append(H)

        # Average the homographies
        H_tensor = torch.stack(Hs, dim=0).mean(dim=0)  # Shape: [1, 3, 3]
        H = H_tensor.squeeze(0).cpu().numpy()

        # Calculate the size of each square in the calibration pattern
        N_squares = 13
        square_size = 2 * insquare_halfside / N_squares

        # Draw the calibration pattern on the screen
        screen.fill((0, 0, 0))

        # Add a white border around the calibration pattern
        expansion = 1.2
        pygame.draw.rect(screen, (255, 255, 255), (projector_space_center[0] - insquare_halfside*expansion, projector_space_center[1] - insquare_halfside*expansion, 2*insquare_halfside*expansion, 2*insquare_halfside*expansion), 0)

        start_x = projector_space_center[0] - insquare_halfside
        start_y = projector_space_center[1] - insquare_halfside

        # Draw the calibration grid of N_squares x N_squares squares
        for i in range(N_squares):
            for j in range(N_squares):
                if (i + j) % 2 == 1:
                    pygame.draw.rect(screen, (0, 0, 0), (start_x + i * square_size, start_y + j * square_size, square_size, square_size), 0)
        pygame.display.flip()

        # Wait a few seconds and then take a picture
        time.sleep(1)

        # Get the screen image
        screen_image = pygame.surfarray.array3d(screen).copy().transpose((1, 0, 2))

        # Convert to grayscale using Kornia
        screen_image_tensor = torch.tensor(screen_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)  # Shape: [1, 3, H, W]
        gray_tensor = kornia.color.rgb_to_grayscale(screen_image_tensor)  # Shape: [1, 1, H, W]
        gray = gray_tensor.squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)

        # Find the chessboard corners in the screen image
        finding_corners = True
        while finding_corners:
            # Keep the pygame event loop running
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # Find the chessboard corners using OpenCV (no Kornia equivalent)
            ret, corners = cv2.findChessboardCorners(gray, (N_squares - 1, N_squares - 1), None)

            if ret:
                # Refine the corners
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                           (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                finding_corners = False
                print("Chessboard corners found.")
            else:
                print("Chessboard corners not found. Trying again.")

        # Draw the chessboard corners on the screen image using OpenCV
        screen_image_bgr = cv2.cvtColor(screen_image, cv2.COLOR_RGB2BGR)
        cv2.drawChessboardCorners(screen_image_bgr, (N_squares - 1, N_squares - 1), corners, ret)

        # Display the screen image with the chessboard corners
        cv2.namedWindow("Screen Image with Chessboard Corners", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Screen Image with Chessboard Corners", 1300, 1300)
        cv2.imshow("Screen Image with Chessboard Corners", screen_image_bgr)
        cv2.waitKey(args.display_wait_time)
        cv2.destroyWindow("Screen Image with Chessboard Corners")

        # Get the camera image as a tensor
        camera_image = cam.get_array(dont_save=True)  # Capture a frame

        # apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        camera_image = clahe.apply(camera_image)

        # Apply Otsu thresholding
        _, camera_image = cv2.threshold(camera_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Convert to tensor
        camera_image_tensor = numpy_to_tensor(camera_image)

        # Warp the camera image using H_tensor
        dst_size = (projector_height, projector_width)
        projected_camera_image_tensor = kornia.geometry.transform.warp_perspective(
            camera_image_tensor, H_tensor, dsize=dst_size)

        # Convert back to numpy array
        projected_camera_image = tensor_to_numpy(projected_camera_image_tensor)

        # Find the chessboard corners in the projected camera feed
        finding_corners = True
        while finding_corners:
            # Keep the pygame event loop running
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # Find the chessboard corners using OpenCV (no Kornia equivalent)
            ret, projected_corners = cv2.findChessboardCorners(projected_camera_image, (N_squares - 1, N_squares - 1), None)

            if ret:
                # Refine the corners
                projected_corners = cv2.cornerSubPix(projected_camera_image, projected_corners, (11, 11), (-1, -1),
                                                     (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                finding_corners = False
                print("Projected Chessboard corners found.")
            else:
                print("Projected Chessboard corners not found. Trying again.")

        # Draw the chessboard corners on the projected camera image using OpenCV
        projected_camera_image_bgr = cv2.cvtColor(projected_camera_image, cv2.COLOR_RGB2BGR)
        cv2.drawChessboardCorners(projected_camera_image_bgr, (N_squares - 1, N_squares - 1), projected_corners, ret)

        # Display the projected camera image with the chessboard corners
        cv2.namedWindow("Projected Camera Image with Chessboard Corners", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Projected Camera Image with Chessboard Corners", 1300, 1300)
        cv2.imshow("Projected Camera Image with Chessboard Corners", projected_camera_image_bgr)
        cv2.waitKey(args.display_wait_time)
        cv2.destroyWindow("Projected Camera Image with Chessboard Corners")
        

        # Transform the corners to the original image by inverting the homography
        H_inv_tensor = torch.inverse(H_tensor)

        # Convert projected_corners to tensor
        projected_corners_tensor = torch.tensor(projected_corners.reshape(-1, 2), dtype=torch.float32).unsqueeze(0).to(device)

        # Transform the corners
        transformed_corners_tensor = kornia.geometry.transform_points(H_inv_tensor, projected_corners_tensor)
        transformed_corners = transformed_corners_tensor.squeeze(0).cpu().numpy()

        # Convert corners to tensors
        corners_tensor = torch.tensor(corners.reshape(-1, 2), dtype=torch.float32).unsqueeze(0).to(device)

        # Estimate refined homography
        Hs = []
        for i in range(args.homography_iterations):
            H_refined_tensor = kornia.geometry.homography.find_homography_dlt(transformed_corners_tensor, corners_tensor)
            Hs.append(H_refined_tensor)

        # Average the homographies
        H_refined_tensor = torch.stack(Hs, dim=0).mean(dim=0)  # Shape: [1, 3, 3]
        H_refined = H_refined_tensor.squeeze(0).cpu().numpy()

        # Map the detail markers to the projector space
        interpolated_projection_space_detail_markers_tensor = torch.tensor(
            interpolated_projection_space_detail_markers, dtype=torch.float32).unsqueeze(0).to(device)  # Shape: [1, N, 2]

        # Transform points using H_refined_tensor
        distortion_corrected_projection_space_detail_markers_tensor = kornia.geometry.transform_points(
            H_refined_tensor, interpolated_projection_space_detail_markers_tensor)  # Shape: [1, N, 2]

        # Convert back to NumPy array
        distortion_corrected_projection_space_detail_markers = distortion_corrected_projection_space_detail_markers_tensor.squeeze(0).cpu().numpy()

        # Create a homography between the projector space and the screen projected projection space
        src_points_tensor = torch.tensor(distortion_corrected_projection_space_detail_markers, dtype=torch.float32).unsqueeze(0).to(device)
        dst_points_tensor = torch.tensor(interpolated_projector_space_detail_markers, dtype=torch.float32).unsqueeze(0).to(device)

        # Estimate homography
        Hs = []
        for i in range(args.homography_iterations):
            H_projector_distortion_corrected_tensor = kornia.geometry.homography.find_homography_dlt(src_points_tensor, dst_points_tensor)
            Hs.append(H_projector_distortion_corrected_tensor)

        # Average the homographies
        H_projector_distortion_corrected_tensor = torch.stack(Hs, dim=0).mean(dim=0)  # Shape: [1, 3, 3]
        H_projector_distortion_corrected = H_projector_distortion_corrected_tensor.squeeze(0).cpu().numpy()

        # Save the transformations to the rig_config
        rig_config['H_refined'] = H_refined.tolist()
        rig_config['H_projector_distortion_corrected'] = H_projector_distortion_corrected.tolist()
        rig_config['distortion_corrected_projection_space_detail_markers'] = distortion_corrected_projection_space_detail_markers.tolist()

        # Save the rig configuration
        with open(os.path.join(repo_dir, 'configs', 'rig_config_hwaccel.json'), 'w') as f:
            json.dump(rig_config, f, indent=4)

        ### TESTING THE CALIBRATION ###

        print("Testing the calibration...")

        # Drawing the uncorrected radial grid
        print("Drawing the radial grid (without distortion correction)...")

        screen.fill((0, 0, 0))

        # Draw the outward radial lines
        for i in range(0, 360, 10):
            x = int(projector_space_center[0] + projector_space_radius * np.cos(np.radians(i)))
            y = int(projector_space_center[1] + projector_space_radius * np.sin(np.radians(i)))
            pygame.draw.line(screen, (255, 255, 255), projector_space_center.astype(int), (x, y), 2)

        # Draw concentric circles
        for i in range(1, 6):
            pygame.draw.circle(screen, (255, 255, 255), projector_space_center.astype(int), int(i * projector_space_radius / 5), 2)

        # Update the display
        pygame.display.flip()

        # Get the screen image
        screen_image = pygame.surfarray.array3d(screen).copy().transpose((1, 0, 2))

        # Remap the screen image with homography using Kornia
        screen_image_tensor = numpy_to_tensor(screen_image)
        H_projector_distortion_corrected_tensor = torch.tensor(H_projector_distortion_corrected, dtype=torch.float32).unsqueeze(0).to(device)

        # Warp the screen image using homography
        screen_image_homography_tensor = kornia.geometry.transform.warp_perspective(
            screen_image_tensor, H_projector_distortion_corrected_tensor, dsize=(projector_height, projector_width), align_corners=True, mode='bilinear', padding_mode='zeros')

        # Convert back to NumPy array
        screen_image_homography = tensor_to_numpy(screen_image_homography_tensor)

        # For demonstration, we'll assume remap_image_with_interpolation remains as a function using NumPy arrays
        screen_image_distortion = remap_image_with_interpolation(screen_image, interpolated_projector_space_detail_markers,
                                                                 distortion_corrected_projection_space_detail_markers,
                                                                 (projector_height, projector_width))

        # Display the uncorrected, homography corrected, and distortion corrected images side by side
        cv2.namedWindow("Uncorrected vs Homography Corrected vs Distortion Corrected", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Uncorrected vs Homography Corrected vs Distortion Corrected", 1300, 650)
        stack = np.hstack([screen_image, screen_image_homography, screen_image_distortion])
        # Put the names on the images
        cv2.putText(stack, "Uncorrected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(stack, "Homography Corrected", (screen_image.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(stack, "Distortion Corrected", (2*screen_image.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Uncorrected vs Homography Corrected vs Distortion Corrected", stack)
        cv2.waitKey(0)
        cv2.destroyWindow("Uncorrected vs Homography Corrected vs Distortion Corrected")

        ## TEST AND SHOW THE DIFFERENCE BETWEEN THE TWO METHODS ##

        # Display the homography corrected image
        screen.fill((0, 0, 0))
        screen.blit(pygame.surfarray.make_surface(screen_image_homography.transpose((1, 0, 2))), (0, 0))
        print("Displaying the homography corrected image...")
        pygame.display.flip()

        # Take a camera image
        time.sleep(1)
        homography_corrected_camera_image = cam.get_array(dont_save=True)  # Capture a frame

        # Display the distortion corrected image
        screen.fill((0, 0, 0))
        screen.blit(pygame.surfarray.make_surface(screen_image_distortion.transpose((1, 0, 2))), (0, 0))
        print("Displaying the distortion corrected image...")
        pygame.display.flip()

        # Take a camera image
        time.sleep(1)
        distortion_corrected_camera_image = cam.get_array(dont_save=True)  # Capture a frame

        # Show the two images side by side in OpenCV
        cv2.namedWindow("Homography Corrected vs Distortion Corrected", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Homography Corrected vs Distortion Corrected", 1300, 650)
        stack = np.hstack([homography_corrected_camera_image, distortion_corrected_camera_image])
        # Put the names on the images
        cv2.putText(stack, "Homography Corrected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(stack, "Distortion Corrected", (homography_corrected_camera_image.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Homography Corrected vs Distortion Corrected", stack)
        cv2.waitKey(0)
        cv2.destroyWindow("Homography Corrected vs Distortion Corrected")

        # Ask the user which method to use
        projector_correction_method = get_predefined_answer(
            "Which method do you want to use for projector correction? [1: Homography, 2: Distortion] ",
            ['1', '2'], default='2')
        if projector_correction_method == '1':
            projector_correction_method = 'homography'
        elif projector_correction_method == '2':
            projector_correction_method = 'distortion'

        # Save the projector_correction_method
        rig_config['projector_correction_method'] = projector_correction_method

        # Save the rig configuration
        with open(os.path.join(repo_dir, 'configs', 'rig_config_hwaccel.json'), 'w') as f:
            json.dump(rig_config, f, indent=4)

        # Draw the corrected radial grid
        print("Drawing the corrected radial grid...")

        screen.fill((0, 0, 0))
        if projector_correction_method == 'homography':
            projected_image = screen_image_homography
        elif projector_correction_method == 'distortion':
            projected_image = screen_image_distortion

        screen.blit(pygame.surfarray.make_surface(projected_image.transpose((1, 0, 2))), (0, 0))
        pygame.display.flip()


        # To go from camera space (the coordinates of the image on the camera) to the distortion corrected projection space (the coordinates of the image on the screen such that the final projection is undistorted), we need to apply the following transformations:
        # 1. Get the camera image
        # 2. Apply the homography H_refined to get the image in the projection space
        # 3. Apply the projector correction method to get the image in the distortion corrected projection space


        # get the camera image
        camera_image = cam.get_array(dont_save=True)

        # improve the camera image by increasing the contrast and thresholding
        camera_image_processed = clahe.apply(camera_image)
        _, camera_image_processed = cv2.threshold(camera_image_processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


        # project the camera image onto the screen using Kornia
        H_refined_tensor = torch.tensor(H_refined, dtype=torch.float32).unsqueeze(0).to(device)
        camera_image_tensor = numpy_to_tensor(camera_image_processed)
        projected_camera_image_tensor = kornia.geometry.transform.warp_perspective(camera_image_tensor, H_refined_tensor, dsize=(projector_height, projector_width))
        projected_camera_image = tensor_to_numpy(projected_camera_image_tensor)
        

        # apply the projector correction methods
        projected_camera_image_corrected_homography_tensor = kornia.geometry.transform.warp_perspective(projected_camera_image_tensor, H_projector_distortion_corrected_tensor, dsize=(projector_height, projector_width))
        projected_camera_image_corrected_homography = tensor_to_numpy(projected_camera_image_corrected_homography_tensor)

        projected_camera_image_corrected_distortion = remap_image_with_interpolation(projected_camera_image, interpolated_projector_space_detail_markers, distortion_corrected_projection_space_detail_markers, (projector_height, projector_width))

        # make the projected camera image grayscale -> RGB for display
        projected_camera_image = cv2.cvtColor(projected_camera_image, cv2.COLOR_BGR2RGB)
        projected_camera_image_corrected_homography = cv2.cvtColor(projected_camera_image_corrected_homography, cv2.COLOR_BGR2RGB)
        projected_camera_image_corrected_distortion = cv2.cvtColor(projected_camera_image_corrected_distortion, cv2.COLOR_BGR2RGB)


        ## TEST AND SHOW THE DIFFERENCE BETWEEN THE THREE METHODS ##

        # display the uncorrected projected camera image
        screen.fill((0, 0, 0))
        screen.blit(pygame.surfarray.make_surface(projected_camera_image.transpose((1, 0, 2))), (0, 0))
        print("Displaying the uncorrected projected camera image...")
        pygame.display.flip()

        # take a camera image
        time.sleep(1)
        uncorrected_projected_camera_image = cam.get_array(dont_save=True)

        # display the homography corrected projected camera image
        screen.fill((0, 0, 0))
        screen.blit(pygame.surfarray.make_surface(projected_camera_image_corrected_homography.transpose((1, 0, 2))), (0, 0))
        print("Displaying the homography corrected projected camera image...")
        pygame.display.flip()

        # take a camera image
        time.sleep(1)
        homography_corrected_projected_camera_image = cam.get_array(dont_save=True)

        # display the distortion corrected projected camera image
        screen.fill((0, 0, 0))
        screen.blit(pygame.surfarray.make_surface(projected_camera_image_corrected_distortion.transpose((1, 0, 2))), (0, 0))
        print("Displaying the distortion corrected projected camera image...")
        pygame.display.flip()

        # take a camera image
        time.sleep(1)
        distortion_corrected_projected_camera_image = cam.get_array(dont_save=True)

        # show the three images side by side in opencv along with the difference from the camera image
        cv2.namedWindow("Uncorrected vs Homography Corrected vs Distortion Corrected", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Uncorrected vs Homography Corrected vs Distortion Corrected", 1300, 866)
        image_stack = np.hstack([uncorrected_projected_camera_image, homography_corrected_projected_camera_image, distortion_corrected_projected_camera_image])
        diff_stack = np.hstack([np.abs(uncorrected_projected_camera_image - camera_image), np.abs(homography_corrected_projected_camera_image - camera_image), np.abs(distortion_corrected_projected_camera_image - camera_image)])
        # remove noise from the difference images by morphological operations
        kernel = np.ones((5, 5), np.uint8)
        diff_stack = cv2.morphologyEx(diff_stack, cv2.MORPH_OPEN, kernel)
        # put the name on the images
        cv2.putText(image_stack, "Uncorrected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(image_stack, "Homography Corrected", (uncorrected_projected_camera_image.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(image_stack, "Distortion Corrected", (uncorrected_projected_camera_image.shape[1]*2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # stack the images and the difference images
        cv2.imshow("Uncorrected vs Homography Corrected vs Distortion Corrected", np.vstack([image_stack, diff_stack]))
        # cv2.imshow("Uncorrected vs Homography Corrected vs Distortion Corrected", np.hstack([uncorrected_projected_camera_image, homography_corrected_projected_camera_image, distortion_corrected_projected_camera_image]))
        cv2.waitKey(0)
        cv2.destroyWindow("Uncorrected vs Homography Corrected vs Distortion Corrected")

        # ask the user which method to use
        camera_correction_method = get_predefined_answer("Which method do you want to use for camera correction? [1: None, 2: Homography, 3: DISTORTION] ", ['1', '2', '3'], default='3')
        if camera_correction_method == '1':
            camera_correction_method = 'none'
        elif camera_correction_method == '2':
            camera_correction_method = 'homography'
        elif camera_correction_method == '3':
            camera_correction_method = 'distortion'

        # save the camera_correction_method
        rig_config['camera_correction_method'] = camera_correction_method

        # save the rig configuration
        with open(os.path.join(repo_dir, 'configs', 'rig_config_hwaccel.json'), 'w') as f:
            json.dump(rig_config, f, indent=4)

        # get the final corrected image
        if projector_correction_method == 'homography':
            projected_camera_image_corrected = projected_camera_image_corrected_homography
        elif projector_correction_method == 'distortion':
            projected_camera_image_corrected = projected_camera_image_corrected_distortion
        elif projector_correction_method == 'none':
            projected_camera_image_corrected = projected_camera_image


        # keep only the red channel of the projected_image and the blue channel of the projected_camera_image
        blue_channel = projected_camera_image[:, :, 2]
        red_channel = projected_image[:, :, 0]

        # mix the two images
        final = np.stack([red_channel, np.zeros_like(projected_camera_image[:, :, 1]), blue_channel], axis=-1)

        # display the final image
        screen.fill((0, 0, 0))
        screen.blit(pygame.surfarray.make_surface(final.transpose((1, 0, 2))), (0, 0))
        pygame.display.flip()

        # ask the user if the radial grid is correct
        if get_boolean_answer("Is the radial grid overlap correct? [Y/n] ", default=True):
            pass

        # Test camera to projector to camera mapping
        print("Testing camera to projector to camera mapping...")

        # clear the screen
        screen.fill((0, 0, 0))
        pygame.display.flip()

        # get the camera image
        time.sleep(0.1)
        camera_image = cam.get_array(dont_save=True)

        # draw the radial grid on the camera image using openCV
        camera_image = cv2.cvtColor(camera_image, cv2.COLOR_RGB2BGR)
        for i in range(0, 360, 10):
            camera_space_arena_center_x = int(camera_space_arena_center[0])
            camera_space_arena_center_y = int(camera_space_arena_center[1])
            x = int(camera_space_arena_center[0] + camera_space_arena_radius*np.cos(np.radians(i)))
            y = int(camera_space_arena_center[1] + camera_space_arena_radius*np.sin(np.radians(i)))
            cv2.line(camera_image, (camera_space_arena_center_x, camera_space_arena_center_y), (x, y), (255, 255, 255), 2)

        # draw concentric circles
        for i in range(1, 6):
            camera_space_arena_center_x = int(camera_space_arena_center[0])
            camera_space_arena_center_y = int(camera_space_arena_center[1])
            cv2.circle(camera_image, (camera_space_arena_center_x, camera_space_arena_center_y), int(i*camera_space_arena_radius/5), (255, 255, 255), 2)

        # display the camera image with the radial grid
        cv2.namedWindow("Camera Image with Radial Grid", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Camera Image with Radial Grid", 1300, 1300)
        cv2.imshow("Camera Image with Radial Grid", camera_image)
        cv2.waitKey(args.display_wait_time)
        cv2.destroyWindow("Camera Image with Radial Grid")

        # convert the camera image to tensor
        camera_image_tensor = numpy_to_tensor(camera_image)

        # project the camera image onto the screen
        projected_camera_image_tensor = kornia.geometry.transform.warp_perspective(camera_image_tensor, H_refined_tensor, dsize=(projector_height, projector_width))
        projected_camera_image = tensor_to_numpy(projected_camera_image_tensor)

        # apply the camera correction methods
        if camera_correction_method == 'homography':
            projected_camera_image_corrected_tensor = kornia.geometry.transform.warp_perspective(projected_camera_image_tensor, H_projector_distortion_corrected_tensor, dsize=(projector_height, projector_width))
            projected_camera_image_corrected = tensor_to_numpy(projected_camera_image_corrected_tensor)
        elif camera_correction_method == 'distortion':
            projected_camera_image_corrected = remap_image_with_interpolation(projected_camera_image, interpolated_projector_space_detail_markers, distortion_corrected_projection_space_detail_markers, (projector_height, projector_width))
        elif camera_correction_method == 'none':
            projected_camera_image_corrected = projected_camera_image

        # make the projected camera image grayscale -> RGB for display
        projected_camera_image = cv2.cvtColor(projected_camera_image, cv2.COLOR_BGR2RGB)
        projected_camera_image_corrected = cv2.cvtColor(projected_camera_image_corrected, cv2.COLOR_BGR2RGB)

        # display the projected camera image
        screen.fill((0, 0, 0))
        screen.blit(pygame.surfarray.make_surface(projected_camera_image.transpose((1, 0, 2))), (0, 0))
        pygame.display.flip()

        # take a camera image
        time.sleep(1)
        projected_camera_image = cam.get_array(dont_save=True)
        # convert for comparison
        projected_camera_image = cv2.cvtColor(projected_camera_image, cv2.COLOR_RGB2BGR)        

        # display the corrected projected camera image along with the camera image and the difference
        cv2.namedWindow("Projected Camera Image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Projected Camera Image", 1300, 866)
        diff = np.abs(projected_camera_image - camera_image)
        # apply morphological operations to remove noise
        diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel)
        image_stack = np.hstack([camera_image, projected_camera_image, diff])
        # put the name on the images
        cv2.putText(image_stack, "Camera Image", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(image_stack, "Projected Camera Image", (camera_image.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(image_stack, "Difference", (camera_image.shape[1]*2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Projected Camera Image", image_stack)
        cv2.waitKey(0)
        cv2.destroyWindow("Projected Camera Image")

        # Clean up and exit
        pygame.quit()
        sys.exit()