from flyprojection.utils import *
import pygame
import os
import sys
import argparse
import datetime
import time
import json
import apriltag
import cv2
import threading
import multiprocessing
from itertools import product
from flyprojection.controllers.basler_camera import BaslerCamera, list_basler_cameras
import matplotlib.pyplot as plt
import numpy as np
from skg import nsphere_fit
import shutil


# Function to continuously display images from the queue in a separate process
def display_images(queue):
    cv2.namedWindow("Projection Feed", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Projection Feed", 1300, 1300)
    
    while True:
        if not queue.empty():
            image = queue.get()  # Get the latest image from the queue
            if image is None:  # Check for exit signal
                break
            # rotate the image 180 degrees
            image = cv2.rotate(image, cv2.ROTATE_180)
            cv2.imshow("Projection Feed", image)
            cv2.waitKey(1)

    cv2.destroyWindow("Projection Feed")


## Set up callback function for mouse events in OpenCV
def mouse_points_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point stored: {points[-1]}")
    elif event == cv2.EVENT_RBUTTONDOWN:
        points.pop()
        print("Point removed.")

### TERMINOLOGY ###
# camera_space: the (pixel x pixel) space of the camera
# projector_space: the (pixel x pixel) space of the projector
# world_space: the (mm x mm) space of the physical world
# projection_space: the (pixel x pixel) space of the projected image on the camera


if __name__ == "__main__":

    # create a multiprocessing queue
    queue = multiprocessing.Manager().Queue()


    # set up the argument parser
    parser = argparse.ArgumentParser(description='Open/Closed Loop Fly Projection System Rig Configuration')
    parser.add_argument('--repo_dir', type=str, default='/mnt/sda1/Rishika/FlyProjection/', help='Path to the repository directory')
    parser.add_argument('--display_wait_time', type=int, default=500, help='Time to wait before closing the display window (in ms)')
    parser.add_argument('--homography_iterations', type=int, default=10, help='Number of iterations to find the homography between the camera and projector detections')

    # parse the arguments
    args = parser.parse_args()
    repo_dir = args.repo_dir

    # assert that there is a configs directory in the repository directory with a archived_configs subdirectory
    assert os.path.isdir(os.path.join(repo_dir, 'configs')), f"Invalid configs directory: {os.path.join(repo_dir, 'configs')}"
    assert os.path.isdir(os.path.join(repo_dir, 'configs', 'archived_configs')), f"Invalid configs directory: {os.path.join(repo_dir, 'configs', 'archived_configs')}"
    # assert that there is a rig_config.json file in the configs directory
    assert os.path.isfile(os.path.join(repo_dir, 'configs', 'rig_config.json')), f"rig_config.json file not found in {os.path.join(repo_dir, 'configs')}"

    # backup the rig_config.json file
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.copy(os.path.join(repo_dir, 'configs', 'rig_config.json'), os.path.join(repo_dir, 'configs', 'archived_configs', f'rig_config_until_{current_time}.json'))

    # load the rig configuration
    with open(os.path.join(repo_dir, 'configs', 'rig_config.json'), 'r') as f:
        rig_config = json.load(f)

    # ask if the width and height of the projector should be changed
    if get_boolean_answer(f"Do you want to use the current projector width and height ({rig_config['projector_width']}x{rig_config['projector_height']})? [Y/n] ", default=True):
        projector_width = int(rig_config['projector_width'])
        projector_height = int(rig_config['projector_height'])
    else:
        projector_width = int(input("Enter the projector width: "))
        projector_height = int(input("Enter the projector height: "))

    rig_config['projector_width'] = projector_width
    rig_config['projector_height'] = projector_height

    # list the cameras
    cameras = list_basler_cameras()
    if len(cameras) == 0:
        print("No cameras found. Exiting.")
        exit()
    else:
        print(f"Found {len(cameras)} camera(s).")
        if len(cameras) > 1:
            # ask the user to select a camera
            if get_boolean_answer(f"Do you want to use the current camera index ({rig_config['camera_index']})? [Y/n] ", default=True):
                camera_index = int(rig_config['camera_index'])
            else:
                camera_index = int(input("Enter the index of the camera you would like to use (0-indexed and must be less than the number of cameras): "))
            assert camera_index < len(cameras), f"Invalid camera index: {camera_index}"
        else:
            camera_index = 0
    rig_config['camera_index'] = camera_index

    # setup FPS
    if get_boolean_answer(f"Do you want to use the current FPS ({rig_config['FPS']})? [Y/n] ", default=True):
        FPS = int(rig_config['FPS'])
    else:
        FPS = int(input("Enter the FPS: "))
    rig_config['FPS'] = FPS

    # Setup display and initialize Pygame
    os.environ['SDL_VIDEO_WINDOW_POS'] = f"0,0"
    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((projector_width, projector_height), pygame.NOFRAME | pygame.HWSURFACE | pygame.DOUBLEBUF)


    # take a picture
    with BaslerCamera(index=camera_index, FPS=FPS, record_video=False, EXPOSURE_TIME=2000) as cam:
        cam.start()

        #### CALIBRATION STEP 1: FIND THE ARENA IN THE PROJECTOR SPACE ####

        # check if projector_space_arena_center and projector_space_arena_radius are present in the rig_config
        if 'projector_space_arena_center' in rig_config and 'projector_space_arena_radius' in rig_config:
            # ask the user if the current projector_space_arena_center and projector_space_arena_radius should be used
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

            # ask user if live feed should be displayed (might cause lag)
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
                                # clear the screen
                                screen.fill((0, 0, 0))
                                
                                if display_live_feed:
                                    # Send exit signal to the display process
                                    queue.put(None)
                                    display_process.join()

                                # stop the loop
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

            # release the cursor
            pygame.mouse.set_visible(True)

            # fit a circle to the marked points
            points = np.array(points)
            projector_space_radius, projector_space_center = nsphere_fit(points)
            print(f"Center: {projector_space_center}, Radius: {projector_space_radius}")

            # draw the circle on the screen
            screen.fill((0, 0, 0))
            pygame.draw.circle(screen, (255, 255, 255), projector_space_center, int(projector_space_radius), 2)
            pygame.display.flip()

            # ask the user if the circle is correct
            if get_boolean_answer("Is the circle correct? [Y/n] ", default=True):
                pass
            else:
                pygame.quit()
                sys.exit()

            # save the projector_space_arena_center and projector_space_arena_radius
            rig_config['projector_space_arena_center'] = projector_space_center.tolist()
            rig_config['projector_space_arena_radius'] = projector_space_radius

            # save the rig configuration
            with open(os.path.join(repo_dir, 'configs', 'rig_config.json'), 'w') as f:
                json.dump(rig_config, f, indent=4)

        projector_space_radius = rig_config['projector_space_arena_radius']
        projector_space_center = np.array(rig_config['projector_space_arena_center'])


        #### CALIBRATION STEP 2: FIND THE DISTORTION IN THE PROJECTION ####


        # check if camera_space_arena_center and camera_space_arena_radius are present in the rig_config
        if 'camera_space_arena_center' in rig_config and 'camera_space_arena_radius' in rig_config:
            # ask the user if the current camera_space_arena_center and camera_space_arena_radius should be used
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
            # draw the arena on the screen
            screen.fill((0, 0, 0))
            pygame.draw.circle(screen, (255, 255, 255), projector_space_center, int(projector_space_radius), 2)
            pygame.display.flip()

            # wait a few seconds and then take a picture
            time.sleep(2)

            # get a camera image
            camera_image = cam.get_array(dont_save=True)

            # display the camera image and mark points on it
            cv2.namedWindow("Mark Arena on the Camera Image", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Mark Arena on the Camera Image", 1300, 1300)

            # set up the mouse callback
            points = []

            # remove any existing mouse callback
            cv2.setMouseCallback("Mark Arena on the Camera Image", lambda *args : None)

            # set the mouse callback
            cv2.setMouseCallback("Mark Arena on the Camera Image", mouse_points_callback)

            # make a copy of the camera image
            camera_image_with_points = camera_image.copy()

            # show the camera image and mark the points
            while True:
                # keep the pygame event loop running
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

                camera_image_with_points = camera_image.copy()
                for point in points:
                    # draw a crosshair at the point with the coordinates
                    cv2.line(camera_image_with_points, (point[0] - 20, point[1]), (point[0] + 20, point[1]), (255, 255, 255), 5)
                    cv2.line(camera_image_with_points, (point[0], point[1] - 20), (point[0], point[1] + 20), (255, 255, 255), 5)

                # fit and draw a circle to the points if there are at least 4 points
                if len(points) >= 4:
                    camera_space_arena_radius, camera_space_arena_center = nsphere_fit(np.array(points))
                    cv2.circle(camera_image_with_points, tuple(camera_space_arena_center.astype(int)), int(camera_space_arena_radius), (255, 255, 255), 2)
                
                # write instructions on the image
                cv2.putText(camera_image_with_points, "Click on points on the circle. Press 'q' and Alt+Tab back to VSCode", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                cv2.imshow("Mark Arena on the Camera Image", camera_image_with_points)

                # break if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q') and len(points) >= 4:
                    break

            # stop the mouse callback
            cv2.setMouseCallback("Mark Arena on the Camera Image", lambda *args : None)

            # convert the points to numpy array
            points = np.array(points)

            # fit a circle to the points
            camera_space_arena_radius, camera_space_arena_center = nsphere_fit(points)

            # draw the circle on the camera image
            cv2.circle(camera_image_with_points, tuple(camera_space_arena_center.astype(int)), int(camera_space_arena_radius), (255, 255, 255), 2)

            # display the camera image with the circle
            cv2.imshow("Mark Arena on the Camera Image", camera_image_with_points)
            cv2.waitKey(0)
            cv2.destroyWindow("Mark Arena on the Camera Image")

        # save the camera_space_arena_center and camera_space_arena_radius
        rig_config['camera_space_arena_center'] = camera_space_arena_center.tolist()
        rig_config['camera_space_arena_radius'] = camera_space_arena_radius

        # save the rig configuration
        with open(os.path.join(repo_dir, 'configs', 'rig_config.json'), 'w') as f:
            json.dump(rig_config, f, indent=4)


        # ask the user for the degree of detail in this step of the calibration process (4-20)
        if 'calibration_detail' in rig_config:
            default_detail = rig_config['calibration_detail']
        else:
            default_detail = 12

        detail = get_predefined_answer(f"Calibration Degree of Detail [3-12] (Default: {default_detail}) ", [str(i) for i in range(3, 13)], default=str(default_detail))
        detail = int(detail)

        flag_for_calibration = False
        if default_detail != detail:
            flag_for_calibration = True

        # define detail radial grid points in a uni
        radial = np.linspace(0, 1, detail//3+2)[1:]
        angular = np.linspace(0, 2*np.pi, detail+1)[:-1]

        detail_markers = list(product(radial, angular))

        # save the degree of detail in the rig_config
        rig_config['calibration_detail'] = detail

        # save the rig configuration
        with open(os.path.join(repo_dir, 'configs', 'rig_config.json'), 'w') as f:
            json.dump(rig_config, f, indent=4)


        # check if projection_space_detail_markers and projector_space_detail_markers are present in the rig_config and flag_for_calibration is False
        if 'projection_space_detail_markers' in rig_config and 'projector_space_detail_markers' in rig_config and not flag_for_calibration:
            # ask the user if the current projection_space_detail_markers and projector_space_detail_markers should be used
            if get_boolean_answer(f"Do you want to use the current projection_space_detail_markers ({rig_config['projection_space_detail_markers']}) and projector_space_detail_markers ({rig_config['projector_space_detail_markers']})? [Y/n] ", default=True):
                projection_space_detail_markers = np.array(rig_config['projection_space_detail_markers'])
                projector_space_detail_markers = np.array(rig_config['projector_space_detail_markers'])
            else:
                projection_space_detail_markers = None
                projector_space_detail_markers = None
        else:
            projection_space_detail_markers = None
            projector_space_detail_markers = None

        if projection_space_detail_markers is None or projector_space_detail_markers is None:
            # find the corresponding points in the projector space
            projector_space_detail_markers = [(projector_space_center[0] + marker[0]*projector_space_radius*np.cos(marker[1]), projector_space_center[1] + marker[0]*projector_space_radius*np.sin(marker[1])) for marker in detail_markers]

            # draw the markers on the screen
            screen.fill((0, 0, 0))
            font = pygame.font.Font(None, 20)
            for marker in projector_space_detail_markers:
                pygame.draw.circle(screen, (255, 255, 255), (int(marker[0]), int(marker[1])), 2)
                # put the number of the marker next to it
                text = font.render(str(projector_space_detail_markers.index(marker)), True, (255, 255, 255))
                screen.blit(text, (int(marker[0]) + 5, int(marker[1]) + 5))
            pygame.display.flip()

            # wait a few seconds and then take a picture
            time.sleep(2)

            # get a camera image
            camera_image = cam.get_array(dont_save=True)

            # display the camera image and mark points on it
            cv2.namedWindow("Mark Markers on the Camera Image", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Mark Markers on the Camera Image", 1300, 1300)

            cv2.setMouseCallback("Mark Markers on the Camera Image", mouse_points_callback)

            points = []

            # make a copy of the camera image
            camera_image_with_points = camera_image.copy()

            # show the camera image and mark the points
            while len(points) < (detail//3+1) * detail:
                camera_image_with_points = camera_image.copy()
                for point in points:
                    # draw a crosshair at the point with the coordinates
                    cv2.line(camera_image_with_points, (point[0] - 20, point[1]), (point[0] + 20, point[1]), (255, 255, 255), 5)
                    cv2.line(camera_image_with_points, (point[0], point[1] - 20), (point[0], point[1] + 20), (255, 255, 255), 5)
                cv2.imshow("Mark Markers on the Camera Image", camera_image_with_points)
                cv2.waitKey(1)

            # stop the mouse callback
            cv2.setMouseCallback("Mark Markers on the Camera Image", lambda *args : None)

            # destroy the window
            cv2.destroyWindow("Mark Markers on the Camera Image")

            # convert the points to numpy array
            projection_space_detail_markers = np.array(points)
            projector_space_detail_markers = np.array(projector_space_detail_markers)

        # save the projection_space_detail_markers and projector_space_detail_markers
        rig_config['projection_space_detail_markers'] = projection_space_detail_markers.tolist()
        rig_config['projector_space_detail_markers'] = projector_space_detail_markers.tolist()

        # save the rig configuration
        with open(os.path.join(repo_dir, 'configs', 'rig_config.json'), 'w') as f:
            json.dump(rig_config, f, indent=4)

        # get a camera image
        camera_image = cam.get_array(dont_save=True)

        fitted_projection_space_detail_markers = np.zeros_like(projection_space_detail_markers)
        fitted_projector_space_detail_markers = np.zeros_like(projector_space_detail_markers)


        # for each outward radial line, fit a quadratic curve to the points and project onto the curve
        for i in range(detail):
            # get the points for the radial line
            projection_space_radial_line_points = projection_space_detail_markers[i::detail]

            # fit a quadratic curve to the points
            projection_space_radial_line_params = fit_linear_curve(projection_space_radial_line_points[:, 0], projection_space_radial_line_points[:, 1])

            # get projected points on the curve
            projected_points = project_to_linear(projection_space_radial_line_points, projection_space_radial_line_params)

            # add the points to the fitted_projection_space_detail_markers
            fitted_projection_space_detail_markers[i::detail] = projected_points[:,:]

            # get the points for the radial line
            projector_space_radial_line_points = projector_space_detail_markers[i::detail]

            # fit a quadratic curve to the points
            projector_space_radial_line_params = fit_linear_curve(projector_space_radial_line_points[:, 0], projector_space_radial_line_points[:, 1])

            # get projected points on the curve
            projected_points = project_to_linear(projector_space_radial_line_points, projector_space_radial_line_params)

            # add the points to the fitted_projector_space_detail_markers
            fitted_projector_space_detail_markers[i::detail] = projected_points[:,:]

        # save the fitted_projection_space_detail_markers and fitted_projector_space_detail_markers
        rig_config['fitted_projection_space_detail_markers'] = fitted_projection_space_detail_markers.tolist()
        rig_config['fitted_projector_space_detail_markers'] = fitted_projector_space_detail_markers.tolist()

        # save the rig configuration
        with open(os.path.join(repo_dir, 'configs', 'rig_config.json'), 'w') as f:
            json.dump(rig_config, f, indent=4)


        interpolated_projection_space_detail_markers = []
        interpolated_projector_space_detail_markers = []

        # for each concentric ellipse, fit a ellipse to the points and interpolate the points on the ellipse to get 100 points per ellipse
        for i in range(detail//3+1):
            # get the points for the ellipse
            projection_space_ellipse_points = projection_space_detail_markers[i*detail:(i+1)*detail]

            # fit an ellipse to the points
            projection_space_ellipse_params = fit_ellipse(projection_space_ellipse_points[:, 0], projection_space_ellipse_points[:, 1])
            
            # interpolate the points on the ellipse
            projection_space_ellipse_points = subdivide_on_ellipse(projection_space_ellipse_points, projection_space_ellipse_params, 10)

            # add the points to the interpolated_projection_space_detail_markers
            interpolated_projection_space_detail_markers.extend(projection_space_ellipse_points)

            # get the points for the ellipse
            projector_space_ellipse_points = projector_space_detail_markers[i*detail:(i+1)*detail]

            # fit an ellipse to the points
            projector_space_ellipse_params = fit_ellipse(projector_space_ellipse_points[:, 0], projector_space_ellipse_points[:, 1])

            # interpolate the points on the ellipse
            projector_space_ellipse_points = subdivide_on_ellipse(projector_space_ellipse_points, projector_space_ellipse_params, 10)

            # add the points to the interpolated_projector_space_detail_markers
            interpolated_projector_space_detail_markers.extend(projector_space_ellipse_points)

        # convert the lists to numpy arrays
        interpolated_projection_space_detail_markers = np.array(interpolated_projection_space_detail_markers)
        interpolated_projector_space_detail_markers = np.array(interpolated_projector_space_detail_markers)

        # save the interpolated_projection_space_detail_markers and interpolated_projector_space_detail_markers
        rig_config['interpolated_projection_space_detail_markers'] = interpolated_projection_space_detail_markers.tolist()
        rig_config['interpolated_projector_space_detail_markers'] = interpolated_projector_space_detail_markers.tolist()


        # draw the markers on the screen
        cv2.namedWindow("Interpolated Markers on the Camera Image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Interpolated Markers on the Camera Image", 1300, 1300)
        for marker in interpolated_projection_space_detail_markers:
            cv2.circle(camera_image, tuple(np.int32(marker)), 2, (255, 255, 255), -1)
        cv2.imshow("Interpolated Markers on the Camera Image", camera_image)
        cv2.waitKey(args.display_wait_time)
        cv2.destroyWindow("Interpolated Markers on the Camera Image")



        # place a calibration tag on the screen just within the circle
        screen.fill((0, 0, 0))

        # calculate incircle projector_space_radius
        insquare_halfside = projector_space_radius/np.sqrt(2)

        # get the apriltag image and place it within the circle
        april_tag = pygame.image.load(os.path.join(repo_dir, 'assets', 'apriltag.png'))
        april_tag = pygame.transform.scale(april_tag, (int(2*insquare_halfside), int(2*insquare_halfside)))

        # place the apriltag on the screen
        screen.blit(april_tag, (projector_space_center[0]-insquare_halfside, projector_space_center[1]-insquare_halfside))

        # update the display
        pygame.display.flip()

        # Find Apriltag in camera image

        finding_apriltag = True
        while finding_apriltag:

            # keep the pygame event loop running
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # wait for the camera to adjust
            time.sleep(0.1)

            # take a picture
            image = cam.get_array(dont_save=True)

            # Step 1: Convert to a suitable format
            image_array = np.clip(image, 0, 255).astype(np.uint8)

            # Step 2: Preprocess the image 
            # increase contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            image_array = clahe.apply(image_array)

            # thresholding
            _, image_array = cv2.threshold(image_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # show the image
            cv2.imshow("Camera Image", image_array)
            cv2.waitKey(args.display_wait_time)
            cv2.destroyWindow("Camera Image")
            

            # Step 3: Initialize the AprilTag detector
            detector = apriltag.Detector()

            # Step 4: Detect AprilTags
            detections = detector.detect(image_array)

            # Step 5: Check if any AprilTag is detected
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

            # keep the pygame event loop running
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # get the screen as an array
            projected_image = pygame.surfarray.array3d(screen).copy().transpose((1, 0, 2))

            # Step 1: Convert to a suitable format
            projected_image = np.clip(projected_image, 0, 255).astype(np.uint8)

            # Convert to grayscale
            projected_image = cv2.cvtColor(projected_image, cv2.COLOR_BGR2GRAY)

            # Step 3: Initialize the AprilTag detector
            detector = apriltag.Detector()

            # Step 4: Detect AprilTags
            detections = detector.detect(projected_image)

            # Step 5: Check if any AprilTag is detected
            if len(detections) == 0:
                print("Apriltag not detected. Conducting the process again.")
                continue
            else:
                finding_apriltag = False
                assert len(detections) == 1, f"Multiple AprilTags detected: {len(detections)}"
                projector_detection = detections[0]
                print(f"Apriltag detected in projected image. ID: {projector_detection.tag_id}")

        # find an transformation between the two detections
        src = np.array([point for point in camera_detection.corners])
        dst = np.array([point for point in projector_detection.corners])

        # find the homography (repeat a few times and average the results)
        Hs = []
        for i in range(args.homography_iterations):
            H, _ = cv2.findHomography(src, dst)
            Hs.append(H)
        H = np.mean(Hs, axis=0)


        # calculate the size of each square in the calibration pattern
        N_squares = 13
        square_size = 2*insquare_halfside/N_squares

        # draw the calibration pattern on the screen
        screen.fill((0, 0, 0))

        # add a white border around the calibration pattern
        expansion = 1.2
        pygame.draw.rect(screen, (255, 255, 255), (projector_space_center[0] - insquare_halfside*expansion, projector_space_center[1] - insquare_halfside*expansion, 2*insquare_halfside*expansion, 2*insquare_halfside*expansion), 0)

        start_x = projector_space_center[0] - insquare_halfside
        start_y = projector_space_center[1] - insquare_halfside

        # draw the calibration grid of N_squares x N_squares squares
        for i in range(N_squares):
            for j in range(N_squares):
                if (i+j)%2 == 1:
                    pygame.draw.rect(screen, (0, 0, 0), (start_x + i*square_size, start_y + j*square_size, square_size, square_size), 0)
        pygame.display.flip()

        # wait a few seconds and then take a picture
        time.sleep(1)
        
        screen_image = pygame.surfarray.array3d(screen).copy().transpose((1, 0, 2))

        # get the chessboard corners using OpenCV
        gray = cv2.cvtColor(screen_image, cv2.COLOR_BGR2GRAY)

        # find the chessboard corners in the screen image
        finding_corners = True
        while finding_corners:
            # keep the pygame event loop running
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (N_squares-1, N_squares-1), None)

            if ret:
                # refine the corners
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                finding_corners = False
                print("Chessboard corners found.")
            else:
                print("Chessboard corners not found. Trying again.")

        # draw the chessboard corners on the screen image using openCV
        screen_image = cv2.cvtColor(screen_image, cv2.COLOR_RGB2BGR)
        cv2.drawChessboardCorners(screen_image, (N_squares-1, N_squares-1), corners, ret)

        # display the screen image with the chessboard corners
        cv2.namedWindow("Screen Image with Chessboard Corners", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Screen Image with Chessboard Corners", 1300, 1300)
        cv2.imshow("Screen Image with Chessboard Corners", screen_image)
        cv2.waitKey(args.display_wait_time)
        cv2.destroyWindow("Screen Image with Chessboard Corners")

        # project the camera image onto the screen
        camera_image = cam.get_array(dont_save=True)
        # improve the camera image by increasing the contrast and thresholding
        camera_image = clahe.apply(camera_image)
        _, camera_image = cv2.threshold(camera_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        projected_camera_image = cv2.warpPerspective(camera_image, H, (projector_width, projector_height))

        # find the chessboard corners in the projected camera feed
        finding_corners = True
        while finding_corners:
            # keep the pygame event loop running
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # find the chessboard corners
            ret, projected_corners = cv2.findChessboardCorners(projected_camera_image, (N_squares-1, N_squares-1), None)

            if ret:
                # refine the corners
                projected_corners = cv2.cornerSubPix(projected_camera_image, projected_corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                finding_corners = False
                print("Projected Chessboard corners found.")
            else:
                print("Projected Chessboard corners not found. Trying again.")
                camera_image = cam.get_array(dont_save=True)
                projected_camera_image = cv2.warpPerspective(camera_image, H, (projector_width, projector_height))


        # transform the corners to the original image by inverting the homography
        H_inv = np.linalg.inv(H)

        # transform the corners to the original image
        projected_corners = cv2.perspectiveTransform(projected_corners.reshape(-1, 1, 2), H_inv).reshape(-1, 2)

        # draw the projected corners on the camera image
        projected_camera_image = cv2.cvtColor(camera_image, cv2.COLOR_BGR2RGB)
        cv2.drawChessboardCorners(camera_image, (N_squares-1, N_squares-1), projected_corners, ret)

        # display the camera image with the projected corners
        cv2.namedWindow("Camera Image with Projected Chessboard Corners", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Camera Image with Projected Chessboard Corners", 1300, 1300)
        cv2.imshow("Camera Image with Projected Chessboard Corners", camera_image)
        cv2.waitKey(args.display_wait_time)
        cv2.destroyWindow("Camera Image with Projected Chessboard Corners")

        # refine the homography using the corners (repeat a few times and average the results)
        Hs = []
        for i in range(args.homography_iterations):
            H, _ = cv2.findHomography(projected_corners, corners)
            Hs.append(H)
        H_refined = np.mean(Hs, axis=0)

        # map the detail markers to the projector space
        distortion_corrected_projection_space_detail_markers = cv2.perspectiveTransform(np.array([interpolated_projection_space_detail_markers], dtype=np.float32), H_refined).reshape(-1, 2)

        # create a homography between the projector space and the screen projected projection space
        H_projector_distortion_corrected = cv2.findHomography(distortion_corrected_projection_space_detail_markers, interpolated_projector_space_detail_markers)[0]

        # Save the transformations to the rig_config
        rig_config['H_refined'] = H_refined.tolist()
        rig_config['H_projector_distortion_corrected'] = H_projector_distortion_corrected.tolist()
        rig_config['distortion_corrected_projection_space_detail_markers'] = distortion_corrected_projection_space_detail_markers.tolist()

        # save the rig configuration
        with open(os.path.join(repo_dir, 'configs', 'rig_config.json'), 'w') as f:
            json.dump(rig_config, f, indent=4)

        ### TESTING THE CALIBRATION ###

        print("Testing the calibration...")

        # To go from projector space (the coordinates of the image on the screen) to the distortion corrected projection space (the coordinates of the image on the screen such that the final projection is undistorted), we need to apply the following transformations:
        # 1. Draw the image we want to project on the screen (projector space)
        # 2. If we want a simple homography, we can apply H_projector_distortion_corrected to get the image in the distortion corrected projection space
        # 3. If we want to apply the distortion correction, we can apply the remap_image_with_interpolation function to get the image in the distortion corrected projection space


        # drawing the uncorrected radial grid
        print("Drawing the radial grid (without distortion correction)...")

        screen.fill((0, 0, 0))

        # draw the outward radial lines
        for i in range(0, 360, 10):
            x = int(projector_space_center[0] + projector_space_radius*np.cos(np.radians(i)))
            y = int(projector_space_center[1] + projector_space_radius*np.sin(np.radians(i)))
            pygame.draw.line(screen, (255, 255, 255), projector_space_center, (x, y), 2)

        # draw concentric circles
        for i in range(1, 6):
            pygame.draw.circle(screen, (255, 255, 255), projector_space_center, int(i*projector_space_radius/5), 2)

        # update the display
        pygame.display.flip()

        # get the screen image and apply the homography
        screen_image = pygame.surfarray.array3d(screen).copy().transpose((1, 0, 2))

        # remap the screen image with homography
        screen_image_homography = cv2.warpPerspective(screen_image, H_projector_distortion_corrected, (projector_width, projector_height))

        # remap the screen image with interpolation
        screen_image_distortion = remap_image_with_interpolation(screen_image, interpolated_projector_space_detail_markers, distortion_corrected_projection_space_detail_markers, (projector_height, projector_width))

        ## TEST AND SHOW THE DIFFERENCE BETWEEN THE TWO METHODS ##

        # display the homography corrected image
        screen.fill((0, 0, 0))
        screen.blit(pygame.surfarray.make_surface(screen_image_homography.transpose((1, 0, 2))), (0, 0))
        print("Displaying the homography corrected image...")
        pygame.display.flip()

        # take a camera image
        time.sleep(0.1)
        homography_corrected_camera_image = cam.get_array(dont_save=True)

        # display the distortion corrected image
        screen.fill((0, 0, 0))
        screen.blit(pygame.surfarray.make_surface(screen_image_distortion.transpose((1, 0, 2))), (0, 0))
        print("Displaying the distortion corrected image...")
        pygame.display.flip()

        # take a camera image
        time.sleep(0.1)
        distortion_corrected_camera_image = cam.get_array(dont_save=True)

        # show the two images side by side in opencv
        cv2.namedWindow("Homography Corrected vs Distortion Corrected", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Homography Corrected vs Distortion Corrected", 1300, 650)
        stack = np.hstack([homography_corrected_camera_image, distortion_corrected_camera_image])
        # put the names on the images
        cv2.putText(homography_corrected_camera_image, "Homography Corrected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(distortion_corrected_camera_image, "Distortion Corrected", (homography_corrected_camera_image.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Homography Corrected vs Distortion Corrected", stack)
        cv2.waitKey(0)
        cv2.destroyWindow("Homography Corrected vs Distortion Corrected")

        # ask the user which method to use
        projector_correction_method = get_predefined_answer("Which method do you want to use for projector correction? [1: Homography, 2: DISTORTION] ", ['1', '2'], default='2')
        if projector_correction_method == '1':
            projector_correction_method = 'homography'
        elif projector_correction_method == '2':
            projector_correction_method = 'distortion'

        # save the projector_correction_method
        rig_config['projector_correction_method'] = projector_correction_method

        # save the rig configuration
        with open(os.path.join(repo_dir, 'configs', 'rig_config.json'), 'w') as f:
            json.dump(rig_config, f, indent=4)

        # draw the corrected radial grid
        print("Drawing the corrected radial grid...")

        screen.fill((0, 0, 0))
        if projector_correction_method == 'homography':
            projected_image = screen_image_homography
        elif projector_correction_method == 'distortion':
            projected_image = screen_image_distortion

        screen.blit(pygame.surfarray.make_surface(projected_image.transpose((1, 0, 2))), (0, 0))

        # To go from camera space (the coordinates of the image on the camera) to the distortion corrected projection space (the coordinates of the image on the screen such that the final projection is undistorted), we need to apply the following transformations:
        # 1. Get the camera image
        # 2. Apply the homography H_refined to get the image in the projection space
        # 3. Apply the projector correction method to get the image in the distortion corrected projection space


        # get the camera image
        camera_image = cam.get_array(dont_save=True)

        # improve the camera image by increasing the contrast and thresholding
        camera_image_processed = clahe.apply(camera_image)
        _, camera_image_processed = cv2.threshold(camera_image_processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


        # project the camera image onto the screen
        projected_camera_image = cv2.warpPerspective(camera_image_processed, H_refined, (projector_width, projector_height))

        # apply the projector correction methods
        projected_camera_image_corrected_homography = cv2.warpPerspective(projected_camera_image, H_projector_distortion_corrected, (projector_width, projector_height))
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
        with open(os.path.join(repo_dir, 'configs', 'rig_config.json'), 'w') as f:
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

        # project the camera image onto the screen
        projected_camera_image = cv2.warpPerspective(camera_image, H_refined, (projector_width, projector_height))
        # apply the camera correction methods
        if camera_correction_method == 'homography':
            projected_camera_image_corrected = cv2.warpPerspective(projected_camera_image, H_projector_distortion_corrected, (projector_width, projector_height))
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






    

    




