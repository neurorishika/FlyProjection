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
    queue = multiprocessing.Queue(maxsize=1)


    # set up the argument parser
    parser = argparse.ArgumentParser(description='Open/Closed Loop Fly Projection System Rig Configuration')
    parser.add_argument('--repo_dir', type=str, default='/media/rutalab/424d5920-5085-424f-a8c7-912f65e8393c/Rishika/FlyProjection/', help='Path to the repository directory')

    # parse the arguments
    args = parser.parse_args()
    repo_dir = args.repo_dir

    # assert that there is a configs directory in the repository directory with a archived_configs subdirectory
    assert os.path.isdir(os.path.join(repo_dir, 'configs')), f"Invalid configs directory: {os.path.join(repo_dir, 'configs')}"
    assert os.path.isdir(os.path.join(repo_dir, 'configs', 'archived_configs')), f"Invalid configs directory: {os.path.join(repo_dir, 'configs', 'archived_configs')}"
    # assert that there is a rig_config.json file in the configs directory
    assert os.path.isfile(os.path.join(repo_dir, 'configs', 'rig_config.json')), f"rig_config.json file not found in {os.path.join(repo_dir, 'configs')}"

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
    screen = pygame.display.set_mode((projector_width, projector_height), pygame.NOFRAME | pygame.HWSURFACE | pygame.DOUBLEBUF)


    # take a picture
    with BaslerCamera(index=camera_index, FPS=FPS, record_video=False, EXPOSURE_TIME=4000) as cam:
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

            # Start the OpenCV display process
            display_process = multiprocessing.Process(target=display_images, args=(queue,))
            display_process.start()

            # find cursor position

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
                        
                    # Capture image from the camera
                    image = cam.get_array(dont_save=True)  # Capture a frame
                    image = np.clip(image, 0, 255).astype(np.uint8)  # Format the image
                    queue.put(image)  # Put the image in the queue

                    # Check for mouse button down event
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        # Store the current mouse position in the points list
                        points.append(pygame.mouse.get_pos())
                        print(f"Point stored: {points[-1]}")  # Print the stored point to the console
                        
                        # Exit the loop if we have 4 points
                        if len(points) >= 4:
                            # clear the screen
                            screen.fill((0, 0, 0))
                            
                            # Send exit signal to the display process
                            queue.put(None)
                            display_process.join()

                            # stop the loop
                            collecting_points = False

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

        # ask the user for the degree of detail in this step of the calibration process (4-20)
        if 'calibration_detail' in rig_config:
            default_detail = rig_config['calibration_detail']
        else:
            default_detail = 10

        detail = get_predefined_answer(f"What degree of detail do you want to use for this step of the calibration process? [3-12] (Default: {default_detail}) ", [str(i) for i in range(3, 13)], default=str(default_detail))
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

            # set the mouse callback
            cv2.setMouseCallback("Mark Arena on the Camera Image", mouse_points_callback)

            points = []

            # make a copy of the camera image
            camera_image_with_points = camera_image.copy()

            # show the camera image and mark the points
            while len(points) < 4:
                camera_image_with_points = camera_image.copy()
                for point in points:
                    # draw a crosshair at the point with the coordinates
                    cv2.line(camera_image_with_points, (point[0] - 20, point[1]), (point[0] + 20, point[1]), (255, 255, 255), 5)
                    cv2.line(camera_image_with_points, (point[0], point[1] - 20), (point[0], point[1] + 20), (255, 255, 255), 5)
                cv2.imshow("Mark Arena on the Camera Image", camera_image_with_points)
                cv2.waitKey(1)

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

            # convert the points to numpy array
            projection_space_detail_markers = np.array(points)
            projector_space_detail_markers = np.array(projector_space_detail_markers)

            # destroy the window
            cv2.destroyWindow("Mark Markers on the Camera Image")

        # save the projection_space_detail_markers and projector_space_detail_markers
        rig_config['projection_space_detail_markers'] = projection_space_detail_markers.tolist()
        rig_config['projector_space_detail_markers'] = projector_space_detail_markers.tolist()

        # save the rig configuration
        with open(os.path.join(repo_dir, 'configs', 'rig_config.json'), 'w') as f:
            json.dump(rig_config, f, indent=4)



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
            # wait for the camera to adjust
            time.sleep(1)

            # take a picture
            image = cam.get_array(dont_save=True)

            # Step 1: Convert to a suitable format
            image_array = np.clip(image, 0, 255).astype(np.uint8)

            # Step 2: Preprocess the image 
            image_array = cv2.GaussianBlur(image_array, (5, 5), 0)

            # thresholding
            _, image_array = cv2.threshold(image_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            

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

        # find the homography
        H, _ = cv2.findHomography(src, dst)

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
        time.sleep(5)
        
        camera_image = cam.get_array(dont_save=True)
        screen_image = pygame.surfarray.array3d(screen).copy().transpose((1, 0, 2))

        # get the chessboard corners using OpenCV
        gray = cv2.cvtColor(screen_image, cv2.COLOR_BGR2GRAY)

        # find the chessboard corners in the screen image
        finding_corners = True
        while finding_corners:

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
        cv2.waitKey(0)
        cv2.destroyWindow("Screen Image with Chessboard Corners")

        # project the camera image onto the screen
        projected_camera_image = cv2.warpPerspective(camera_image, H, (projector_width, projector_height))

        # find the chessboard corners in the projected camera feed
        finding_corners = True
        while finding_corners:
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

        # # draw the projected chessboard corners on the projected camera feed
        # projected_camera_image = cv2.cvtColor(projected_camera_image, cv2.COLOR_BGR2RGB)
        # cv2.drawChessboardCorners(projected_camera_image, (N_squares-1, N_squares-1), projected_corners, ret)

        # # display the camera image with the projected chessboard corners
        # cv2.namedWindow("Camera Image with Projected Chessboard Corners", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("Camera Image with Projected Chessboard Corners", 1300, 1300)
        # cv2.imshow("Camera Image with Projected Chessboard Corners", projected_camera_image)
        # cv2.waitKey(0)
        # cv2.destroyWindow("Camera Image with Projected Chessboard Corners")


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
        cv2.waitKey(0)
        cv2.destroyWindow("Camera Image with Projected Chessboard Corners")

        # refine the homography using the corners
        H_refined, _ = cv2.findHomography(projected_corners, corners) # camera space to projection space


        # map the detail markers to the projector space
        screen_projected_projection_space_detail_markers = cv2.perspectiveTransform(np.array([projection_space_detail_markers], dtype=np.float32), H_refined).reshape(-1, 2)

        # create a homography between the projector space and the screen projected projection space
        H_projector_screen_projected = cv2.findHomography(screen_projected_projection_space_detail_markers, projector_space_detail_markers)[0]

        from scipy.interpolate import griddata

        def remap_image_with_interpolation(camera_image, X, Y, image_size):
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
            mapped_points = griddata(X, Y, points_grid, method='cubic')

            # Split the mapped points back into x and y components
            mapx = mapped_points[:, 0].reshape(image_size)
            mapy = mapped_points[:, 1].reshape(image_size)

            # Ensure that mapped points are within the image boundaries
            mapx = np.clip(mapx, 0, image_size[1] - 1)
            mapy = np.clip(mapy, 0, image_size[0] - 1)

            # Remap the image
            remapped_image = cv2.remap(camera_image, mapx.astype(np.float32), mapy.astype(np.float32), interpolation=cv2.INTER_LINEAR)

            return remapped_image


        # draw a radial grid on the screen
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
        # screen_image = cv2.warpPerspective(screen_image, H_projector_screen_projected, (projector_width, projector_height))

        # remap the screen image with interpolation
        screen_image = remap_image_with_interpolation(screen_image, projector_space_detail_markers, screen_projected_projection_space_detail_markers, (projector_height, projector_width))

        # # remove the blue and green channels
        # screen_image[:, :, 0] = 0
        # screen_image[:, :, 1] = 0

        # make this new image the screen image using blit
        screen_image = pygame.surfarray.make_surface(screen_image.transpose((1, 0, 2)))
        screen.blit(screen_image, (0, 0))

        # ask the user if the user is ready to check the radial grid
        if get_boolean_answer("Are you ready to check the radial grid? [Y/n] ", default=True):
            pass

        # update the display
        pygame.display.flip()

        # wait for the user to check the radial grid
        if get_boolean_answer("Is the radial grid correct? [Y/n] ", default=True):
            pass        

        # get the current display
        projected_image = pygame.surfarray.array3d(screen).copy().transpose((1, 0, 2))


        # get the camera image
        camera_image = cam.get_array(dont_save=True)

        # project the camera image onto the screen
        projected_camera_image = cv2.warpPerspective(camera_image, H_refined, (projector_width, projector_height))

        # convert the projected camera feed to RGB
        projected_camera_image = cv2.cvtColor(projected_camera_image, cv2.COLOR_BGR2RGB)

        # draw projected camera image on the screen
        screen.fill((0, 0, 0))
        screen.blit(pygame.surfarray.make_surface(projected_camera_image.transpose((1, 0, 2))), (0, 0))
        
        # get the screen image and apply the remapping
        screen_image = pygame.surfarray.array3d(screen).copy().transpose((1, 0, 2))
        screen_image = remap_image_with_interpolation(screen_image, projector_space_detail_markers, screen_projected_projection_space_detail_markers, (projector_height, projector_width))

        # make this new image the screen image using blit
        screen_image = pygame.surfarray.make_surface(screen_image.transpose((1, 0, 2)))
        screen.blit(screen_image, (0, 0))

        # save this as the projected camera image
        projected_camera_image = pygame.surfarray.array3d(screen).copy().transpose((1, 0, 2))



        # apply the remapping to the projected camera feed
        # projected_camera_image = remap_image_with_interpolation(projected_camera_image, projection_space_detail_markers, screen_projected_projection_space_detail_markers, (projector_height, projector_width))


        # mix the projected image with the radial grid (blue channel of the projected camera feed with the red channel of the radial grid)

        # # keep only the blue channel of the projected camera feed
        # projected_camera_image[:, :, 1] = 0
        # projected_camera_image[:, :, 2] = 0


        

        # # keep only the red channel of the radial grid
        # projected_image[:, :, 0] = 0
        # projected_image[:, :, 1] = 0

        # # mix the two images
        # projected_image = projected_image + projected_camera_image

        # alternate between the two images with 0.1 interval

        try:
            while True:
                # display the projected image
                screen.fill((0, 0, 0))
                screen.blit(pygame.surfarray.make_surface(projected_image.transpose((1, 0, 2))), (0, 0))
                pygame.display.flip()
                
                # ask the user to check the radial grid
                if get_boolean_answer("Is the radial grid correct? [Y/n] ", default=False):
                    break

                # display the projected camera image
                screen.fill((0, 0, 0))
                screen.blit(pygame.surfarray.make_surface(projected_camera_image.transpose((1, 0, 2))), (0, 0))
                pygame.display.flip()
                
                # ask the user to check the radial grid
                if get_boolean_answer("Is the radial grid correct? [Y/n] ", default=False):
                    break
        except KeyboardInterrupt:
            pass


        # keep only the red channel of the projected_image and the blue channel of the projected_camera_image\
        blue_channel = projected_camera_image[:, :, 2]
        red_channel = projected_image[:, :, 0]

        # mix the two images
        final = np.stack([red_channel, np.zeros_like(projected_camera_image[:, :, 1]), blue_channel], axis=-1)

        # display the final image
        screen.fill((0, 0, 0))
        screen.blit(pygame.surfarray.make_surface(final.transpose((1, 0, 2))), (0, 0))
        pygame.display.flip()

        # ask the user if the radial grid is correct
        if get_boolean_answer("Is the radial grid correct? [Y/n] ", default=True):
            pass



        # Clean up and exit
        pygame.quit()
        sys.exit()




#     input("Press Enter to continue.")
#     pygame.quit()
#     sys.exit()

#     while True:

#         # # fill the screen with red
#         # screen.fill((255, 0, 0))
#         # pygame.display.flip()

#         # wait for the camera to adjust
#         time.sleep(2)

#         image = cam.get_array(dont_save=True)

#         # ask user mark the four corners of the projected image using matplotlib ginput
#         plt.imshow(image)
#         plt.title("Mark the four corners of the projected image\nTop left -> Top right -> Bottom right -> Bottom left")
#         plt.axis('off')
#         projection_margins = np.array(plt.ginput(4, timeout=0))
#         plt.close()
        
#         # fit a rectangle to the marked points
#         projection_margins, params_outer = rect_fit(projection_margins)

#         # ask the user to turn on the backlight
#         input("Turn on the backlight and press Enter.")

#         # display a calibration pattern
#         screen.fill((0, 0, 0))

#         # draw the calibration grid of 12x12 squares
#         for i in range(12):
#             for j in range(12):
#                 if (i+j)%2 == 0:
#                     pygame.draw.rect(screen, (255, 255, 255), (i*projector_width//12, j*projector_height//12, projector_width//12, projector_height//12), 0)
#         pygame.display.flip()

#         # fill the screen with black
#         screen.fill((0, 0, 0))
#         pygame.display.flip()

#         # wait for the camera to adjust
#         time.sleep(2)

#         # get an image from the camera
#         image = cam.get_array(dont_save=True)

#         # ask user to mark the four corners of the working area using matplotlib ginput
#         plt.imshow(image)
#         plt.title("Mark the four corners of the working area\nTop left -> Top right -> Bottom right -> Bottom left")
#         plt.axis('off')
#         working_area = np.array(plt.ginput(4, timeout=0))
#         plt.close()

#         # fit a rectangle to the marked points forcing the rotation to be the same
#         working_area, params_inner = rect_fit(working_area, force_rotation=params_outer[4])

#         # show the marked rectangles
#         plt.imshow(image)
#         for i in [1,2,3,0]:
#             plt.plot([projection_margins[i-1][0], projection_margins[i][0]], [projection_margins[i-1][1], projection_margins[i][1]], 'r')
#             plt.plot([working_area[i-1][0], working_area[i][0]], [working_area[i-1][1], working_area[i][1]], 'g')
#         plt.show()

#         print("Projecting working area, press Enter to continue.")
    
#         # convert to relative coordinates
#         rel_width = params_inner[2]/params_outer[2]
#         rel_height = params_inner[3]/params_outer[3]

#         top_corner_x = ((params_inner[0]-params_inner[2]/2) - (params_outer[0]-params_outer[2]/2))/params_outer[2]
#         top_corner_y = ((params_inner[1]-params_inner[3]/2) - (params_outer[1]-params_outer[3]/2))/params_outer[3]

#         # save the projection margins and working area
#         rig_config['projection_margins'] = projection_margins
#         rig_config['working_area'] = working_area
#         rig_config['projection_params'] = list(params_outer)
#         rig_config['working_params'] = list(params_inner)
#         rig_config['exp_region_x'] = np.clip([top_corner_x, top_corner_x + rel_width], 0, 1).tolist()
#         rig_config['exp_region_y'] = np.clip([top_corner_y, top_corner_y + rel_height], 0, 1).tolist()

#         print("Experiment region:")
#         print(f"X: {rig_config['exp_region_x']}")
#         print(f"Y: {rig_config['exp_region_y']}")

#         # show the working area
#         BOUNDARY_X = (int(rig_config['exp_region_x'][0]*projector_width), int(rig_config['exp_region_x'][1]*projector_width))
#         BOUNDARY_Y = (int(rig_config['exp_region_y'][0]*projector_height), int(rig_config['exp_region_y'][1]*projector_height))

#         screen.fill((0, 0, 0))
#         pygame.draw.rect(screen, (255, 255, 255), (BOUNDARY_X[0], BOUNDARY_Y[0], BOUNDARY_X[1]-BOUNDARY_X[0], BOUNDARY_Y[1]-BOUNDARY_Y[0]), 0)
#         pygame.display.flip()

#         # ask the user if the working area is correct
#         if get_boolean_answer("Are the working area correct? [Y/n] ", default=True):
#             break
    
#     # measure the physical dimensions of the working area
#     input("Measure the physical dimensions of the working area and press Enter.")

#     # enter the physical dimensions of the working area
#     PHYSICAL_X = float(input("Enter the physical width of the working area in inches: "))
#     PHYSICAL_Y = float(input("Enter the physical height of the working area in inches: "))
#     rig_config['physical_x'] = PHYSICAL_X
#     rig_config['physical_y'] = PHYSICAL_Y

#     # stop the camera
#     cam.stop()

# timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
# rig_config['timestamp'] = timestamp

# # save the rig configuration
# with open(os.path.join(repo_dir, 'configs', 'rig_config.json'), 'w') as f:
#     json.dump(rig_config, f, indent=4)

# # save a copy of the rig configuration with the current timestamp
# with open(os.path.join(repo_dir, 'configs', 'archived_configs', f'rig_config_{timestamp}.json'), 'w') as f:
#     json.dump(rig_config, f, indent=4)






    

    




