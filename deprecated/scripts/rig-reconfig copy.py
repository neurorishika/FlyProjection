from flyprojection.utils import hex_to_rgb, relative_to_absolute, get_boolean_answer, rect_fit
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

from flyprojection.controllers.basler_camera import BaslerCamera, list_basler_cameras
import matplotlib.pyplot as plt
import numpy as np
from skg import nsphere_fit


# # Function to continuously capture and display the camera feed
# def display_camera_feed(cam):

#     cv2.namedWindow("Projection Feed", cv2.WINDOW_NORMAL)  # Create a resizable window
#     cv2.resizeWindow("Projection Feed", 800, 800)  # Resize the window

#     while True:
#         image = cam.get_array(dont_save=True)  # Capture a frame
#         image = np.clip(image, 0, 255).astype(np.uint8)  # Format the image

#         # rotate the image 180 degrees
#         image = cv2.rotate(image, cv2.ROTATE_180)

        
#         cv2.imshow("Projection Feed", image)  # Display the feed

#         if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit if 'q' is pressed
#             break

#     cv2.destroyWindow("Projection Feed")

# def capture_images(cam, queue):
#     while True:
#         image = cam.get_array(dont_save=True)  # Capture a frame
#         image = np.clip(image, 0, 255).astype(np.uint8)  # Format the image
#         queue.put(image)  # Put the image in the queue

# Function to continuously display images from the queue in a separate process
def display_images(queue):
    cv2.namedWindow("Projection Feed", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Projection Feed", 800, 800)
    
    while True:
        if not queue.empty():
            image = queue.get()  # Get the latest image from the queue
            if image is None:  # Check for exit signal
                break
            cv2.imshow("Projection Feed", image)
            cv2.waitKey(1)

    cv2.destroyAllWindows()


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
    with BaslerCamera(index=camera_index, FPS=FPS, record_video=False, EXPOSURE_TIME=3000) as cam:
        cam.start()

        # # Start the camera image capture in a separate process
        # camera_process = multiprocessing.Process(target=capture_images, args=(cam, queue))
        # camera_process.start()

        # # Start the camera feed display in a separate thread
        # camera_thread = threading.Thread(target=display_camera_feed, args=(cam,))
        # camera_thread.daemon = True  # Ensure it exits when the main program does
        # camera_thread.start()

         # Start the OpenCV display process
        display_process = multiprocessing.Process(target=display_images, args=(queue,))
        display_process.start()

        # # create the opencv window
        # cv2.namedWindow("Projection Feed", cv2.WINDOW_NORMAL)  # Create a resizable window
        # cv2.resizeWindow("Projection Feed", 800, 800)  # Resize the window

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

                # # Check if there's a new image in the queue
                # if not queue.empty():
                #     image = queue.get()  # Get the latest image from the queue
                #     image = cv2.rotate(image, cv2.ROTATE_180)  # Rotate the image if needed
                #     cv2.imshow("Projection Feed", image)  # Display the feed
                #     cv2.resizeWindow("Projection Feed", 800, 800)
                    
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
                        collecting_points = False
                        # tell the user to press 'q' to exit
                        print("Press 'q' to exit.")
                        # join the camera thread
                        # camera_thread.join()
                        # camera_process.terminate()  # Terminate the image capture process
                        # camera_process.join()  # Wait for the process to finish
                        # # close the opencv window
                        # cv2.destroyWindow("Projection Feed")
                        # send an exit signal to the display process
                        queue.put(None)
                        display_process.join()
                        break

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
                pygame.draw.circle(screen, (255, 0, 0), point, 5)  # Draw a red circle for each stored point

            # Draw the custom cursor
            screen.blit(crosshair_cursor, (mouse_x - 5, mouse_y - 5))  # Center the cursor on the mouse position

            # Update the display
            pygame.display.flip()

        # release the cursor
        pygame.mouse.set_visible(True)

        # fit a circle to the marked points
        points = np.array(points)
        radius, center = nsphere_fit(points)
        print(f"Center: {center}, Radius: {radius}")

        # draw the circle on the screen
        screen.fill((0, 0, 0))
        pygame.draw.circle(screen, (255, 255, 255), center, int(radius), 2)
        pygame.display.flip()

        # ask the user if the circle is correct
        if get_boolean_answer("Is the circle correct? [Y/n] ", default=True):
            pass
        else:
            pygame.quit()
            sys.exit()

        # place a calibration tag on the screen just within the circle
        screen.fill((0, 0, 0))

        # calculate incircle radius
        incircle_radius = radius/np.sqrt(2)

        # get the apriltag image and place it within the circle
        april_tag = pygame.image.load(os.path.join(repo_dir, 'assets', 'apriltag.png'))
        april_tag = pygame.transform.scale(april_tag, (int(2*incircle_radius), int(2*incircle_radius)))

        # # keep only the red channel making everything else transparent
        # april_tag = pygame.surfarray.pixels3d(april_tag)
        # april_tag[:, :, 1] = 0
        # april_tag[:, :, 2] = 0
        # april_tag = pygame.surfarray.make_surface(april_tag)

        # place the apriltag on the screen
        screen.blit(april_tag, (center[0]-incircle_radius, center[1]-incircle_radius))

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

        # find an affine transformation between the two detections
        dst = np.array([point for point in camera_detection.corners])
        src = np.array([point for point in projector_detection.corners])

        # find the affine transformation
        affine_transform = cv2.estimateAffinePartial2D(src, dst)[0]

        # project the circle to the camera image
        projected_center = cv2.transform(np.array([center], dtype=np.float32).reshape(1, 1, 2), affine_transform)[0][0]
        projected_radius = cv2.transform(np.array([center + np.array([radius, 0])], dtype=np.float32).reshape(1, 1, 2), affine_transform)[0][0] - projected_center
        projected_radius = np.linalg.norm(projected_radius)

        # draw the projected circle on the camera image
        image = cv2.circle(image, tuple(projected_center.astype(int)), int(projected_radius), (255, 255, 255), 2)

        # display the camera image with the projected circle
        cv2.namedWindow("Camera Image with Projected Circle", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Camera Image with Projected Circle", 800, 800)
        cv2.imshow("Camera Image with Projected Circle", image)
        cv2.waitKey(0)
        cv2.destroyWindow("Camera Image with Projected Circle")



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






    

    




