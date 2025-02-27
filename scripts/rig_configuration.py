import time
import threading
import sys
import shutil
import pygame
import os
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
import json
import datetime
import cv2
import argparse
import apriltag
from skg import nsphere_fit
from itertools import product
from flyprojection.utils.input import get_boolean_answer, get_predefined_answer
from flyprojection.utils.networking import validate_ip_address
from flyprojection.utils.geometry import fit_linear_curve, project_to_linear, fit_ellipse, subdivide_on_ellipse
from flyprojection.utils.transforms import generate_grid_map, remap_image_with_interpolation, remap_image_with_map
from flyprojection.controllers.basler_camera import BaslerCamera, list_basler_cameras
from flyprojection.controllers.led_server import KasaPowerController
# Import torch and kornia if available
try:
    import torch
    import kornia
    from flyprojection.utils.utils import tensor_to_numpy, numpy_to_tensor
except ImportError:
    torch = None
    kornia = None
    print("Warning: PyTorch and Kornia are not installed. 'kornia' method will not be available.")


## Set up callback function for mouse events in OpenCV
def point_selection_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point stored: {points[-1]}")
    elif event == cv2.EVENT_RBUTTONDOWN:
        if points:  # Check if points is not empty before popping
            points.pop()
            print("Point removed.")
        else:
            print("No points to remove.")


### TERMINOLOGY ###
# camera_space: the (pixel x pixel) space of the camera
# projector_space: the (pixel x pixel) space of the projector
# world_space: the (mm x mm) space of the physical world
# projection_space: the (pixel x pixel) space of the projected image on the camera


if __name__ == "__main__":

    # set up the argument parser
    parser = argparse.ArgumentParser(description='Open/Closed Loop Fly Projection System Rig Configuration')
    
    ## SETUP INPUT ARGUMENTS ##
    
    parser.add_argument('--repo_dir', type=str, default='/mnt/sda1/Rishika/FlyProjection/', help='Path to the repository directory')
    parser.add_argument('--display_wait_time', type=int, default=500, help='Time to wait before closing the display window (in ms)')
    parser.add_argument('--homography_iterations', type=int, default=10, help='Number of iterations to find the homography between the camera and projector detections')
    parser.add_argument('--interpolation_method', type=str, default='cubic', help='Interpolation method to use for the calibration process')
    parser.add_argument('--clahe_clip_limit', type=float, default=2.0, help='Clip limit for CLAHE')
    parser.add_argument('--clahe_tile_size', type=int, default=8, help='Tile size for CLAHE')
    parser.add_argument('--calibration_square_count', type=int, default=20, help='Number of squares in the calibration pattern')

    ## SETUP FLAGS ##

    parser.add_argument('--conservative_saving', dest='conservative_saving', action='store_true', help='Whether to save the rig configuration after each step of the calibration process')
    parser.add_argument('--no_conservative_saving', dest='conservative_saving', action='store_false', help='Whether to save the rig configuration after each step of the calibration process')
    parser.set_defaults(conservative_saving=True)

    parser.add_argument('--verify_map', dest='verify_map', action='store_true', help='Whether to verify the map before using it')
    parser.add_argument('--no_verify_map', dest='verify_map', action='store_false', help='Whether to verify the map before using it')
    parser.set_defaults(verify_map=False)

    parser.add_argument('--gpu_mode', dest='gpu_mode', action='store_true', help='Whether to use the GPU for the calibration process')
    parser.add_argument('--no_gpu_mode', dest='gpu_mode', action='store_false', help='Whether to use the GPU for the calibration process')
    parser.set_defaults(gpu_mode=False)

    # parse the arguments
    args = parser.parse_args()
    repo_dir = args.repo_dir

    if (torch is None or kornia is None) and args.gpu_mode:
        print("Warning: PyTorch and Kornia are not installed. GPU mode will not be available.")
        args.gpu_mode = False
    
    if args.gpu_mode:
        # Set the device to be used
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    
    # add the flag for GPU mode to the rig_config
    rig_config['gpu_mode'] = args.gpu_mode

    print("INFO: Starting the calibration process.")
    print("INFO: Please follow the instructions on the screen.\n")

    print("INFO: Setting up Controllers.\n")
    
    # ask for the IR LED IP address
    if get_boolean_answer(f"Do you want to use the default IR LED IP address ({rig_config.get('IR_LED_IP', '192.168.0.1')})? [Y/n] ", default=True):
        IR_LED_IP = rig_config.get('IR_LED_IP', '192.168.0.1')
    else:
        IR_LED_IP = input("Enter the IP address of the IR LED: ")
        # validate the IP address
        assert validate_ip_address(IR_LED_IP), f"Invalid IP address: {IR_LED_IP}"

    rig_config['IR_LED_IP'] = IR_LED_IP

    # ask for Visual LED Panel hostname and port
    if get_boolean_answer(f"Do you want to use the default Visual LED Panel hostname ({rig_config.get('visual_led_panel_hostname', 'flyprojection-server')}) and port ({rig_config.get('visual_led_panel_port', 65432)})? [Y/n] ", default=True):
        visual_led_panel_hostname = rig_config.get('visual_led_panel_hostname', 'flyprojection-server')
        visual_led_panel_port = rig_config.get('visual_led_panel_port', 65432)
    else:
        visual_led_panel_hostname = input("Enter the hostname of the Visual LED Panel: ")
        visual_led_panel_port = int(input("Enter the port of the Visual LED Panel: "))
    rig_config['visual_led_panel_hostname'] = visual_led_panel_hostname
    rig_config['visual_led_panel_port'] = visual_led_panel_port

    # ask if the width and height of the projector should be changed
    if get_boolean_answer(f"Do you want to use the current projector width and height ({rig_config.get('projector_width', 1200)}x{rig_config.get('projector_height', 800)})? [Y/n] ", default=True):
        projector_width = int(rig_config.get('projector_width', 1200))
        projector_height = int(rig_config.get('projector_height', 800))
    else:
        projector_width = int(input("Enter the projector width: "))
        projector_height = int(input("Enter the projector height: "))

    rig_config['projector_width'] = projector_width
    rig_config['projector_height'] = projector_height

    print("\nINFO: Setting up the Camera.\n")

    # list the cameras
    cameras = list_basler_cameras()
    if len(cameras) == 0:
        print("No cameras found. Exiting.")
        exit()
    else:
        print(f"Found {len(cameras)} camera(s).")
        if len(cameras) > 1:
            # ask the user to select a camera
            if get_boolean_answer(f"Do you want to use the current camera index ({rig_config.get('camera_index', 0)})? [Y/n] ", default=True):
                camera_index = int(rig_config.get('camera_index', 0))
            else:
                camera_index = int(input("Enter the index of the camera you would like to use (0-indexed and must be less than the number of cameras): "))
            assert camera_index < len(cameras), f"Invalid camera index: {camera_index}"
        else:
            camera_index = 0
    rig_config['camera_index'] = camera_index

    # ask if the camera size should be changed
    if get_boolean_answer(f"Do you want to use the current camera width and height ({rig_config.get('camera_width', 2048)}x{rig_config.get('camera_height', 2048)}) with offset ({rig_config.get('camera_offset_x', 0)},{rig_config.get('camera_offset_y', 0)})? [Y/n] ", default=True):
        camera_width = int(rig_config.get('camera_width', 2048))
        camera_height = int(rig_config.get('camera_height', 2048))
        camera_offset_x = int(rig_config.get('camera_offset_x', 0))
        camera_offset_y = int(rig_config.get('camera_offset_y', 0))
    else:
        while True:
            try:
                camera_width = int(input("Enter the camera width: "))
                camera_height = int(input("Enter the camera height: "))
                camera_offset_x = int(input("Enter the camera offset x: "))
                camera_offset_y = int(input("Enter the camera offset y: "))
                if camera_width <= 0 or camera_height <= 0 or camera_offset_x < 0 or camera_offset_y < 0:
                    raise ValueError("Width and height must be positive and offset must be non-negative.")
                break
            except ValueError as e:
                print(e)
                continue
    rig_config['camera_width'] = camera_width
    rig_config['camera_height'] = camera_height
    rig_config['camera_offset_x'] = camera_offset_x
    rig_config['camera_offset_y'] = camera_offset_y
    
    # ask if calibration and experiment exposure times should be changed
    if get_boolean_answer(f"Do you want to use the current calibration exposure time ({rig_config.get('calibration_exposure_time', 40000)})? [Y/n] ", default=True):
        calibration_exposure_time = int(rig_config.get('calibration_exposure_time', 40000))
    else:
        while True:
            try:
                calibration_exposure_time = int(input("Enter the calibration exposure time (in microseconds): "))
                if calibration_exposure_time <= 0:
                    raise ValueError("Exposure time must be positive.")
                break
            except ValueError as e:
                print(e)
                continue
    rig_config['calibration_exposure_time'] = calibration_exposure_time

    if get_boolean_answer(f"Do you want to use the current experiment exposure time ({rig_config.get('experiment_exposure_time', 9000)})? [Y/n] ", default=True):
        experiment_exposure_time = int(rig_config.get('experiment_exposure_time', 9000))
    else:
        while True:
            try:
                experiment_exposure_time = int(input("Enter the experiment exposure time (in microseconds): "))
                if experiment_exposure_time <= 0:
                    raise ValueError("Exposure time must be positive.")
                break
            except ValueError as e:
                print(e)
                continue
    rig_config['experiment_exposure_time'] = experiment_exposure_time

    # ask if the camera gain should be changed
    if get_boolean_answer(f"Do you want to use the current camera gain ({rig_config.get('camera_gain', 0.0)})? [Y/n] ", default=True):
        camera_gain = float(rig_config.get('camera_gain', 0.0))
    else:
        while True:
            try:
                camera_gain = float(input("Enter the camera gain: "))
                if camera_gain < 0:
                    raise ValueError("Gain must be non-negative.")
                break
            except ValueError as e:
                print(e)
                continue
    rig_config['camera_gain'] = camera_gain

    print("\nINFO: Setting up the Physical Arena Properties.\n")

    # ask if physical arena radius should be changed
    if get_boolean_answer(f"Do you want to use the current physical arena radius ({rig_config.get('physical_arena_radius', 75.0)})? [Y/n] ", default=True):
        physical_arena_radius = float(rig_config.get('physical_arena_radius', 75.0))
    else:
        while True:
            try:
                physical_arena_radius = float(input("Enter the physical arena radius (in mm): "))
                if physical_arena_radius <= 0:
                    raise ValueError("Radius must be positive.")
                break
            except ValueError as e:
                print(e)
                continue
    rig_config['physical_arena_radius'] = physical_arena_radius

    # print("\nINFO: Setting up the Saving Parameters.\n")

    # # ask if the saving chunk size should be changed
    # if get_boolean_answer(f"Do you want to use the current saving chunk size ({rig_config.get('saving_chunk_size', 3000)})? [Y/n] ", default=True):
    #     saving_chunk_size = int(rig_config.get('saving_chunk_size', 300))
    # else:
    #     while True:
    #         try:
    #             saving_chunk_size = int(input("Enter the saving chunk size: "))
    #             if saving_chunk_size <= 0:
    #                 raise ValueError("Chunk size must be positive.")
    #             break
    #         except ValueError as e:
    #             print(e)
    #             continue
    # rig_config['saving_chunk_size'] = saving_chunk_size

    # # ask which pre-saving processing steps should be used
    # if get_boolean_answer(f"Do you want to use the current pre-saving processing steps ({rig_config.get('pre_saving_processing_steps', 'NA')})? [Y/n] ", default=True):
    #     pre_saving_processing_steps = rig_config.get('pre_saving_processing_steps', 'NA')
    # else:
    #     pre_saving_processing_steps = get_predefined_answer("Enter the pre-saving processing steps (difference (D), background subtraction (BS), none (NA)): ", ['D', 'BS', 'NA'], default='NA')
    # rig_config['pre_saving_processing_steps'] = pre_saving_processing_steps

    # # ask for the compression level
    # if get_boolean_answer(f"Do you want to use the current compression level ({rig_config.get('compression_level', 5)})? [Y/n] ", default=True):
    #     compression_level = int(rig_config.get('compression_level', 5))
    # else:
    #     while True:
    #         try:
    #             compression_level = int(input("Enter the compression level (0-9): "))
    #             if compression_level < 0 or compression_level > 9:
    #                 raise ValueError("Compression level must be between 0 and 9.")
    #             break
    #         except ValueError as e:
    #             print(e)
    #             continue
    # rig_config['compression_level'] = compression_level


    # Setup display and initialize Pygame
    os.environ['SDL_VIDEO_WINDOW_POS'] = f"0,0"
    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((projector_width, projector_height), pygame.NOFRAME | pygame.HWSURFACE | pygame.DOUBLEBUF)

    # take a picture
    with BaslerCamera(
        index=camera_index, 
        WIDTH=camera_width,
        HEIGHT=camera_height,
        OFFSETX=camera_offset_x,
        OFFSETY=camera_offset_y,
        EXPOSURE_TIME=calibration_exposure_time,
        GAIN=camera_gain,
        record_video=False,
        TRIGGER_MODE="Continuous",
        CAMERA_FORMAT="Mono8"
        ) as cam, \
        KasaPowerController(
            ip=IR_LED_IP,
            default_state="off"
        ) as ir_led_controller:

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

                    # Check for mouse button down event
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == 1:
                            points.append(pygame.mouse.get_pos())
                            print(f"Point stored: {points[-1]}")

                        elif event.button == 3:
                            if points:
                                points.pop()
                                print("Point removed.")
                            else:
                                print("No points to remove.")
                    
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:  # Enter key
                            if len(points) < 5:
                                print("At least 5 points are required to fit a circle.")
                                continue
                            collecting_points = False
                            break # Break the event loop

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

                
                # Fit and draw the circle (INTERACTIVE PART)
                if len(points) >= 3:  # Need at least 3 points for a circle fit
                    points_np = np.array(points)
                    try:
                        projector_space_radius, projector_space_center = nsphere_fit(points_np)
                        pygame.draw.circle(screen, (255, 255, 255), projector_space_center.astype(int), int(projector_space_radius), 2)
                    except Exception as e: # Catch potential errors during fitting
                        print(f"Error during circle fitting: {e}")
                        pass # or handle in a way that is appropriate

                # Draw the custom cursor
                screen.blit(crosshair_cursor, (mouse_x - 5, mouse_y - 5))

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
            pygame.draw.circle(screen, (255, 255, 255), projector_space_center.astype(int), int(projector_space_radius), 2)
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
            if args.conservative_saving:
                print("INFO: Conservative saving is enabled. Saving the rig configuration.")
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
            pygame.draw.circle(screen, (255, 255, 255), projector_space_center.astype(int), int(projector_space_radius), 2)
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
            cv2.setMouseCallback("Mark Arena on the Camera Image", point_selection_callback)

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

        # calculate the scaling factor between camera and physical arena space and vice versa
        camera_to_physical_scaling = physical_arena_radius / camera_space_arena_radius
        physical_to_camera_scaling = camera_space_arena_radius / physical_arena_radius

        # save the scaling factors
        rig_config['camera_to_physical_scaling'] = camera_to_physical_scaling
        rig_config['physical_to_camera_scaling'] = physical_to_camera_scaling


        # save the rig configuration
        if args.conservative_saving:
            print("INFO: Conservative saving is enabled. Saving the rig configuration.")
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

            # set up the mouse callback
            points = []
            cv2.setMouseCallback("Mark Markers on the Camera Image", point_selection_callback)

            # make a copy of the camera image
            camera_image_with_points = camera_image.copy()

            # show the camera image and mark the points
            num_markers = (detail//3+1) * detail
            while len(points) < num_markers:
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
        if args.conservative_saving:
            print("INFO: Conservative saving is enabled. Saving the rig configuration.")
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
        if args.conservative_saving:
            print("INFO: Conservative saving is enabled. Saving the rig configuration.")
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
        cv2.resizeWindow("Interpolated Markers on the Camera Image", 800, 800)
        for marker in interpolated_projection_space_detail_markers:
            cv2.circle(camera_image, tuple(np.int32(marker)), 2, (255, 255, 255), -1)
        cv2.imshow("Interpolated Markers on the Camera Image", camera_image)
        cv2.waitKey(args.display_wait_time)
        cv2.destroyWindow("Interpolated Markers on the Camera Image")

        # setup CLAHE
        clahe = cv2.createCLAHE(clipLimit=args.clahe_clip_limit, tileGridSize=(args.clahe_tile_size, args.clahe_tile_size))


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
            image_array = clahe.apply(image_array)

            # thresholding
            _, image_array = cv2.threshold(image_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # show the image
            cv2.namedWindow("Camera Image", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Camera Image", 800, 800)
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
        src = np.array([point for point in camera_detection.corners]) # Shape: [4, 2]
        dst = np.array([point for point in projector_detection.corners]) # Shape: [4, 2]

        if args.gpu_mode:
            src_tensor = torch.tensor(src, dtype=torch.float32).unsqueeze(0).to(device)  # Shape: [1, 4, 2]
            dst_tensor = torch.tensor(dst, dtype=torch.float32).unsqueeze(0).to(device)  # Shape: [1, 4, 2]

        # find the homography (repeat a few times and average the results)
        Hs = []
        for i in range(args.homography_iterations):
            if args.gpu_mode:
                H = kornia.geometry.homography.find_homography_dlt(src_tensor, dst_tensor)  # Shape: [1, 3, 3]
            else:
                H, _ = cv2.findHomography(src, dst)
            Hs.append(H)
        
        if args.gpu_mode:
            # Average the homographies
            H_tensor = torch.stack(Hs, dim=0).mean(dim=0)  # Shape: [1, 3, 3]
            H = H_tensor.squeeze(0).cpu().numpy()
        else:
            H = np.mean(Hs, axis=0)


        # calculate the size of each square in the calibration pattern
        N_squares = args.calibration_square_count
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
        if args.gpu_mode:
            # Convert to grayscale using Kornia
            screen_image_tensor = torch.tensor(screen_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)  # Shape: [1, 3, H, W]
            gray_tensor = kornia.color.rgb_to_grayscale(screen_image_tensor)  # Shape: [1, 1, H, W]
            gray = gray_tensor.squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)
        else:
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
        cv2.resizeWindow("Screen Image with Chessboard Corners", 800, 800)
        cv2.imshow("Screen Image with Chessboard Corners", screen_image)
        cv2.waitKey(args.display_wait_time)
        cv2.destroyWindow("Screen Image with Chessboard Corners")

        # project the camera image onto the screen
        camera_image = cam.get_array(dont_save=True)
        # improve the camera image by increasing the contrast and thresholding
        camera_image = clahe.apply(camera_image)
        _, camera_image = cv2.threshold(camera_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if args.gpu_mode:
            # Convert to tensor
            camera_image_tensor = numpy_to_tensor(camera_image)

            # Warp the camera image using H_tensor
            dst_size = (projector_height, projector_width)
            projected_camera_image_tensor = kornia.geometry.transform.warp_perspective(
                camera_image_tensor, H_tensor, dsize=dst_size)

            # Convert back to numpy array
            projected_camera_image = tensor_to_numpy(projected_camera_image_tensor)
        else:
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
                if args.gpu_mode:
                    camera_image_tensor = numpy_to_tensor(camera_image)
                    camera_image_tensor = kornia.geometry.transform.warp_perspective(camera_image_tensor, H_tensor, dsize=dst_size)
                    camera_image = tensor_to_numpy(camera_image_tensor)
                else:
                    projected_camera_image = cv2.warpPerspective(camera_image, H, (projector_width, projector_height))


        # transform the corners to the original image by inverting the homography
        if args.gpu_mode:
            H_inv = torch.inverse(H_tensor)

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
        
        else:
            H_inv = np.linalg.inv(H)

            # transform the corners to the original image
            projected_corners = cv2.perspectiveTransform(projected_corners.reshape(-1, 1, 2), H_inv).reshape(-1, 2)

            # draw the projected corners on the camera image
            projected_camera_image = cv2.cvtColor(camera_image, cv2.COLOR_BGR2RGB)
            cv2.drawChessboardCorners(camera_image, (N_squares-1, N_squares-1), projected_corners, ret)

            # display the camera image with the projected corners
            cv2.namedWindow("Camera Image with Projected Chessboard Corners", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Camera Image with Projected Chessboard Corners", 800, 800)
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

        # Create and cache the maps for remapping for:
        # 1. H_refined from Camera to Projector
        # 2. H_projector_distortion_corrected from Projector to Distortion Corrected Projector (homography method)
        # 3. H_projector_distortion_corrected from Projector to Distortion Corrected Projector (distortion method)

        # create the maps for remapping
        # 1. H_refined from Camera to Projector

        # get every pixel in the camera image
        h, w = camera_height, camera_width
        y, x = np.mgrid[0:w, 0:h]
        yx = np.array([x.ravel(), y.ravel()]).T

        # remap the camera image with H_refined
        print("Transforming the camera coordinates to projector coordinates...")
        if args.gpu_mode:
            remapped_camera_points_tensor = kornia.geometry.transform_points(H_refined_tensor, torch.tensor(yx, dtype=torch.float32).unsqueeze(0).to(device))
            remapped_camera_points = remapped_camera_points_tensor.squeeze(0).cpu().numpy()
        else:
            remapped_camera_points = cv2.perspectiveTransform(yx.reshape(-1, 1, 2).astype(np.float32), H_refined)[:,0,:]

        # create the map for remapping
        print("Generating the map for remapping...")
        H_refined_mapx, H_refined_mapy = generate_grid_map(image_size=(projector_height, projector_width), from_points=remapped_camera_points, to_points=yx, input_size=(camera_height, camera_width), method=args.interpolation_method)

        print("Map generated.")

        # save the H_refined_map
        rig_config['H_refined_mapx'] = H_refined_mapx.tolist()
        rig_config['H_refined_mapy'] = H_refined_mapy.tolist()

        # create the inverse map for remapping
        print("Generating the inverse map for remapping...")
        H_refined_inv_mapx, H_refined_inv_mapy = generate_grid_map(image_size=(camera_height, camera_width), from_points=yx, to_points=remapped_camera_points, input_size=(projector_height, projector_width), method=args.interpolation_method)

        print("Inverse map generated.")

        # save the H_refined_inv_map
        rig_config['H_refined_inv_mapx'] = H_refined_inv_mapx.tolist()
        rig_config['H_refined_inv_mapy'] = H_refined_inv_mapy.tolist()

        # 2. H_projector_distortion_corrected from Projector to Distortion Corrected Projector (homography method)

        # get every pixel in the projector image
        h, w = projector_height, projector_width
        y, x = np.mgrid[0:h, 0:w]
        yx = np.array([x.ravel(), y.ravel()]).T

        # remap the projector image with H_projector_distortion_corrected
        print("Transforming the projector coordinates to distortion corrected projector coordinates...")
        if args.gpu_mode:
            remapped_projector_image_tensor = kornia.geometry.transform_points(H_projector_distortion_corrected_tensor, torch.tensor(yx, dtype=torch.float32).unsqueeze(0).to(device))
            remapped_projector_image = remapped_projector_image_tensor.squeeze(0).cpu().numpy()
        else:
            remapped_projector_image = cv2.perspectiveTransform(yx.reshape(-1, 1, 2).astype(np.float32), H_projector_distortion_corrected)[:,0,:]

        # create the map for remapping
        print("Generating the map for remapping...")
        H_projector_distortion_corrected_homography_mapx, H_projector_distortion_corrected_homography_mapy = generate_grid_map(image_size=(projector_height, projector_width), from_points=remapped_projector_image, to_points=yx, input_size=(projector_height, projector_width), method=args.interpolation_method)

        print("Map generated.")

        # save the H_projector_distortion_corrected_map
        rig_config['H_projector_distortion_corrected_homography_mapx'] = H_projector_distortion_corrected_homography_mapx.tolist()
        rig_config['H_projector_distortion_corrected_homography_mapy'] = H_projector_distortion_corrected_homography_mapy.tolist()

        # create the inverse map for remapping
        print("Generating the inverse map for remapping...")

        H_projector_distortion_corrected_homography_inv_mapx, H_projector_distortion_corrected_homography_inv_mapy = generate_grid_map(image_size=(projector_height, projector_width), from_points=yx, to_points=remapped_projector_image, input_size=(projector_height, projector_width), method=args.interpolation_method)

        print("Inverse map generated.")

        # save the H_projector_distortion_corrected_inv_map
        rig_config['H_projector_distortion_corrected_homography_inv_mapx'] = H_projector_distortion_corrected_homography_inv_mapx.tolist()
        rig_config['H_projector_distortion_corrected_homography_inv_mapy'] = H_projector_distortion_corrected_homography_inv_mapy.tolist()

        # 3. H_projector_distortion_corrected from Projector to Distortion Corrected Projector (distortion method)

        # use the point we have already found interpolated_projector_space_detail_markers to distortion_corrected_projection_space_detail_markers to create the map
        print("Transforming the projector coordinates to distortion corrected projector coordinates using the distortion method...")
        
        print("Generating the map for remapping...")
        H_projector_distortion_corrected_distortion_mapx, H_projector_distortion_corrected_distortion_mapy = generate_grid_map(image_size=(projector_height, projector_width), from_points=interpolated_projector_space_detail_markers, to_points=distortion_corrected_projection_space_detail_markers, input_size=(projector_height, projector_width), method=args.interpolation_method)

        print("Map generated.")
        # save the H_projector_distortion_corrected_map
        rig_config['H_projector_distortion_corrected_distortion_mapx'] = H_projector_distortion_corrected_distortion_mapx.tolist()
        rig_config['H_projector_distortion_corrected_distortion_mapy'] = H_projector_distortion_corrected_distortion_mapy.tolist()

        # create the inverse map for remapping
        print("Generating the inverse map for remapping...")

        H_projector_distortion_corrected_distortion_inv_mapx, H_projector_distortion_corrected_distortion_inv_mapy = generate_grid_map(image_size=(projector_height, projector_width), from_points=distortion_corrected_projection_space_detail_markers, to_points=interpolated_projector_space_detail_markers, input_size=(projector_height, projector_width), method=args.interpolation_method)

        print("Inverse map generated.")

        # save the H_projector_distortion_corrected_inv_map
        rig_config['H_projector_distortion_corrected_distortion_inv_mapx'] = H_projector_distortion_corrected_distortion_inv_mapx.tolist()
        rig_config['H_projector_distortion_corrected_distortion_inv_mapy'] = H_projector_distortion_corrected_distortion_inv_mapy.tolist()

        # save the rig configuration
        if args.conservative_saving:
            print("INFO: Conservative saving is enabled. Saving the rig configuration.")
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
        screen_image_homography = remap_image_with_map(screen_image, H_projector_distortion_corrected_homography_mapx, H_projector_distortion_corrected_homography_mapy)

        # compare to using warpPerspective
        if args.gpu_mode:
            screen_image_tensor = numpy_to_tensor(screen_image)
            screen_image_homography_warp_tensor = kornia.geometry.transform.warp_perspective(screen_image_tensor, H_projector_distortion_corrected_tensor, dsize=(projector_height, projector_width))
            screen_image_homography_warp = tensor_to_numpy(screen_image_homography_warp_tensor)
        else:
            screen_image_homography_warp = cv2.warpPerspective(screen_image, H_projector_distortion_corrected, (projector_width, projector_height))

        # display the homography corrected image and the warpPerspective corrected image and the difference in opencv
        if args.verify_map:
            cv2.namedWindow("Homography Corrected vs WarpPerspective Corrected", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Homography Corrected vs WarpPerspective Corrected", 1300, 650)
            diff = np.abs(screen_image_homography - screen_image_homography_warp)
            diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX) # normalize the difference
            stack = np.hstack([screen_image_homography, screen_image_homography_warp, diff])
            cv2.imshow("Homography Corrected vs WarpPerspective Corrected", stack)
            cv2.waitKey(0)
            cv2.destroyWindow("Homography Corrected vs WarpPerspective Corrected")

        # remap the screen image with interpolation
        screen_image_distortion = remap_image_with_map(screen_image, H_projector_distortion_corrected_distortion_mapx, H_projector_distortion_corrected_distortion_mapy)

        # compare to using remap_image_with_interpolation
        screen_image_distortion_interp = remap_image_with_interpolation(screen_image, interpolated_projector_space_detail_markers, distortion_corrected_projection_space_detail_markers, (projector_height, projector_width), method=args.interpolation_method)

        # display the distortion corrected image and the difference in opencv
        if args.verify_map:
            cv2.namedWindow("Distortion Corrected", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Distortion Corrected", 1300, 650)
            diff = np.abs(screen_image_distortion - screen_image_distortion_interp)
            diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX) # normalize the difference
            stack = np.hstack([screen_image_distortion, screen_image_distortion_interp, diff])
            cv2.imshow("Distortion Corrected", stack)
            cv2.waitKey(0)
            cv2.destroyWindow("Distortion Corrected")
        
        
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
        if args.conservative_saving:
            print("INFO: Conservative saving is enabled. Saving the rig configuration.")
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
        projected_camera_image = remap_image_with_map(camera_image_processed, H_refined_mapx, H_refined_mapy)

        # compare to using warpPerspective
        if args.gpu_mode:
            camera_image_tensor = numpy_to_tensor(camera_image_processed)
            projected_camera_image_warp_tensor = kornia.geometry.transform.warp_perspective(camera_image_tensor, H_refined_tensor, dsize=(projector_height, projector_width))
            projected_camera_image_warp = tensor_to_numpy(projected_camera_image_warp_tensor)
        else:
            projected_camera_image_warp = cv2.warpPerspective(camera_image_processed, H_refined, (projector_width, projector_height))

        # display the homography corrected image and the warpPerspective corrected image and the difference in opencv
        if args.verify_map:
            cv2.namedWindow("Homography Corrected vs WarpPerspective Corrected", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Homography Corrected vs WarpPerspective Corrected", 1300, 650)
            diff = np.abs(projected_camera_image - projected_camera_image_warp)
            diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX) # normalize the difference
            stack = np.hstack([projected_camera_image, projected_camera_image_warp, diff])
            cv2.imshow("Homography Corrected vs WarpPerspective Corrected", stack)
            cv2.waitKey(0)
            cv2.destroyWindow("Homography Corrected vs WarpPerspective Corrected")


        # apply the projector correction method (homography)
        projected_camera_image_corrected_homography = remap_image_with_map(projected_camera_image, H_projector_distortion_corrected_homography_mapx, H_projector_distortion_corrected_homography_mapy)
        # compare to using warpPerspective
        if args.gpu_mode:
            projected_camera_image_tensor = numpy_to_tensor(projected_camera_image)
            projected_camera_image_corrected_homography_warp_tensor = kornia.geometry.transform.warp_perspective(projected_camera_image_tensor, H_projector_distortion_corrected_tensor, dsize=(projector_height, projector_width))
            projected_camera_image_corrected_homography_warp = tensor_to_numpy(projected_camera_image_corrected_homography_warp_tensor)
        projected_camera_image_corrected_homography_warp = cv2.warpPerspective(projected_camera_image, H_projector_distortion_corrected, (projector_width, projector_height))

        # display the homography corrected image and the warpPerspective corrected image and the difference in opencv
        if args.verify_map:
            cv2.namedWindow("Homography Corrected vs WarpPerspective Corrected", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Homography Corrected vs WarpPerspective Corrected", 1300, 650)
            diff = np.abs(projected_camera_image_corrected_homography - projected_camera_image_corrected_homography_warp)
            diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX) # normalize the difference
            stack = np.hstack([projected_camera_image_corrected_homography, projected_camera_image_corrected_homography_warp, diff])
            cv2.imshow("Homography Corrected vs WarpPerspective Corrected", stack)
            cv2.waitKey(0)
            cv2.destroyWindow("Homography Corrected vs WarpPerspective Corrected")

        # apply the projector correction method (distortion)
        projected_camera_image_corrected_distortion = remap_image_with_map(projected_camera_image, H_projector_distortion_corrected_distortion_mapx, H_projector_distortion_corrected_distortion_mapy)
        # compare to using remap_image_with_interpolation
        projected_camera_image_corrected_distortion_interp = remap_image_with_interpolation(projected_camera_image, interpolated_projector_space_detail_markers, distortion_corrected_projection_space_detail_markers, (projector_height, projector_width), method=args.interpolation_method)

        # display the distortion corrected image and the difference in opencv
        if args.verify_map:
            cv2.namedWindow("Distortion Corrected", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Distortion Corrected", 1300, 650)
            diff = np.abs(projected_camera_image_corrected_distortion - projected_camera_image_corrected_distortion_interp)
            diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX) # normalize the difference
            stack = np.hstack([projected_camera_image_corrected_distortion, projected_camera_image_corrected_distortion_interp, diff])
            cv2.imshow("Distortion Corrected", stack)
            cv2.waitKey(0)
            cv2.destroyWindow("Distortion Corrected")




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
        if args.conservative_saving:
            print("INFO: Conservative saving is enabled. Saving the rig configuration.")
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
            cv2.line(camera_image, (camera_space_arena_center_x, camera_space_arena_center_y), (x, y), (255, 255, 255), 10)

        # draw concentric circles
        for i in range(1, 6):
            camera_space_arena_center_x = int(camera_space_arena_center[0])
            camera_space_arena_center_y = int(camera_space_arena_center[1])
            cv2.circle(camera_image, (camera_space_arena_center_x, camera_space_arena_center_y), int(i*camera_space_arena_radius/5), (255, 255, 255), 10)

        # display the camera image with the radial grid
        cv2.namedWindow("Camera Image with Radial Grid", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Camera Image with Radial Grid", 800, 800)
        cv2.imshow("Camera Image with Radial Grid", camera_image)
        cv2.waitKey(args.display_wait_time)
        cv2.destroyWindow("Camera Image with Radial Grid")

        # project the camera image onto the screen
        projected_camera_image = remap_image_with_map(camera_image, H_refined_mapx, H_refined_mapy)
        # apply the camera correction methods
        if camera_correction_method == 'homography':
            projected_camera_image_corrected = remap_image_with_map(projected_camera_image, H_projector_distortion_corrected_homography_mapx, H_projector_distortion_corrected_homography_mapy)
        elif camera_correction_method == 'distortion':
            projected_camera_image_corrected = remap_image_with_map(projected_camera_image, H_projector_distortion_corrected_distortion_mapx, H_projector_distortion_corrected_distortion_mapy)
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

        # apply the clahe and thresholding
        camera_image = cv2.cvtColor(camera_image, cv2.COLOR_BGR2GRAY)
        camera_image = clahe.apply(camera_image)
        _, camera_image = cv2.threshold(camera_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        projected_camera_image = clahe.apply(projected_camera_image)
        _, projected_camera_image = cv2.threshold(projected_camera_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # convert for comparison
        camera_image = cv2.cvtColor(camera_image, cv2.COLOR_RGB2BGR)
        projected_camera_image = cv2.cvtColor(projected_camera_image, cv2.COLOR_RGB2BGR)        

        # display the corrected projected camera image along with the camera image and the difference
        cv2.namedWindow("Projected Camera Image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Projected Camera Image", 1300, 866)
        diff = np.abs(projected_camera_image - camera_image)
        # apply morphological operations to remove noise
        diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel)
        # normalize the difference
        diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        image_stack = np.hstack([camera_image, projected_camera_image, diff])
        # put the name on the images
        cv2.putText(image_stack, "Camera Image", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(image_stack, "Projected Camera Image", (camera_image.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(image_stack, "Difference", (camera_image.shape[1]*2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Projected Camera Image", image_stack)
        cv2.waitKey(0)
        cv2.destroyWindow("Projected Camera Image")

        # save the rig configuration
        with open(os.path.join(repo_dir, 'configs', 'rig_config.json'), 'w') as f:
            print("INFO: Saving the rig configuration...")
            json.dump(rig_config, f, indent=4)

        # Clean up and exit
        pygame.quit()
        sys.exit()