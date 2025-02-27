import pygame
import sys
import os

# Set the window position to the top-left corner
os.environ['SDL_VIDEO_WINDOW_POS'] = "0,0"

# Initialize Pygame
pygame.init()

# Set up the display
projector_width = 1200  # Adjust to your projector's resolution
projector_height = 800  # Adjust to your projector's resolution
screen = pygame.display.set_mode((projector_width, projector_height), pygame.NOFRAME | pygame.HWSURFACE | pygame.DOUBLEBUF)
pygame.display.set_caption("Grid Pattern for Projector Focus")

# Define grid parameters
grid_color = (255, 255, 255)  # White color
background_color = (0, 0, 0)  # Black background
grid_spacing = 50  # Space between grid lines in pixels

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Clear the screen with the background color
    screen.fill(background_color)

    # Draw vertical grid lines
    for x in range(0, projector_width, grid_spacing):
        pygame.draw.line(screen, grid_color, (x, 0), (x, projector_height))

    # Draw horizontal grid lines
    for y in range(0, projector_height, grid_spacing):
        pygame.draw.line(screen, grid_color, (0, y), (projector_width, y))

    # Update the display
    pygame.display.flip()

# Clean up and exit
pygame.quit()
sys.exit()