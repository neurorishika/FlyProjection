import sys
from PySide2 import QtWidgets, QtCore
import pyqtgraph as pg

def main():
    app = QtWidgets.QApplication(sys.argv)
    
    # List available screens
    screens = app.screens()
    print(f"Available screens: {len(screens)}")
    for index, screen in enumerate(screens):
        print(f"Screen {index}: {screen.name()}")
    
    # Choose the second screen (index 1) if available
    if len(screens) > 1:
        screen = screens[1]
        print(f"Using Screen 1: {screen.name()}")
    else:
        screen = screens[0]
        print("Warning: Only one screen detected. Using primary screen.")
    
    # Create the main window
    window = QtWidgets.QMainWindow()
    window.setWindowTitle("Fullscreen Circle Test")
    # Remove window decorations (title bar, borders)
    window.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.FramelessWindowHint)
    # Move the window to the desired screen
    geometry = screen.geometry()
    window.setGeometry(geometry)
    
    # Create a central widget
    central_widget = QtWidgets.QWidget()
    window.setCentralWidget(central_widget)
    
    # Create a GraphicsLayoutWidget (from PyQtGraph)
    graphics_widget = pg.GraphicsLayoutWidget()
    graphics_widget.setBackground('k')  # Set background to black
    graphics_widget.ci.setContentsMargins(0, 0, 0, 0)  # Remove outer margins

    # Create a layout and add the graphics widget
    layout = QtWidgets.QVBoxLayout()
    layout.addWidget(graphics_widget)
    layout.setContentsMargins(0, 0, 0, 0)  # Set layout margins to zero
    central_widget.setLayout(layout)
    
    # Create a plot area (ViewBox)
    view = graphics_widget.addViewBox()
    view.setAspectLocked(True)  # Lock aspect ratio
    view.setBackgroundColor('k')  # Set background to black

    # Draw a circle at the center of the screen
    circle_diameter = 500  # Diameter of the circle in pixels
    circle = QtWidgets.QGraphicsEllipseItem(
        geometry.width() / 4 - circle_diameter / 2,  # x position
        geometry.height() / 2 - circle_diameter / 2,  # y position
        circle_diameter,  # width
        circle_diameter   # height
    )
    circle.setBrush(pg.mkBrush('w'))  # Fill color white
    circle.setPen(pg.mkPen(None))     # No border
    view.addItem(circle)

    # Set the view range to match the geometry of the screen
    view.setRange(QtCore.QRectF(0, 0, geometry.width(), geometry.height()))

    # Show the window in fullscreen mode
    window.showFullScreen()
    
    # Handle key press events to exit on ESC
    def keyPressEvent(event):
        if event.key() == QtCore.Qt.Key_Escape:
            app.quit()
    window.keyPressEvent = keyPressEvent
    
    # Start the application event loop
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
