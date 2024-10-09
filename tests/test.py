import cv2
import threading
import numpy as np

# test a single window
cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)
while True:
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    cv2.imshow("Camera Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyWindow("Camera Feed")

# test a second window
cv2.namedWindow("Camera Feed part 2", cv2.WINDOW_NORMAL)
while True:
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    cv2.imshow("Camera Feed part 2", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyWindow("Camera Feed part 2")

# test two windows in parallel

def window1():
    cv2.namedWindow("Camera Feed 1", cv2.WINDOW_NORMAL)
    while True:
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imshow("Camera Feed 1", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyWindow("Camera Feed 1")

def window2():
    cv2.namedWindow("Camera Feed 2", cv2.WINDOW_NORMAL)
    while True:
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imshow("Camera Feed 2", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyWindow("Camera Feed 2")

t1 = threading.Thread(target=window1)
t2 = threading.Thread(target=window2)

t1.start()
t2.start()

t1.join()
t2.join()
