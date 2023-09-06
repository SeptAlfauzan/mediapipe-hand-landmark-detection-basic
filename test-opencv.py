import cv2
import numpy as np
import time

# Capture a frame from the camera

# Get the current time in milliseconds
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Camera not found or could not be opened.")
    exit()

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    frame_timestamp_ms = int(time.time() * 1000)

    # Check if the frame was read successfully
    if not ret:
        print("Error: Could not read frame.")
        break

    # You can now process or display the frame as needed.
    # For example, displaying it in a window:
    cv2.imshow("Camera Video", frame)
    print("timeframe", frame_timestamp_ms)
    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the camera and close any open windows
cap.release()
cv2.destroyAllWindows()
