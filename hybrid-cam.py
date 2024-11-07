import cv2
import time
import threading
from picamera2 import Picamera2

# Combined function to handle both cameras
def run_both_cameras():
    # OpenCV camera setup
    cap = cv2.VideoCapture(8)  # Adjust the index for your camera
    if not cap.isOpened():
        print("Error: Could not open OpenCV camera.")
        return

    # PiCamera setup
    picam2 = Picamera2()
    video_config = picam2.create_video_configuration(main={"size": (1280, 720), "format": "RGB888"})
    picam2.configure(video_config)
    picam2.start()

    # Start time to control duration for both cameras
    start_time_opencv = time.time()
    start_time_picamera = time.time()

    # Loop to run both cameras
    while True:
        # Read frame from OpenCV camera
        ret, frame_opencv = cap.read()
        if not ret:
            print("Error: Failed to capture OpenCV image.")
            break

        # Capture frame from PiCamera
        frame_picamera = picam2.capture_array()

        # Convert PiCamera frame to RGB if necessary
        if frame_picamera.shape[2] == 4:  # If RGBA, convert to RGB
            frame_picamera = cv2.cvtColor(frame_picamera, cv2.COLOR_RGBA2RGB)

        # Display OpenCV camera feed
        cv2.imshow('OpenCV Camera Feed', frame_opencv)

        # Display PiCamera feed
        cv2.imshow('PiCamera Feed', frame_picamera)

        # Break if 20 seconds have passed for OpenCV and 10 seconds for PiCamera
        if time.time() - start_time_opencv > 20 and time.time() - start_time_picamera > 10:
            break

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release OpenCV camera and stop PiCamera
    cap.release()
    picam2.stop()

    # Close all OpenCV windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_both_cameras()
