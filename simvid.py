from ultralytics import YOLO
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput
import cv2
import time
import numpy as np

# Initialize Picamera2
picam2 = Picamera2()

# Configure video settings
video_config = picam2.create_video_configuration(main={"size": (1280, 720), "format": "RGB888"})
picam2.configure(video_config)

# Create H264 encoder
encoder = H264Encoder(bitrate=10000000)

# Create output for video file
output = FfmpegOutput('output_video.mp4')

# Load YOLO11 model
model = YOLO("yolov8n.pt")  # Using YOLOv8 nano model

# Start the camera
picam2.start()

# Start recording video
picam2.start_recording(encoder, output)

# Record and display for 50 seconds
start_time = time.time()
while (time.time() - start_time) < 50:
    # Capture frame
    frame = picam2.capture_array()
    
    # Ensure frame is in correct format (RGB)
    if frame.shape[2] == 4:  # If RGBA
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
    
    # Perform object detection
    results = model(frame)
    
    # Draw bounding boxes and labels
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"{r.names[int(box.cls[0])]} {box.conf[0]:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow("Camera Feed", frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Stop recording
picam2.stop_recording()

# Stop the camera
picam2.stop()

# Close all OpenCV windows
cv2.destroyAllWindows()

print("Video recording completed.")
