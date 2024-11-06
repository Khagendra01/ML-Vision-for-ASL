from ultralytics import YOLO
from picamera2 import Picamera2
import cv2
import time
import numpy as np

# Initialize Picamera2
picam2 = Picamera2()

# Configure video settings
video_config = picam2.create_video_configuration(main={"size": (1280, 720), "format": "RGB888"})
picam2.configure(video_config)

# Load YOLO model
model = YOLO("yolov8n.pt")  # Using YOLOv8 nano model

# Start the camera
picam2.start()

# Scale factor for 7-inch screen (adjust as needed)
scale_factor = 0.5  # Example scale; adjust depending on the exact display resolution
display_width = int(1280 * scale_factor)
display_height = int(720 * scale_factor)

# Display without saving
try:
    while True:
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

        # Resize frame to fit the display
        resized_frame = cv2.resize(frame, (display_width, display_height))

        # Display the frame
        cv2.imshow("Camera Feed", resized_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop the camera
    picam2.stop()
    # Close all OpenCV windows
    cv2.destroyAllWindows()

print("Display terminated.")
