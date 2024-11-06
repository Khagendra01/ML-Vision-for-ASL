from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
import time

picam2 = Picamera2()

# Configure video settings
video_config = picam2.create_video_configuration(main={"size": (1920, 1080)})
picam2.configure(video_config)

# Create H264 encoder
encoder = H264Encoder(bitrate=10000000)

# Start the camera
picam2.start()

# Start recording video
picam2.start_recording(encoder, 'output_video.h264')

# Record for 10 seconds
time.sleep(10)

# Stop recording
picam2.stop_recording()

# Stop the camera
picam2.stop()

print("Video recording completed.")
