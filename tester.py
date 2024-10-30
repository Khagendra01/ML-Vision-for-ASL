import cv2
import numpy as np
from keras.models import load_model

# Load the pre-trained model
model = load_model('model.h5')

import collections

# Buffer to store the sequence of frames
frame_buffer = collections.deque(maxlen=30)  # Buffer stores the last 30 frames

def preprocess_frame(frame):
    # Resize the frame to the required size (you may need to adjust this)
    frame_resized = cv2.resize(frame, (224, 224))  # Resize frame to match your pre-trained model's input size
    frame_resized = frame_resized.astype('float32') / 255.0  # Normalize frame
    return frame_resized

def predict_action(sequence):
    # Assume the model expects a sequence of 30 frames, where each frame is represented by a feature vector
    # You may need to extract features (e.g., using a pre-trained network) if your model was trained on features
    prediction = model.predict(np.expand_dims(sequence, axis=0))  # Add batch dimension
    return prediction

# Open webcam for real-time testing
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Get prediction from the model
    prediction = predict_action(frame)
    
    # Display the prediction on the frame (you can adjust this according to your model's output)
    action_label = np.argmax(prediction)  # Assuming model output is a softmax of class probabilities
    confidence = np.max(prediction)
    
    # You can map action_label to actual action names using a label dictionary
    action_dict = {0: 'Action 1', 1: 'Action 2', 2: 'Action 3'}  # Adjust based on your classes
    predicted_action = action_dict.get(action_label, 'Unknown')
    
    # Display the prediction and confidence on the video feed
    cv2.putText(frame, f'Action: {predicted_action}, Confidence: {confidence:.2f}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Real-time Action Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
