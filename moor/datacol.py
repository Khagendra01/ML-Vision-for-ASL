import cv2
import numpy as np
import os
import mediapipe as mp
from collections import deque

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Configuration
SEQUENCE_LENGTH = 54
DATA_PATH = 'dataset'
ACTIONS = ['violence', 'neutral']
MIN_DETECTION_CONFIDENCE = 0.75

class DataCollector:
    def __init__(self):
        self.sequence = []
        self.action = 'neutral'
        self.collecting = False
        self.frame_count = 0
        
        # Create dataset directory structure
        for action in ACTIONS:
            os.makedirs(os.path.join(DATA_PATH, action), exist_ok=True)

    def normalize_keypoints(self, keypoints):
        """Convert to hip-centered coordinates and normalize"""
        hip = keypoints[23]  # Left hip landmark
        normalized = keypoints - hip
        return normalized.flatten()  # Flatten to 1D array (33*3=99 elements)

    def collect_data(self):
        cap = cv2.VideoCapture(0)
        print("Data collection started. Press:")
        print("v - Start recording violence sequence")
        print("n - Start recording neutral sequence")
        print("q - Quit")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Mirror display
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Pose detection
            results = pose.process(rgb_frame)
            
            if results.pose_landmarks:
                # Extract and normalize keypoints
                keypoints = np.array([[lmk.x, lmk.y, lmk.z] 
                                    for lmk in results.pose_landmarks.landmark])
                normalized_kps = self.normalize_keypoints(keypoints)
                
                # Visualization
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                # Collect sequence
                if self.collecting:
                    self.sequence.append(normalized_kps)
                    self.frame_count += 1
                    
                    # Save sequence when collected enough frames
                    if self.frame_count == SEQUENCE_LENGTH:
                        action_dir = os.path.join(DATA_PATH, self.action)
                        sequence_count = len(os.listdir(action_dir))
                        np.save(os.path.join(action_dir, f'seq_{sequence_count}'), 
                               np.array(self.sequence))
                        
                        print(f"Saved {self.action} sequence {sequence_count}")
                        self.collecting = False
                        self.sequence = []
                        self.frame_count = 0

            # Display info
            cv2.putText(frame, f"Mode: {self.action.upper()}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Data Collection', frame)

            # Key controls
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('p'):
                self.action = 'violence'
                self.collecting = True
            elif key == ord('n'):
                self.action = 'neutral'
                self.collecting = True

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    collector = DataCollector()
    collector.collect_data()