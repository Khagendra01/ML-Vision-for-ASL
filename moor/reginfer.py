import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Configuration
SEQUENCE_LENGTH = 54
INPUT_SHAPE = (SEQUENCE_LENGTH, 99)  # 33 landmarks * 3 coordinates
ACTIONS = ['violence', 'neutral']

class FightDetector:
    def __init__(self, model_path):
        # Load the trained model
        self.model = tf.keras.models.load_model(model_path)
        self.sequence = []

    def normalize_keypoints(self, keypoints):
        """Convert to hip-centered coordinates and normalize"""
        hip = keypoints[23]  # Left hip landmark
        normalized = keypoints - hip
        return normalized.flatten()  # Flatten to 1D array (33*3=99 elements)

    def preprocess_frame(self, frame):
        """Process frame to extract and normalize keypoints"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        
        if results.pose_landmarks:
            # Extract and normalize keypoints
            keypoints = np.array([[lmk.x, lmk.y, lmk.z] 
                                for lmk in results.pose_landmarks.landmark])
            normalized_kps = self.normalize_keypoints(keypoints)
            return normalized_kps
        return None

    def predict_action(self):
        """Predict action based on the collected sequence"""
        if len(self.sequence) == SEQUENCE_LENGTH:
            sequence_array = np.array(self.sequence)
            sequence_array = np.expand_dims(sequence_array, axis=0)  # Add batch dimension
            prediction = self.model.predict(sequence_array)
            action_idx = int(prediction[0][0] > 0.5)  # Threshold at 0.5 for binary classification
            return ACTIONS[action_idx]
        return None

    def run_inference(self):
        cap = cv2.VideoCapture(0)
        print("Running inference. Press 'q' to quit.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Mirror display
            frame = cv2.flip(frame, 1)
            
            # Process frame and collect sequence
            normalized_kps = self.preprocess_frame(frame)
            if normalized_kps is not None:
                self.sequence.append(normalized_kps)
                if len(self.sequence) > SEQUENCE_LENGTH:
                    self.sequence.pop(0)  # Maintain fixed sequence length

                # Predict action
                action = self.predict_action()
                if action:
                    cv2.putText(frame, f"Action: {action.upper()}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Display frame
            cv2.imshow('Fight Detection', frame)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    model_path = 'fight_detector.h5'  # Path to the trained model
    detector = FightDetector(model_path)
    detector.run_inference()