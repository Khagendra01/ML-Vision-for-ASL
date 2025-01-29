import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
import os

# Configuration
DATA_PATH = 'dataset'
ACTIONS = ['violence', 'neutral']
SEQUENCE_LENGTH = 54
INPUT_SHAPE = (SEQUENCE_LENGTH, 99)  # 33 landmarks * 3 coordinates

class FightDetectorTrainer:
    def __init__(self):
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_dataset(self):
        sequences = []
        labels = []
        
        for action_idx, action in enumerate(ACTIONS):
            action_dir = os.path.join(DATA_PATH, action)
            for seq_file in os.listdir(action_dir):
                seq_path = os.path.join(action_dir, seq_file)
                sequence = np.load(seq_path)
                sequences.append(sequence)
                labels.append(action_idx)
        
        # Convert to numpy arrays
        X = np.array(sequences)
        y = np.array(labels)  # Binary labels (0 or 1)
        
        # Split dataset
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.1, random_state=42, stratify=y
        )

    def build_model(self):
        self.model = Sequential([
            LSTM(32, input_shape=INPUT_SHAPE, return_sequences=False),
            Dropout(0.1),
            Dense(16, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')  # Single output neuron for binary classification
        ])

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )

    def train(self):
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=7, restore_best_weights=True)
        
        history = self.model.fit(
            self.X_train,
            self.y_train,
            epochs=10,
            batch_size=32,
            validation_data=(self.X_test, self.y_test),
            callbacks=[early_stop],
            class_weight=self._get_class_weights()  # Handle class imbalance
        )
        
        return history

    def _get_class_weights(self):
        # Calculate class weights to handle imbalance
        class_counts = np.bincount(self.y_train)
        return {0: 1.0/class_counts[0], 1: 1.0/class_counts[1]}

    def save_model(self, format='tflite'):
        self.model.save('fight_detector.h5')
        
        if format == 'tflite':
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
            converter._experimental_lower_tensor_list_ops = False
            
            tflite_model = converter.convert()
            
            with open('fight_detector.tflite', 'wb') as f:
                f.write(tflite_model)

if __name__ == "__main__":
    trainer = FightDetectorTrainer()
    trainer.load_dataset()
    print(f"Dataset loaded: {len(trainer.X_train)} training samples, {len(trainer.X_test)} test samples")
    
    trainer.build_model()
    trainer.model.summary()
    
    print("Starting training...")
    history = trainer.train()
    
    print("Evaluating model:")
    loss, accuracy, precision, recall = trainer.model.evaluate(trainer.X_test, trainer.y_test)
    print(f"Test accuracy: {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}% | Recall: {recall*100:.2f}%")
    
    trainer.save_model()
    print("Model saved in H5 and TFLite formats")
