# IMMEDIATE CONFIRMATION - Print before ANY imports
print("🚀 Script started...", flush=True)
print("Loading libraries...", flush=True)

import os
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.mixed_precision import set_global_policy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import pickle

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import sys

print("✅ All libraries loaded!", flush=True)

# FORCE UNBUFFERED OUTPUT
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)

# Configure GPU FIRST (with immediate feedback)
print("\n" + "="*70, flush=True)
print("🚀 CONFIGURING GPU AND CUDA", flush=True)
print("="*70, flush=True)

print(f"TensorFlow version: {tf.__version__}", flush=True)

# List GPUs
gpus = tf.config.list_physical_devices('GPU')
print(f"\n🔍 Available GPUs: {len(gpus)}", flush=True)

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"   ✅ {gpu.name} - Memory growth enabled", flush=True)
        
        tf.config.set_visible_devices(gpus, 'GPU')
        
        # Enable mixed precision for RTX 3060
        print("\n⚡ Enabling mixed precision (FP16)...", flush=True)
        set_global_policy('mixed_float16')
        print("   ✅ Mixed precision enabled", flush=True)
        
        print(f"\n✅ GPU configured: {gpus[0].name}", flush=True)
        
    except RuntimeError as e:
        print(f"❌ GPU error: {e}", flush=True)
else:
    print("⚠️  NO GPU DETECTED - Training will be SLOW", flush=True)

print("="*70, flush=True)

class BiLSTMExerciseClassifier:
    """BiLSTM-based exercise classifier optimized for RTX 3060"""
    
    def __init__(self):
        print("\n🔥 Initializing BiLSTM classifier...", flush=True)
        
        self.sequence_length = 30
        self.frame_size = (128, 128)
        self.n_features = 99
        
        # MediaPipe setup
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.exercise_classes = [
            'barbell_biceps_curl',
            'hammer_curl',
            'push_up',
            'shoulder_press',
            'squat'
        ]
        
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.model = None
        
        print(f"   • Sequence length: {self.sequence_length} frames", flush=True)
        print(f"   • Features: {self.n_features}", flush=True)
        print(f"   • Classes: {len(self.exercise_classes)}", flush=True)
        print("✅ Classifier initialized", flush=True)
    
    def extract_keypoints_from_frame(self, frame):
        """Extract 99 keypoint features (x,y,z only)"""
        try:
            frame_resized = cv2.resize(frame, self.frame_size)
            rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                keypoints = []
                for landmark in results.pose_landmarks.landmark:
                    keypoints.extend([landmark.x, landmark.y, landmark.z])
                return np.array(keypoints)
            else:
                return np.zeros(self.n_features)
        except:
            return np.zeros(self.n_features)
    
    def process_video_to_sequence(self, video_path, stride=3):
        """Extract sequences with sliding window"""
        sequences = []
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return sequences
            
            all_keypoints = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                keypoints = self.extract_keypoints_from_frame(frame)
                all_keypoints.append(keypoints)
            
            cap.release()
            
            # Sliding window
            if len(all_keypoints) >= self.sequence_length:
                for i in range(0, len(all_keypoints) - self.sequence_length + 1, stride):
                    sequence = all_keypoints[i:i + self.sequence_length]
                    sequences.append(np.array(sequence))
            
        except Exception as e:
            print(f"Error: {e}", flush=True)
        
        return sequences
    
    def load_dataset(self, data_dir="dataset", stride=3):
        """Load dataset with progress tracking"""
        print(f"\n📊 Loading dataset from: {data_dir}", flush=True)
        print(f"   Sliding window stride: {stride}", flush=True)
        
        sequences = []
        labels = []
        
        for idx, exercise_name in enumerate(self.exercise_classes):
            exercise_path = os.path.join(data_dir, exercise_name)
            
            if not os.path.isdir(exercise_path):
                print(f"⚠️  Missing: {exercise_path}", flush=True)
                continue
            
            print(f"\n🎬 [{idx+1}/{len(self.exercise_classes)}] {exercise_name}", flush=True)
            
            video_files = [f for f in os.listdir(exercise_path) if f.endswith('.mp4')]
            print(f"   Videos found: {len(video_files)}", flush=True)
            
            for i, video_file in enumerate(video_files):
                video_path = os.path.join(exercise_path, video_file)
                print(f"   [{i+1}/{len(video_files)}] {video_file[:30]}...", end=' ', flush=True)
                
                video_sequences = self.process_video_to_sequence(video_path, stride=stride)
                
                if video_sequences:
                    sequences.extend(video_sequences)
                    labels.extend([exercise_name] * len(video_sequences))
                    print(f"✅ {len(video_sequences)} seq", flush=True)
                else:
                    print(f"❌", flush=True)
        
        X = np.array(sequences)
        y = np.array(labels)
        
        print(f"\n✅ Dataset loaded: {len(X)} sequences", flush=True)
        for cls in np.unique(y):
            print(f"   • {cls}: {np.sum(y == cls)}", flush=True)
        
        return X, y
    
    def preprocess_data(self, X, y):
        """Normalize and encode"""
        print("\n🔧 Preprocessing...", flush=True)
        
        n_samples, n_timesteps, n_features = X.shape
        X_reshaped = X.reshape(-1, n_features)
        X_scaled = self.scaler.fit_transform(X_reshaped)
        X_processed = X_scaled.reshape(n_samples, n_timesteps, n_features)
        
        y_encoded = self.label_encoder.fit_transform(y)
        y_categorical = to_categorical(y_encoded, num_classes=len(self.exercise_classes))
        
        print("   ✅ Normalized + Encoded", flush=True)
        return X_processed, y_categorical
    
    def build_model(self):
        """Build GPU-optimized BiLSTM"""
        print("\n🏗️  Building model on GPU...", flush=True)
        
        gpus = tf.config.list_physical_devices('GPU')
        device = '/GPU:0' if gpus else '/CPU:0'
        print(f"Using device: {device}", flush=True)

        with tf.device(device):
            model = Sequential([
                Bidirectional(LSTM(256, return_sequences=True),
                             input_shape=(self.sequence_length, self.n_features)),
                Dropout(0.2),
                Bidirectional(LSTM(128)),
                Dropout(0.2),
                Dense(128, activation='relu'),
                Dropout(0.3),
                Dense(64, activation='relu'),
                Dense(len(self.exercise_classes), activation='softmax', dtype='float32')
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=0.0005),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
        
        print(f"   ✅ Model built on {device}", flush=True)
        model.summary()
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=60, batch_size=32, class_weight=None):
        """Train with PROPER validation split"""
        print(f"\n🏋️ Training on GPU...", flush=True)
        print(f"   • Train: {len(X_train)} | Val: {len(X_val)}", flush=True)
        print(f"   • Batch: {batch_size} | Epochs: {epochs}", flush=True)
        print(f"   • Batches/epoch: {len(X_train)//batch_size}", flush=True)
        
        os.makedirs('models', exist_ok=True)
        
        # Create SEPARATE datasets (FIX for validation_split bug)
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.shuffle(1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1),
            ModelCheckpoint('models/bilstm_best_model.keras', monitor='val_accuracy', 
                          save_best_only=True, verbose=1)
        ]
        
        print("\n🚀 Starting training NOW...\n", flush=True)
        
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,  # FIXED: Use validation_data, not validation_split
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1
        )
        
        print("\n✅ Training done!", flush=True)
        return history
    
    def evaluate(self, X_test, y_test):
        """Evaluate on GPU"""
        print("\n📊 Evaluating...", flush=True)
        
        y_pred_proba = self.model.predict(X_test, batch_size=64, verbose=1)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        accuracy = np.mean(y_pred == y_true)
        print(f"\n🎯 Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)", flush=True)
        
        print("\n📋 Classification Report:", flush=True)
        print(classification_report(y_true, y_pred, target_names=self.exercise_classes, digits=4))
        
        cm = confusion_matrix(y_true, y_pred)
        print("\n🔢 Confusion Matrix:", flush=True)
        print(cm, flush=True)
        
        return accuracy, cm
    
    def save_model(self, model_path="models/bilstm_exercise_model.pkl"):
        """Save model"""
        print("\n💾 Saving...", flush=True)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        keras_path = model_path.replace('.pkl', '.keras')
        self.model.save(keras_path)
        print(f"   ✅ {keras_path}", flush=True)
        
        model_data = {
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'exercise_classes': self.exercise_classes,
            'sequence_length': self.sequence_length,
            'n_features': self.n_features
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"   ✅ {model_path}", flush=True)
    
    def plot_training_history(self, history):
        """Save plots"""
        print("\n📈 Saving plots...", flush=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].plot(history.history['accuracy'], label='Train', linewidth=2)
        axes[0].plot(history.history['val_accuracy'], label='Val', linewidth=2)
        axes[0].set_title('Accuracy', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(history.history['loss'], label='Train', linewidth=2)
        axes[1].plot(history.history['val_loss'], label='Val', linewidth=2)
        axes[1].set_title('Loss', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('models/training_history.png', dpi=300, bbox_inches='tight')
        print("   ✅ models/training_history.png", flush=True)
        plt.close()


def main():
    """GPU-accelerated training pipeline"""
    print("\n" + "="*70, flush=True)
    print("🔥 BiLSTM EXERCISE CLASSIFIER - GPU TRAINING", flush=True)
    print("="*70, flush=True)
    print("⚙️  BiLSTM 256→128 | LR 0.0005 | Batch 32 | FP16", flush=True)
    print("="*70, flush=True)
    
    classifier = BiLSTMExerciseClassifier()
    
    # STEP 1: Load dataset
    print("\n" + "="*70, flush=True)
    print("STEP 1: LOADING DATASET", flush=True)
    print("="*70, flush=True)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.dirname(os.path.dirname(script_dir))
    dataset_dir = os.path.join(backend_dir, "dataset")
    print(f"Using dataset directory: {dataset_dir}", flush=True)

    X, y = classifier.load_dataset(data_dir=dataset_dir, stride=3)
    
    if len(X) == 0:
        print("\n❌ NO DATA! Check dataset folder structure", flush=True)
        print("\n📁 Expected:", flush=True)
        print(f"   {os.path.join(dataset_dir, 'barbell_biceps_curl')}/*.mp4", flush=True)
        print(f"   {os.path.join(dataset_dir, 'hammer_curl')}/*.mp4", flush=True)
        print(f"   {os.path.join(dataset_dir, 'push_up')}/*.mp4", flush=True)
        print(f"   {os.path.join(dataset_dir, 'shoulder_press')}/*.mp4", flush=True)
        print(f"   {os.path.join(dataset_dir, 'squat')}/*.mp4", flush=True)
        return
    
    # STEP 2: Split
    print("\n" + "="*70, flush=True)
    print("STEP 2: TRAIN/TEST SPLIT", flush=True)
    print("="*70, flush=True)
    
    X_train, X_test, y_train_raw, y_test_raw = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train: {len(X_train)} | Test: {len(X_test)}", flush=True)
    
    # STEP 3: Preprocess
    print("\n" + "="*70, flush=True)
    print("STEP 3: PREPROCESSING", flush=True)
    print("="*70, flush=True)
    
    X_train_processed, y_train = classifier.preprocess_data(X_train, y_train_raw)
    
    # Transform test
    n_samples, n_timesteps, n_features = X_test.shape
    X_test_scaled = classifier.scaler.transform(X_test.reshape(-1, n_features))
    X_test_processed = X_test_scaled.reshape(n_samples, n_timesteps, n_features)
    y_test = to_categorical(
        classifier.label_encoder.transform(y_test_raw),
        num_classes=len(classifier.exercise_classes)
    )
    print("   ✅ Test set processed", flush=True)
    
    # STEP 4: Create validation split BEFORE building model
    print("\n" + "="*70, flush=True)
    print("STEP 4: VALIDATION SPLIT", flush=True)
    print("="*70, flush=True)
    
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_processed, y_train, test_size=0.2, random_state=42
    )
    print(f"   Train: {len(X_train_final)} | Val: {len(X_val)}", flush=True)
    
    y_train_indices = np.argmax(y_train_final, axis=1)
    classes = np.unique(y_train_indices)
    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_train_indices
    )
    class_weights = {int(cls): float(weight) for cls, weight in zip(classes, class_weights_array)}
    print("\n📊 Class weights:", flush=True)
    for cls, weight in class_weights.items():
        print(f"   • Class {cls}: {weight:.4f}", flush=True)
    
    # STEP 5: Build
    print("\n" + "="*70, flush=True)
    print("STEP 5: BUILDING MODEL", flush=True)
    print("="*70, flush=True)
    
    classifier.build_model()
    
    # STEP 6: Train (FIXED - no validation_split bug)       
    print("\n" + "="*70, flush=True)
    print("STEP 6: TRAINING", flush=True)
    print("="*70, flush=True)
    
    history = classifier.train(
        X_train_final, y_train_final,
        X_val, y_val,
        epochs=60,
        batch_size=32,
        class_weight=class_weights
    )
    
    # STEP 7: Evaluate
    print("\n" + "="*70, flush=True)
    print("STEP 7: EVALUATION", flush=True)
    print("="*70, flush=True)
    
    accuracy, cm = classifier.evaluate(X_test_processed, y_test)
    
    # STEP 8: Save
    print("\n" + "="*70, flush=True)
    print("STEP 8: SAVING", flush=True)
    print("="*70, flush=True)
    
    classifier.save_model()
    classifier.plot_training_history(history)
    
    # FINAL
    print("\n" + "="*70, flush=True)
    print("🏁 COMPLETE!", flush=True)
    print("="*70, flush=True)
    print(f"🎯 Final: {accuracy*100:.2f}%", flush=True)
    print("\n⚡ GPU Benefits:", flush=True)
    print("   • FP16 mixed precision: ~2x faster", flush=True)
    print("   • CUDA LSTM kernels: optimized", flush=True)
    print("   • RTX 3060: fully utilized", flush=True)
    print("\n📁 Saved:", flush=True)
    print("   • models/bilstm_exercise_model.pkl", flush=True)
    print("   • models/bilstm_best_model.keras", flush=True)
    print("   • models/training_history.png", flush=True)
    print("="*70, flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted!", flush=True)
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()