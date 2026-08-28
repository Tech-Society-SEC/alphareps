"""
Unified Workout System - Integrates video_exercise_classifier with rep counters
"""
import cv2
import numpy as np
import mediapipe as mp
import os
import pickle
import tensorflow as tf
from rep_counters.pushup_counter import PushupCounter
from rep_counters.squat_counter import SquatCounter
from rep_counters.curl_counter import CurlCounter
from rep_counters.shoulder_press_counter import ShoulderPressCounter

class UnifiedWorkoutSystem:
    """
    Unified system that:
    1. Uses video_exercise_classifier for exercise detection
    2. Routes to appropriate rep counter based on detected exercise
    3. Provides real-time feedback
    """
    
    def __init__(self, model_path=None):
        print("Initializing Unified Workout System...")
        
        # Load BiLSTM exercise classifier
        meta_path = os.path.join("scripts", "models", "bilstm_exercise_model.pkl")
        keras_path = os.path.join("scripts", "models", "bilstm_exercise_model.keras")
        if not (os.path.exists(meta_path) and os.path.exists(keras_path)):
            print("No trained BiLSTM model available. Please train the model first.")
            print("   Run: python backend/scripts/models/bilstm_training.py")
            raise FileNotFoundError("No trained BiLSTM model found")

        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        self.label_encoder = meta["label_encoder"]
        self.scaler = meta["scaler"]
        self.sequence_length = meta["sequence_length"]
        self.lstm_model = tf.keras.models.load_model(keras_path)
        
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize rep counters
        self.counters = {
            'push_up': PushupCounter(),
            'squat': SquatCounter(),
            'barbell_biceps_curl': CurlCounter(),
            'hammer_curl': CurlCounter(),
            'shoulder_press': ShoulderPressCounter()
        }
        
        # State tracking
        self.current_exercise = None
        self.current_counter = None
        self.exercise_history = []
        self.confidence_threshold = 0.7
        self.stable_frames = 0
        self.required_stable_frames = 5  # Need 5 consistent frames to change exercise
        self.min_reps_before_switch = 10
        self.sequence_buffer = []
        
        print("Unified Workout System Ready!")
        print(f"Supported Exercises: {list(self.counters.keys())}")
    
    def reset_session(self):
        """Reset current workout session state and all rep counters"""
        self.current_exercise = None
        self.current_counter = None
        self.exercise_history = []
        self.stable_frames = 0
        
        # Reset all individual exercise counters
        for counter in self.counters.values():
            counter.reset()
    
    def process_frame(self, frame):
        """
        Process a single frame:
        1. Detect pose landmarks
        2. Classify exercise
        3. Count reps with appropriate counter
        4. Return results
        """
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect pose
        results = self.pose.process(rgb_frame)
        
        if not results.pose_landmarks:
            return {
                'success': False,
                'message': 'No person detected',
                'exercise': self.current_exercise or 'NONE',
                'reps': 0,
                'stage': 'ready',
                'feedback': 'Position yourself in frame'
            }
        
        # Extract simple keypoints (x,y,z only) for LSTM
        keypoints = []
        for lm in results.pose_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
        features = np.array(keypoints, dtype=np.float32)

        # Maintain sequence buffer
        self.sequence_buffer.append(features)
        if len(self.sequence_buffer) > self.sequence_length:
            self.sequence_buffer.pop(0)

        if len(self.sequence_buffer) < self.sequence_length:
            detected_exercise = self.current_exercise or 'push_up'
        else:
            seq = np.array(self.sequence_buffer[-self.sequence_length:], dtype=np.float32)
            seq_reshaped = seq.reshape(-1, seq.shape[-1])
            seq_scaled = self.scaler.transform(seq_reshaped)
            seq_scaled = seq_scaled.reshape(1, self.sequence_length, -1)
            probs = self.lstm_model.predict(seq_scaled, verbose=0)[0]
            class_idx = int(np.argmax(probs))
            detected_exercise = self.label_encoder.inverse_transform([class_idx])[0]
        
        # Smooth exercise transitions (require multiple consistent frames)
        # and lock current exercise until minimum reps are completed
        if detected_exercise != self.current_exercise:
            # Determine how many reps have been completed for the current exercise
            current_reps = self.current_counter.get_count() if self.current_counter else 0

            # Allow switching only if no exercise is set yet or minimum reps reached
            if self.current_exercise is None or current_reps >= self.min_reps_before_switch:
                self.stable_frames += 1
                if self.stable_frames >= self.required_stable_frames:
                    self._switch_exercise(detected_exercise)
                    self.stable_frames = 0
            else:
                # Not enough reps yet; keep current exercise locked
                self.stable_frames = 0
        else:
            self.stable_frames = 0
        
        # Get rep counting results
        rep_result = self._count_reps(results.pose_landmarks)
        
        # Draw landmarks on frame
        self.mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )
        
        # Add overlay information
        self._draw_overlay(frame, rep_result)
        
        return {
            'success': True,
            'frame': frame,
            'exercise': self.current_exercise or detected_exercise,
            'detected_exercise': detected_exercise,
            **rep_result
        }
    
    def _switch_exercise(self, new_exercise):
        """Switch to a new exercise and reset counter"""
        print(f"Switching exercise: {self.current_exercise} -> {new_exercise}")
        
        self.current_exercise = new_exercise
        self.current_counter = self.counters.get(new_exercise)
        
        if self.current_counter:
            # Reset the counter for new exercise
            self.current_counter.reset()
        
        self.exercise_history.append(new_exercise)
    
    def _count_reps(self, landmarks):
        """Count reps using the appropriate counter"""
        if not self.current_counter:
            return {
                'reps': 0,
                'stage': 'ready',
                'feedback': f'Detecting {self.current_exercise}...',
                'form_quality': 'UNKNOWN',
                'angles': {}
            }
        
        # Different counters return different formats
        prev_count = self.current_counter.get_count()
        result = self.current_counter.count_rep(landmarks)
        new_count = self.current_counter.get_count()

        # Debug: print when a rep is actually counted
        if new_count != prev_count:
            print(f"Rep detected: {self.current_exercise} -> {new_count}")
        
        # Standardize output format
        if isinstance(result, tuple):
            if len(result) == 6:  # PushupCounter format
                reps, stage, elbow_angle, feedback, back_angle, form_quality = result
                return {
                    'reps': reps,
                    'stage': stage,
                    'feedback': feedback,
                    'form_quality': form_quality,
                    'angles': {
                        'elbow': elbow_angle,
                        'back': back_angle
                    }
                }
            elif len(result) == 3:  # Standard format (reps, stage, angle)
                reps, stage, angle = result
                return {
                    'reps': reps,
                    'stage': stage if stage else 'ready',
                    'feedback': f'{stage.upper()}' if stage else 'READY',
                    'form_quality': 'GOOD',
                    'angles': {'primary': angle}
                }
        
        # Fallback
        return {
            'reps': 0,
            'stage': 'ready',
            'feedback': 'Processing...',
            'form_quality': 'UNKNOWN',
            'angles': {}
        }
    
    def _draw_overlay(self, frame, rep_result):
        """Draw information overlay on frame"""
        h, w = frame.shape[:2]
        
        # Background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Exercise name
        cv2.putText(frame, f"Exercise: {self.current_exercise or 'DETECTING'}", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Rep count
        cv2.putText(frame, f"Reps: {rep_result['reps']}", 
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Stage
        cv2.putText(frame, f"Stage: {rep_result['stage']}", 
                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Feedback
        feedback_color = (0, 255, 0) if 'GOOD' in rep_result['feedback'] else (0, 165, 255)
        cv2.putText(frame, rep_result['feedback'], 
                    (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, feedback_color, 2)
    
    def run_webcam(self):
        """Run the system with webcam input"""
        print("\nStarting webcam workout session...")
        print("Press 'q' to quit")
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        frame_count = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Process every frame
                result = self.process_frame(frame)
                
                if result['success']:
                    cv2.imshow('AlphaReps - AI Workout Trainer', result['frame'])
                else:
                    cv2.imshow('AlphaReps - AI Workout Trainer', frame)
                
                frame_count += 1
                
                # Quit on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self._print_session_summary()
    
    def _print_session_summary(self):
        """Print workout session summary"""
        print("\n" + "="*50)
        print("WORKOUT SESSION SUMMARY")
        print("="*50)
        
        if self.current_counter:
            print(f"Final Exercise: {self.current_exercise}")
            print(f"Total Reps: {self.current_counter.counter}")
        
        if self.exercise_history:
            print(f"\nExercises Detected: {set(self.exercise_history)}")
        
        print("="*50)


def main():
    """Main entry point"""
    try:
        # Initialize system
        system = UnifiedWorkoutSystem()
        
        # Run with webcam
        system.run_webcam()
        
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print("\n📝 To train the model, run:")
        print("   cd backend/scripts")
        print("   python train_video_model.py")
    except KeyboardInterrupt:
        print("\n\n👋 Workout session ended by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
