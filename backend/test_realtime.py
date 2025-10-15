#!/usr/bin/env python3
"""
Real-time Exercise Classification Test

This script uses your webcam to test the trained model in real-time.
Perform exercises in front of your camera and see live predictions!

Usage:
    python test_realtime.py

Controls:
- Press 'q' to quit
- Press 's' to save current frame
- Press 'r' to reset prediction history
"""

import cv2
import numpy as np
import sys
import os
from collections import deque, Counter
import time

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.video_exercise_classifier import VideoExerciseClassifier

class RealTimeClassifier:
    def __init__(self):
        self.classifier = VideoExerciseClassifier()
        self.model_loaded = False
        self.prediction_history = deque(maxlen=10)  # Store last 10 predictions
        self.confidence_history = deque(maxlen=10)
        
    def load_model(self):
        """Load the trained model"""
        print("📚 Loading trained model...")
        success = self.classifier.load_model("models/video_exercise_model.pkl")
        if success:
            print("✅ Model loaded successfully!")
            self.model_loaded = True
            print(f"📋 Supported exercises: {', '.join(self.classifier.exercise_classes)}")
        else:
            print("❌ Failed to load model. Please train the model first.")
        return success
    
    def get_stable_prediction(self):
        """Get stable prediction from recent history"""
        if len(self.prediction_history) < 3:
            return "Detecting...", 0.0
        
        # Get most common prediction
        prediction_counts = Counter(self.prediction_history)
        most_common = prediction_counts.most_common(1)[0]
        prediction = most_common[0]
        
        # Calculate average confidence for this prediction
        relevant_confidences = [
            conf for pred, conf in zip(self.prediction_history, self.confidence_history)
            if pred == prediction
        ]
        avg_confidence = np.mean(relevant_confidences) if relevant_confidences else 0.0
        
        return prediction, avg_confidence
    
    def draw_info_overlay(self, frame, prediction, confidence, fps):
        """Draw information overlay on frame"""
        height, width = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        
        # Top bar for prediction
        cv2.rectangle(overlay, (0, 0), (width, 80), (0, 0, 0), -1)
        
        # Bottom bar for instructions
        cv2.rectangle(overlay, (0, height-60), (width, height), (0, 0, 0), -1)
        
        # Blend overlay
        alpha = 0.7
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # Draw prediction text
        pred_text = f"Exercise: {prediction.upper().replace('_', ' ')}"
        conf_text = f"Confidence: {confidence:.1%}"
        fps_text = f"FPS: {fps:.1f}"
        
        # Color based on confidence
        if confidence > 0.8:
            color = (0, 255, 0)  # Green
        elif confidence > 0.6:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 0, 255)  # Red
        
        cv2.putText(frame, pred_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, conf_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, fps_text, (width-150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Instructions
        instructions = [
            "Press 'q' to quit | 's' to save frame | 'r' to reset"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (10, height-40+i*20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run_realtime_test(self):
        """Run real-time classification"""
        if not self.model_loaded:
            print("❌ Model not loaded!")
            return
        
        print("\n🎥 Starting real-time exercise classification...")
        print("📹 Make sure you're visible in the camera frame")
        print("🏋️ Perform exercises and see live predictions!")
        print("\nControls:")
        print("  - Press 'q' to quit")
        print("  - Press 's' to save current frame")
        print("  - Press 'r' to reset prediction history")
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Could not open camera")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        frame_count = 0
        start_time = time.time()
        
        print("✅ Camera initialized. Starting classification...")
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("❌ Failed to read from camera")
                break
            
            frame_count += 1
            current_time = time.time()
            fps = frame_count / (current_time - start_time)
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process every 5th frame to reduce computation
            if frame_count % 5 == 0:
                # Extract landmarks
                landmarks = self.classifier.extract_landmarks_from_frame(frame)
                
                if landmarks is not None:
                    # Get prediction
                    prediction = self.classifier.predict(landmarks)
                    prob_dict = self.classifier.predict_proba(landmarks)
                    confidence = max(prob_dict.values()) if prob_dict else 0.0
                    
                    # Add to history
                    self.prediction_history.append(prediction)
                    self.confidence_history.append(confidence)
            
            # Get stable prediction
            stable_pred, stable_conf = self.get_stable_prediction()
            
            # Draw overlay
            frame = self.draw_info_overlay(frame, stable_pred, stable_conf, fps)
            
            # Show frame
            cv2.imshow('AlphaRep - Real-time Exercise Classification', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("👋 Quitting...")
                break
            elif key == ord('s'):
                # Save current frame
                filename = f"captured_frame_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Frame saved as {filename}")
            elif key == ord('r'):
                # Reset prediction history
                self.prediction_history.clear()
                self.confidence_history.clear()
                print("🔄 Prediction history reset")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Camera released")

def main():
    """Main function"""
    print("🎥 AlphaRep Real-time Exercise Classifier")
    print("=" * 45)
    
    classifier = RealTimeClassifier()
    
    # Load model
    if not classifier.load_model():
        print("\n💡 To train the model first, run:")
        print("   python train_video_model.py")
        return
    
    # Run real-time test
    try:
        classifier.run_realtime_test()
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
