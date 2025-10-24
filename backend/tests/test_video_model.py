#!/usr/bin/env python3
"""
Test script for the Video Exercise Classifier

This script allows you to test the trained model with new video files.
You can add your own videos and see how well the model classifies them.

Usage:
    python test_video_model.py

Features:
- Test individual video files
- Batch test multiple videos
- Real-time prediction with confidence scores
- Detailed analysis of predictions
"""

import asyncio
import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.video_exercise_classifier import VideoExerciseClassifier

class VideoTester:
    def __init__(self):
        self.classifier = VideoExerciseClassifier()
        self.model_loaded = False
        
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
    
    def test_single_video(self, video_path: str):
        """Test a single video file"""
        if not self.model_loaded:
            print("❌ Model not loaded!")
            return
        
        if not os.path.exists(video_path):
            print(f"❌ Video file not found: {video_path}")
            return
        
        print(f"\n🎬 Testing video: {os.path.basename(video_path)}")
        print("-" * 50)
        
        # Extract landmarks from video
        landmarks_list = self.classifier.process_video_file(video_path, max_frames=10)
        
        if not landmarks_list:
            print("❌ No pose landmarks detected in video")
            return
        
        print(f"📊 Extracted {len(landmarks_list)} frames with pose data")
        
        # Get predictions for each frame
        predictions = []
        confidences = []
        
        for i, landmarks in enumerate(landmarks_list):
            prediction = self.classifier.predict(landmarks)
            prob_dict = self.classifier.predict_proba(landmarks)
            
            if prob_dict:
                confidence = max(prob_dict.values())
                predictions.append(prediction)
                confidences.append(confidence)
                
                print(f"Frame {i+1:2d}: {prediction:15s} (confidence: {confidence:.2%})")
        
        if predictions:
            # Get overall prediction (most common)
            from collections import Counter
            prediction_counts = Counter(predictions)
            final_prediction = prediction_counts.most_common(1)[0][0]
            avg_confidence = np.mean(confidences)
            
            print(f"\n🎯 Final Prediction: {final_prediction.upper()}")
            print(f"📊 Average Confidence: {avg_confidence:.2%}")
            print(f"📈 Prediction Distribution:")
            
            for exercise, count in prediction_counts.items():
                percentage = (count / len(predictions)) * 100
                print(f"   • {exercise:15s}: {count:2d} frames ({percentage:5.1f}%)")
    
    def test_multiple_videos(self, video_directory: str):
        """Test all videos in a directory"""
        if not self.model_loaded:
            print("❌ Model not loaded!")
            return
        
        if not os.path.exists(video_directory):
            print(f"❌ Directory not found: {video_directory}")
            return
        
        # Find all video files
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(Path(video_directory).glob(f"*{ext}"))
        
        if not video_files:
            print(f"❌ No video files found in {video_directory}")
            return
        
        print(f"\n📁 Testing {len(video_files)} videos from {video_directory}")
        print("=" * 60)
        
        results = []
        
        for video_file in video_files:
            print(f"\n🎬 Testing: {video_file.name}")
            
            # Extract landmarks
            landmarks_list = self.classifier.process_video_file(str(video_file), max_frames=5)
            
            if landmarks_list:
                # Get prediction for first frame (quick test)
                prediction = self.classifier.predict(landmarks_list[0])
                prob_dict = self.classifier.predict_proba(landmarks_list[0])
                confidence = max(prob_dict.values()) if prob_dict else 0.0
                
                results.append({
                    'file': video_file.name,
                    'prediction': prediction,
                    'confidence': confidence,
                    'frames': len(landmarks_list)
                })
                
                print(f"   Prediction: {prediction:15s} (confidence: {confidence:.2%})")
            else:
                print("   ❌ No pose detected")
        
        # Summary
        print(f"\n📊 SUMMARY - Tested {len(results)} videos")
        print("=" * 60)
        
        for result in sorted(results, key=lambda x: x['confidence'], reverse=True):
            print(f"{result['file']:25s} → {result['prediction']:15s} ({result['confidence']:.2%})")
    
    def interactive_test(self):
        """Interactive testing mode"""
        if not self.model_loaded:
            print("❌ Model not loaded!")
            return
        
        print("\n🎮 Interactive Video Testing Mode")
        print("=" * 40)
        print("Commands:")
        print("  1. Test single video")
        print("  2. Test directory of videos")
        print("  3. Exit")
        
        while True:
            try:
                choice = input("\nEnter your choice (1-3): ").strip()
                
                if choice == '1':
                    video_path = input("Enter video file path: ").strip()
                    self.test_single_video(video_path)
                
                elif choice == '2':
                    directory = input("Enter directory path: ").strip()
                    self.test_multiple_videos(directory)
                
                elif choice == '3':
                    print("👋 Goodbye!")
                    break
                
                else:
                    print("❌ Invalid choice. Please enter 1, 2, or 3.")
            
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """Main function"""
    print("🧪 AlphaRep Video Model Tester")
    print("=" * 40)
    
    tester = VideoTester()
    
    # Load the model
    if not tester.load_model():
        print("\n💡 To train the model first, run:")
        print("   python train_video_model.py")
        return
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        if os.path.isfile(video_path):
            tester.test_single_video(video_path)
        elif os.path.isdir(video_path):
            tester.test_multiple_videos(video_path)
        else:
            print(f"❌ Path not found: {video_path}")
    else:
        # Interactive mode
        tester.interactive_test()

if __name__ == "__main__":
    main()
