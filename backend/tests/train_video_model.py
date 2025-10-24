#!/usr/bin/env python3
"""
Training script for the Video Exercise Classifier

This script trains a machine learning model to classify exercises from video data.
It processes video files, extracts pose landmarks, and trains a Random Forest classifier.

Usage:
    python train_video_model.py

The script will:
1. Process all video files in the data/ directory
2. Extract pose landmarks from video frames
3. Train a Random Forest classifier
4. Save the trained model to models/video_exercise_model.pkl
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.video_exercise_classifier import VideoExerciseClassifier

async def train_model():
    """Train the video exercise classification model"""
    
    print("=" * 60)
    print("🏋️  AlphaRep Video Exercise Classifier Training")
    print("=" * 60)
    
    # Initialize classifier
    classifier = VideoExerciseClassifier()
    
    # Check if data directory exists
    data_dir = "dataset"
    if not os.path.exists(data_dir):
        print(f"❌ Data directory '{data_dir}' not found!")
        print("Please ensure your video data is organized in folders like:")
        print("  data/")
        print("    ├── barbell-biceps-curl/")
        print("    ├── hammer-curl/")
        print("    ├── push-up/")
        print("    ├── shoulder-press/")
        print("    └── squat/")
        return False
    
    # List available exercises
    exercise_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    print(f"\n📁 Found {len(exercise_folders)} exercise folders:")
    for folder in exercise_folders:
        video_count = len([f for f in os.listdir(os.path.join(data_dir, folder)) if f.endswith('.mp4')])
        print(f"  • {folder}: {video_count} videos")
    
    print(f"\n🎯 Target exercise classes:")
    for i, exercise in enumerate(classifier.exercise_classes, 1):
        print(f"  {i}. {exercise}")
    
    # Train the model
    try:
        print(f"\n🚀 Starting training process...")
        accuracy = await classifier.train_model(data_dir)
        
        print("\n" + "=" * 60)
        print(f"🎉 Training completed successfully!")
        print(f"📊 Final Accuracy: {accuracy:.2%}")
        print(f"💾 Model saved to: models/video_exercise_model.pkl")
        print("=" * 60)
        
        # Test the model with a quick prediction
        print(f"\n🧪 Testing model loading...")
        test_classifier = VideoExerciseClassifier()
        if test_classifier.load_model():
            print("✅ Model loads successfully!")
            print(f"📋 Supported exercises: {', '.join(test_classifier.exercise_classes)}")
        else:
            print("❌ Error loading saved model!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Training failed with error:")
        print(f"   {str(e)}")
        return False

def main():
    """Main entry point"""
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required!")
        sys.exit(1)
    
    # Check required packages
    try:
        import cv2
        import mediapipe
        import sklearn
        import numpy
        import pandas
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please install required packages with:")
        print("pip install opencv-python mediapipe scikit-learn numpy pandas")
        sys.exit(1)
    
    # Run training
    success = asyncio.run(train_model())
    
    if success:
        print(f"\n🎯 Next steps:")
        print(f"1. Test the model with new video data")
        print(f"2. Integrate with the AlphaRep web application")
        print(f"3. Fine-tune parameters if needed")
        sys.exit(0)
    else:
        print(f"\n💡 Troubleshooting tips:")
        print(f"1. Ensure all video files are in MP4 format")
        print(f"2. Check that videos contain clear human poses")
        print(f"3. Verify folder names match expected exercise types")
        sys.exit(1)

if __name__ == "__main__":
    main()
