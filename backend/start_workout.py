"""
Quick Start Script - Launch AlphaReps Workout System
Uses video_exercise_classifier for exercise detection and rep counting
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    print("="*60)
    print("  ALPHAREPS - AI WORKOUT TRAINER")
    print("="*60)
    print()
    
    # Check if model exists
    model_paths = [
        'models/video_exercise_model.pkl',          # Primary location
        'backend/models/video_exercise_model.pkl',  # Full backend path
        'scripts/models/bilstm_exercise_model.pkl', # BiLSTM model
        'backend/scripts/models/bilstm_exercise_model.pkl' # BiLSTM full path
    ]
    
    model_found = False
    for path in model_paths:
        if os.path.exists(path):
            model_found = True
            print(f"Model found: {path}")
            break
    
    if not model_found:
        print("Model not found!")
        print()
        print("Please train the model first:")
        print("   cd backend/scripts")
        print("   python train_video_model.py")
        print()
        return
    
    print()
    print()
    print("Starting Unified Workout System...")
    from unified_workout_system import UnifiedWorkoutSystem
    system = UnifiedWorkoutSystem()
    system.run_webcam()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
