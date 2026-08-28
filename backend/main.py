from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import mediapipe as mp
import numpy as np
import base64
import json
from typing import Dict, Any, Optional
from datetime import datetime
import os
import sys
import pickle
import tensorflow as tf

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our improved models with FSM
from rep_counters.pushup_counter import PushupCounter
from rep_counters.squat_counter import SquatCounter
from rep_counters.curl_counter import CurlCounter
from rep_counters.shoulder_press_counter import ShoulderPressCounter

# Pydantic models
class LoginRequest(BaseModel):
    name: str
    role: str = "user"

class WorkoutFrame(BaseModel):
    frame: str  # base64 encoded image

class User(BaseModel):
    name: str
    role: str
    email: str
    joinDate: str

# FastAPI app
app = FastAPI(
    title="AlphaReps API",
    description="AI-Powered Gym Trainer API",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MediaPipe (optimized for speed)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,  # Reduced for faster processing
    enable_segmentation=False,
    min_detection_confidence=0.5,  # Slightly lower for faster detection
    min_tracking_confidence=0.5
)

# Initialize classifier and counters
classifier = None
label_encoder = None
scaler = None
sequence_length = 30
c_lstm_model = None
counters = {
    'push_up': PushupCounter(),
    'squat': SquatCounter(),
    'barbell_biceps_curl': CurlCounter(),
    'hammer_curl': CurlCounter(),
    'shoulder_press': ShoulderPressCounter()
}

# Session storage (in-memory for demo)
user_sessions = {}
current_exercise = None
current_counter = None
# Optimized classification settings for faster response
classification_buffer = []
classification_buffer_size = 1  # Further reduced for immediate response
min_classification_confidence = 0.35  # Lower for faster detection
frame_skip_counter = 0
process_every_nth_frame = 2  # Process every 2nd frame for better performance
exercise_history = []
stable_frames = 0
required_stable_frames = 5
min_reps_before_switch = 10

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    global classifier, label_encoder, scaler, c_lstm_model
    print("="*60)
    print("  ALPHAREPS API STARTING")
    print("="*60)
    
    # Load BiLSTM exercise classifier - use absolute path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    meta_path = os.path.join(script_dir, "scripts", "models", "bilstm_exercise_model.pkl")
    keras_path = os.path.join(script_dir, "scripts", "models", "bilstm_exercise_model.keras")
    
    print(f"[*] Looking for model at: {meta_path}")
    
    if os.path.exists(meta_path) and os.path.exists(keras_path):
        print("[*] Loading BiLSTM exercise classifier...")
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        global label_encoder, scaler, sequence_length
        label_encoder = meta["label_encoder"]
        scaler = meta["scaler"]
        sequence_length = meta["sequence_length"]
        c_lstm_model = tf.keras.models.load_model(keras_path)
        classifier = True
        print("[+] BiLSTM model loaded successfully!")
    else:
        print("[!] BiLSTM model not found. Please train the model first.")
        print(f"    Expected path: {meta_path}")
        print("    Run: python backend/scripts/models/bilstm_training.py")
        classifier = None
    
    print("[+] AlphaReps API Ready!")
    print("[*] Listening on http://localhost:8000")
    print("="*60)

@app.get("/")
async def root():
    return {
        "message": "Welcome to AlphaReps - AI-Powered Gym Trainer! 💪",
        "version": "2.0.0",
        "status": "running",
        "model_loaded": classifier is not None,
        "features": [
            "Real-time exercise detection",
            "Automatic rep counting",
            "Posture correction",
            "5 supported exercises",
            "95%+ AI accuracy"
        ]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": classifier is not None,
        "timestamp": datetime.now().isoformat()
    }

# Authentication endpoints
@app.post("/api/auth/login")
async def login(request: LoginRequest):
    """Simple login endpoint"""
    user_data = {
        "name": request.name,
        "role": request.role,
        "email": f"{request.name.lower().replace(' ', '')}@alphareps.com",
        "joinDate": datetime.now().strftime("%b %Y")
    }
    
    # Store session
    user_sessions[request.name] = {
        "user": user_data,
        "login_time": datetime.now(),
        "workouts": []
    }
    
    return {
        "success": True,
        "user": user_data,
        "message": f"Welcome, {request.name}!"
    }

# Workout endpoints
@app.post("/api/workout/analyze")
async def analyze_frame(data: WorkoutFrame):
    """Analyze a single workout frame"""
    global current_exercise, current_counter, stable_frames, exercise_history
    
    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )
    
    try:
        # Decode base64 image
        image_data = base64.b64decode(data.frame.split(',')[1] if ',' in data.frame else data.frame)
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Get pose landmarks (single processing for both classification and counting)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        
        if not results.pose_landmarks:
            return {
                "success": False,
                "exercise": "NONE",
                "reps": 0,
                "stage": "ready",
                "feedback": "No person detected. Position yourself in frame.",
                "confidence": 0,
                "formQuality": "UNKNOWN",
                "angles": {}
            }
        
        # Extract simple keypoints (x,y,z only) as in BiLSTM core
        keypoints = []
        for lm in results.pose_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
        features = np.array(keypoints, dtype=np.float32)

        # Prepare sequence buffer for LSTM (repurpose classification_buffer)
        classification_buffer.append(features)
        if len(classification_buffer) > sequence_length:
            classification_buffer.pop(0)

        if len(classification_buffer) < sequence_length:
            most_common_exercise = None
            exercise_confidence = 0.0
        else:
            seq = np.array(classification_buffer[-sequence_length:], dtype=np.float32)
            seq_reshaped = seq.reshape(-1, seq.shape[-1])
            seq_scaled = scaler.transform(seq_reshaped)
            seq_scaled = seq_scaled.reshape(1, sequence_length, -1)
            probs = c_lstm_model.predict(seq_scaled, verbose=0)[0]
            class_idx = int(np.argmax(probs))
            detected_exercise = label_encoder.inverse_transform([class_idx])[0]
            exercise_confidence = float(np.max(probs))
            most_common_exercise = detected_exercise
        
        # Exercise switching logic with stable frames only
        if current_exercise is None:
            # No exercise locked yet – require reasonable confidence and a few stable frames
            if exercise_confidence >= min_classification_confidence:
                stable_frames += 1
                if stable_frames >= required_stable_frames:
                    current_exercise = most_common_exercise
                    current_counter = counters.get(current_exercise)
                    if current_counter:
                        current_counter.reset()
                    exercise_history.append(current_exercise)
                    print(f"\n{'='*60}")
                    print(f"[EXERCISE DETECTED] {current_exercise.upper().replace('_', ' ')}")
                    print(f"[CONFIDENCE] {exercise_confidence:.1%}")
                    print(f"{'='*60}\n")
                    stable_frames = 0
        else:
            # Allow switching if classifier detects a different exercise
            if most_common_exercise != current_exercise and exercise_confidence >= min_classification_confidence:
                stable_frames += 1
                if stable_frames >= required_stable_frames:
                    current_reps = current_counter.get_count() if current_counter else 0
                    print(f"\n{'='*60}")
                    print(f"[EXERCISE SWITCH] {current_exercise.upper().replace('_', ' ')} → {most_common_exercise.upper().replace('_', ' ')}")
                    print(f"[COMPLETED REPS] {current_reps}")
                    print(f"{'='*60}\n")
                    current_exercise = most_common_exercise
                    current_counter = counters.get(current_exercise)
                    if current_counter:
                        current_counter.reset()
                    classification_buffer.clear()
                    exercise_history.append(current_exercise)
                    stable_frames = 0
            else:
                # Same exercise prediction – reset stability counter
                stable_frames = 0
        
        # Count reps using new frontend-friendly method
        if current_counter and current_exercise:
            # Use new process_frame() method for clean JSON output
            counter_result = current_counter.process_frame(results.pose_landmarks)
            
            # Extract common values
            reps = counter_result.get("reps", 0)
            stage = counter_result.get("state", "idle")
            angle = counter_result.get("angle", 0)
            
            # Extract form feedback from all counters (all have form feedback now)
            feedback = counter_result.get("form_feedback", f"{stage.upper()} - {reps} REPS")
            form_quality = counter_result.get("form_quality", "GOOD")
            
            # Build exercise-specific angles dict
            if current_exercise == 'push_up':
                angles = {
                    "elbow": angle if angle else 0,
                    "back": counter_result.get("back_angle", 0)
                }
            elif current_exercise == 'squat':
                angles = {
                    "knee": angle if angle else 0,
                    "back": counter_result.get("back_angle", 0)
                }
            elif current_exercise in ['barbell_biceps_curl', 'hammer_curl']:
                angles = {
                    "elbow": angle if angle else 0,
                    "arm": self.primary_arm if hasattr(counter_result, 'primary_arm') else "right"
                }
            elif current_exercise == 'shoulder_press':
                angles = {
                    "elbow": angle if angle else 0,
                    "alignment": counter_result.get("wrist_alignment", 0)
                }
            else:
                angles = {"primary": angle if angle else 0}
            
            # Real-time terminal display
            if current_exercise:
                status_line = f"\r[{current_exercise.upper().replace('_', ' ')}] State: {stage.upper():<5} | Angle: {angle:5.1f}° | Reps: {reps}"
                sys.stdout.write(status_line)
                sys.stdout.flush()
            
            # Log rep increments (only when rep is actually counted)
            rep_incremented = counter_result.get("rep_incremented", False)
            if rep_incremented:
                sys.stdout.write("\n")  # Move to next line to preserve log
                print(f"[REP COUNTED] {current_exercise.upper().replace('_', ' ')} #{reps} | Angle: {angle:.1f}°")
        else:
            reps, stage = 0, "idle"
            feedback = f"Detecting {most_common_exercise.replace('_', ' ').title()}..." if most_common_exercise else "Detecting..."
            form_quality = "UNKNOWN"
            angles = {}
            rep_incremented = False
        
        # Use current_exercise if set, otherwise use detected exercise
        display_exercise = current_exercise if current_exercise else (most_common_exercise if most_common_exercise else "Detecting...")
        
        return {
            "success": True,
            "exercise": display_exercise.upper().replace('_', ' ') if display_exercise else "DETECTING...",
            "reps": reps,
            "stage": stage,
            "feedback": feedback,
            "confidence": int(exercise_confidence * 100),
            "formQuality": form_quality,
            "angles": angles,
            "rep_incremented": rep_incremented
        }
        
    except Exception as e:
        print(f"Error analyzing frame: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# User stats endpoints
@app.get("/api/user/stats")
async def get_user_stats():
    """Get user workout statistics"""
    return {
        "totalWorkouts": 24,
        "totalReps": 1240,
        "avgAccuracy": 94,
        "streak": 7,
        "recentWorkouts": [
            {
                "exercise": "Push-ups",
                "reps": 50,
                "accuracy": 96,
                "date": "Today",
                "time": "10:30 AM"
            },
            {
                "exercise": "Squats",
                "reps": 60,
                "accuracy": 94,
                "date": "Today",
                "time": "09:15 AM"
            }
        ]
    }

@app.get("/api/user/workouts")
async def get_user_workouts():
    """Get user workout history"""
    return {
        "workouts": [
            {
                "id": 1,
                "exercise": "Push-ups",
                "reps": 50,
                "accuracy": 96,
                "date": "2025-10-31",
                "duration": 300
            }
        ]
    }

# Admin endpoints
@app.get("/api/admin/members")
async def get_all_members():
    """Get all gym members"""
    return {
        "members": [
            {
                "id": 1,
                "name": "John Doe",
                "email": "john@example.com",
                "workouts": 24,
                "accuracy": 95,
                "status": "active",
                "joined": "2 days ago"
            },
            {
                "id": 2,
                "name": "Sarah Smith",
                "email": "sarah@example.com",
                "workouts": 18,
                "accuracy": 92,
                "status": "active",
                "joined": "1 week ago"
            }
        ]
    }

@app.get("/api/admin/stats")
async def get_gym_stats():
    """Get gym-wide statistics"""
    return {
        "totalMembers": 142,
        "activeToday": 48,
        "totalWorkouts": 1248,
        "avgAccuracy": 93
    }

@app.get("/api/exercises")
async def get_supported_exercises():
    """Get list of supported exercises"""
    return {
        "exercises": [
            "push_up",
            "squat",
            "barbell_biceps_curl",
            "hammer_curl",
            "shoulder_press"
        ],
        "total_count": 5
    }

@app.post("/api/workout/reset")
async def reset_workout():
    """Reset workout session"""
    global current_exercise, current_counter, classification_buffer, stable_frames, exercise_history
    
    current_exercise = None
    current_counter = None
    classification_buffer.clear()
    stable_frames = 0
    exercise_history = []
    for counter in counters.values():
        counter.reset()
    
    print("[INFO] Workout session reset")
    
    return {
        "success": True,
        "message": "Workout session reset successfully"
    }

if __name__ == "__main__":
    import uvicorn
    
    print("\nStarting AlphaReps API Server...\n")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
