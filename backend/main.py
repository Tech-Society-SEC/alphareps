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

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our models
from scripts.models.video_exercise_classifier import VideoExerciseClassifier
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

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    global classifier
    print("="*60)
    print("  ALPHAREPS API STARTING")
    print("="*60)
    
    # Load exercise classifier
    model_path = "scripts/models/video_exercise_model.pkl"
    if os.path.exists(model_path):
        print("[*] Loading exercise classifier...")
        classifier = VideoExerciseClassifier()
        classifier.load_model(model_path)
        print("[+] Model loaded successfully!")
    else:
        print("[!] Model not found. Please train the model first.")
        print("    Run: python backend/scripts/train_video_model.py")
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
    global current_exercise, current_counter
    
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
        
        # Extract landmarks for classification (reuse existing pose results)
        landmarks = classifier.extract_landmarks_from_results(results)
        
        if landmarks is None:
            return {
                "success": False,
                "exercise": "NONE",
                "reps": 0,
                "stage": "ready",
                "feedback": "Could not extract landmarks",
                "confidence": 0,
                "formQuality": "UNKNOWN",
                "angles": {}
            }
        
        # Classify exercise with confidence
        detected_exercise = classifier.predict(landmarks)
        probabilities = classifier.predict_proba(landmarks)
        exercise_confidence = max(probabilities.values()) if probabilities else 0
        
        # Simplified classification for faster response
        classification_buffer.append(detected_exercise)
        if len(classification_buffer) > classification_buffer_size:
            classification_buffer.pop(0)
        
        # Use immediate detection with minimal buffering
        most_common_exercise = detected_exercise
        vote_confidence = exercise_confidence
        
        # Faster exercise detection logic
        if current_exercise is None:
            # First detection - set exercise immediately if confidence is reasonable
            if exercise_confidence >= min_classification_confidence:
                current_exercise = most_common_exercise
                current_counter = counters.get(current_exercise)
                if current_counter:
                    current_counter.reset()
                print(f"[INFO] Exercise detected: {current_exercise} (confidence: {exercise_confidence:.2f})")
        elif most_common_exercise != current_exercise and exercise_confidence >= min_classification_confidence + 0.1:
            # Allow faster exercise change with slightly higher confidence
            print(f"[INFO] Exercise changed from {current_exercise} to {most_common_exercise}")
            current_exercise = most_common_exercise
            current_counter = counters.get(current_exercise)
            if current_counter:
                current_counter.reset()
            classification_buffer.clear()
        
        # Count reps
        if current_counter and current_exercise:
            result = current_counter.count_rep(results.pose_landmarks)
            print(f"[DEBUG] Exercise: {current_exercise}, Counter result: {result}")
            
            # Parse result based on counter type
            if isinstance(result, tuple) and len(result) >= 2:
                if len(result) == 6:  # PushupCounter
                    reps, stage, elbow_angle, feedback, back_angle, form_quality = result
                    angles = {"elbow": elbow_angle or 0, "back": back_angle or 0}
                elif len(result) == 3:  # Standard counters (Curl, Squat, etc.)
                    reps, stage, angle = result
                    feedback = f"{stage.upper()} - {reps} REPS" if stage else "READY"
                    form_quality = "GOOD"
                    angles = {"primary": angle if angle else 0}
                elif len(result) == 2:  # Basic counter
                    reps, stage = result
                    feedback = f"{stage.upper()} - {reps} REPS" if stage else "READY"
                    form_quality = "GOOD"
                    angles = {}
                else:
                    reps, stage = result[0] if len(result) > 0 else 0, result[1] if len(result) > 1 else "ready"
                    feedback = f"{stage.upper()} - {reps} REPS" if stage else "READY"
                    form_quality = "GOOD"
                    angles = {}
            else:
                reps, stage = 0, "ready"
                feedback = "Processing..."
                form_quality = "UNKNOWN"
                angles = {}
        else:
            reps, stage = 0, "ready"
            feedback = f"Detecting {most_common_exercise.replace('_', ' ').title()}..."
            form_quality = "UNKNOWN"
            angles = {}
        
        # Use current_exercise if set, otherwise use detected exercise
        display_exercise = current_exercise if current_exercise else most_common_exercise
        
        return {
            "success": True,
            "exercise": display_exercise.upper().replace('_', ' '),
            "reps": reps,
            "stage": stage,
            "feedback": feedback,
            "confidence": int(exercise_confidence * 100),
            "formQuality": form_quality,
            "angles": angles
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
    global current_exercise, current_counter, classification_buffer
    
    current_exercise = None
    current_counter = None
    classification_buffer.clear()
    
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
