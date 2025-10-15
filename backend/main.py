from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import cv2
import mediapipe as mp
import numpy as np
import base64
import json
import asyncio
from typing import List, Dict, Any
import pickle
from datetime import datetime
import os

# Import our custom modules
from models.exercise_classifier import ExerciseClassifier
from models.pose_analyzer import PoseAnalyzer
from models.rep_counter import RepCounter
from models.form_checker import FormChecker
from database.db_manager import DatabaseManager
# from utils.face_recognition_utils import FaceRecognitionManager

app = FastAPI(title="AlphaRep API", description="AI-Powered Personal Trainer API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize our models
exercise_classifier = ExerciseClassifier()
pose_analyzer = PoseAnalyzer()
rep_counter = RepCounter()
form_checker = FormChecker()
db_manager = DatabaseManager()
# face_manager = FaceRecognitionManager()

# Store active connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.user_sessions: Dict[str, Dict] = {}

    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.user_sessions[user_id] = {
            "websocket": websocket,
            "current_exercise": None,
            "rep_count": 0,
            "session_start": datetime.now(),
            "form_feedback": []
        }

    def disconnect(self, websocket: WebSocket, user_id: str):
        self.active_connections.remove(websocket)
        if user_id in self.user_sessions:
            del self.user_sessions[user_id]

    async def send_personal_message(self, message: dict, user_id: str):
        if user_id in self.user_sessions:
            websocket = self.user_sessions[user_id]["websocket"]
            await websocket.send_text(json.dumps(message))

manager = ConnectionManager()

# Startup initialization moved to avoid deprecation warning
async def initialize_app():
    """Initialize the application"""
    print("🚀 AlphaRep API Starting...")
    
    # Load or train the exercise classification model
    if not os.path.exists("models/exercise_model.pkl"):
        print("📚 Training exercise classification model...")
        await exercise_classifier.train_model()
    else:
        print("📚 Loading pre-trained exercise model...")
        exercise_classifier.load_model()
    
    # Initialize database
    await db_manager.initialize()
    print("✅ AlphaRep API Ready!")

@app.get("/")
async def root():
    return {
        "message": "Welcome to AlphaRep - AI-Powered Personal Trainer! 💪",
        "version": "1.0.0",
        "features": [
            "Real-time exercise detection",
            "Repetition counting",
            "Form correction",
            "Face recognition attendance",
            "Workout analytics"
        ]
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now()}

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for real-time pose analysis"""
    await manager.connect(websocket, user_id)
    
    try:
        while True:
            # Receive frame data from client
            data = await websocket.receive_text()
            frame_data = json.loads(data)
            
            # Decode base64 image
            image_data = base64.b64decode(frame_data['image'].split(',')[1])
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Process frame
            results = await process_frame(frame, user_id)
            
            # Send results back to client
            await manager.send_personal_message(results, user_id)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id)
        print(f"User {user_id} disconnected")

async def process_frame(frame: np.ndarray, user_id: str) -> Dict[str, Any]:
    """Process a single frame for pose analysis"""
    try:
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get pose landmarks
        results = pose.process(rgb_frame)
        
        if results.pose_landmarks:
            # Extract landmarks
            landmarks = pose_analyzer.extract_landmarks(results.pose_landmarks)
            
            # Classify exercise
            exercise_type = exercise_classifier.predict(landmarks)
            
            # Count repetitions
            rep_count = rep_counter.count_reps(landmarks, exercise_type)
            
            # Check form
            form_feedback = form_checker.analyze_form(landmarks, exercise_type)
            
            # Update user session
            if user_id in manager.user_sessions:
                session = manager.user_sessions[user_id]
                session["current_exercise"] = exercise_type
                session["rep_count"] = rep_count
                session["form_feedback"] = form_feedback
            
            # Calculate calories (rough estimate)
            calories = calculate_calories(exercise_type, rep_count)
            
            return {
                "exercise_detected": exercise_type,
                "rep_count": rep_count,
                "form_feedback": form_feedback,
                "calories_burned": calories,
                "landmarks_detected": True,
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "exercise_detected": "none",
                "rep_count": 0,
                "form_feedback": ["No pose detected. Please ensure you're visible in the camera."],
                "calories_burned": 0,
                "landmarks_detected": False,
                "timestamp": datetime.now().isoformat()
            }
            
    except Exception as e:
        return {
            "error": f"Error processing frame: {str(e)}",
            "exercise_detected": "error",
            "rep_count": 0,
            "form_feedback": ["Error in pose analysis"],
            "calories_burned": 0,
            "landmarks_detected": False,
            "timestamp": datetime.now().isoformat()
        }

def calculate_calories(exercise_type: str, rep_count: int) -> float:
    """Calculate estimated calories burned"""
    # Rough calorie estimates per rep
    calorie_per_rep = {
        "pushup": 0.5,
        "pullup": 1.0,
        "squat": 0.4,
        "lunges": 0.6,
        "plank": 0.3,  # per second
        "shoulder_press": 0.4,
        "bent_over_row": 0.5,
        "chest_dips": 0.7,
        "glute_bridge": 0.3
    }
    
    return rep_count * calorie_per_rep.get(exercise_type, 0.5)

@app.get("/user/{user_id}/stats")
async def get_user_stats(user_id: str):
    """Get user workout statistics"""
    try:
        stats = await db_manager.get_user_stats(user_id)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching stats: {str(e)}")

@app.post("/user/{user_id}/workout")
async def save_workout(user_id: str, workout_data: dict):
    """Save completed workout session"""
    try:
        await db_manager.save_workout(user_id, workout_data)
        return {"message": "Workout saved successfully", "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving workout: {str(e)}")

@app.get("/exercises")
async def get_supported_exercises():
    """Get list of supported exercises"""
    return {
        "exercises": [
            "pushup", "pullup", "squat", "lunges", "plank",
            "shoulder_press", "bent_over_row", "chest_dips", "glute_bridge"
        ],
        "total_count": 9
    }

if __name__ == "__main__":
    import uvicorn
    import asyncio
    
    # Initialize the app
    asyncio.run(initialize_app())
    
    # Start the server
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
