# 💪 AlphaRep - AI-Powered Personal Trainer

AlphaRep is a cutting-edge web application that serves as your AI-powered personal trainer. Using computer vision and machine learning, it provides real-time exercise detection, repetition counting, form correction, and workout analytics through your webcam.


### 🤖 AI-Powered Exercise Detection
- **Real-time pose detection** using MediaPipe
- **5 exercise types supported**: Push-ups, Hammer curls, Bicep Curls, Squats, Shoulder Press
- **Automatic exercise classification** with 90%+ accuracy

### 📊 Smart Form Analysis
- **Real-time form feedback** with specific corrections
- **Form scoring system** (0-100%) with detailed breakdowns
- **Pose similarity analysis** comparing to ideal form templates
- **Joint angle calculations** for precise movement analysis

### 🔢 Intelligent Rep Counting
- **Automatic repetition counting** based on joint angles
- **Exercise-specific thresholds** for accurate counting
- **Plank timer** for static exercises
- **Rep quality scoring** to ensure proper range of motion

### 👤 Face Recognition Attendance
- **Contactless login** using face recognition
- **User registration** with face encoding
- **Attendance logging** with timestamps
- **Secure face data storage**

### 📈 Comprehensive Analytics
- **Workout history tracking**
- **Progress visualization** with charts
- **Calorie burn estimation**
- **Performance metrics** and trends
- **Leaderboard system**

### 🎯 Modern Web Interface
- **Responsive React.js frontend**
- **Real-time WebSocket communication**
- **Beautiful UI with Tailwind CSS**
- **Mobile-friendly design**

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   React.js      │    │   FastAPI        │    │   PostgreSQL    │
│   Frontend      │◄──►│   Backend        │◄──►│   Database      │
│                 │    │                  │    │                 │
│ • Webcam        │    │ • MediaPipe      │    │ • User Data     │
│ • Real-time UI  │    │ • ML Models      │    │ • Workouts      │
│ • Charts        │    │ • WebSockets     │    │ • Analytics     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- Webcam access
- Modern web browser

### Backend Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd AlphaReps
```

2. **Create virtual environment**
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Train the ML model**
```bash
cd backend
python train_video_model.py
```

**Note**: Training requires the dataset videos. The model will:
- Process 101 videos (3,015 frames)
- Extract 146 features per frame
- Train ensemble model (RF + GB + SVM)
- Save model to `models/video_exercise_model.pkl`

5. **Start the backend server**
```bash
cd backend
python main.py
```

### Frontend Setup

1. **Install Node.js dependencies**
```bash
cd frontend
npm install
```

2. **Start the development server**
```bash
npm start
```

3. **Open your browser**
Navigate to `http://localhost:3000`

## 📱 Usage Guide

### 1. **Login/Registration**
- **Face Recognition**: Position your face in the camera and click "Login with Face"
- **Credentials**: Use any username/password for demo
- **Register Face**: Enter your name and capture your face for future logins

### 2. **Start a Workout**
- Navigate to the **Workout** page
- Allow camera access when prompted
- Click **"Start Workout"** to begin
- Position yourself in the camera frame

### 3. **Exercise Detection**
- Perform any supported exercise
- Watch real-time detection in the top-left overlay
- See your rep count in the top-right corner
- Monitor form score in the bottom-left

### 4. **Form Feedback**
- Read real-time feedback in the right panel
- Adjust your form based on AI suggestions
- Aim for 90+ form score for excellent technique

### 5. **View Analytics**
- Check your **Dashboard** for workout history
- View progress charts and statistics
- Compare with others on the **Leaderboard**

## 🧠 ML Model Details

### Video Exercise Classification Model
- **Algorithm**: Ensemble (Random Forest + Gradient Boosting + SVM)
- **Features**: 146 enhanced features (132 pose landmarks + 14 curl-specific features)
- **Training Data**: 3,015 video frames from 101 exercise videos
- **Accuracy**: 99.50% on test set
- **Classes**: 5 exercise types (barbell biceps curl, hammer curl, push-up, shoulder press, squat)

### Pose Analysis Pipeline
1. **MediaPipe Pose Detection** → Extract 33 body landmarks
2. **Feature Engineering** → Calculate joint angles and distances
3. **Exercise Classification** → Predict exercise type
4. **Rep Counting** → Track movement cycles
5. **Form Analysis** → Compare with ideal poses

## 🗂️ Project Structure

```
AlphaReps/
├── backend/
│   ├── main.py                           # FastAPI server
│   ├── train_video_model.py              # Model training script
│   ├── test_realtime.py                  # Real-time testing
│   ├── test_video_model.py               # Model evaluation
│   ├── models/
│   │   ├── video_exercise_classifier.py  # Enhanced ML classifier
│   │   └── video_exercise_model.pkl      # Trained model (excluded from git)
│   ├── database/
│   │   └── db_manager.py                 # Database operations
│   └── utils/
│       └── face_recognition_utils.py     # Face recognition utilities
├── dataset/                              # Training videos (excluded from git)
│   ├── barbell-biceps-curl/             # 25 videos
│   ├── hammer-curl/                     # 12 videos  
│   ├── push-up/                         # 25 videos
│   ├── shoulder-press/                  # 20 videos
│   └── squat/                           # 19 videos
├── .gitignore                           # Git ignore rules
├── requirements.txt                     # Python dependencies
└── README.md                            # Project documentation
```

### 📁 File Management

**Included in Git:**
- ✅ Source code (`backend/*.py`)
- ✅ Configuration files (`.gitignore`, `requirements.txt`)
- ✅ Documentation (`README.md`)

**Excluded from Git:**
- ❌ Trained models (`*.pkl`, `*.joblib`)
- ❌ Dataset videos (`dataset/*.mp4`)
- ❌ Cache files (`__pycache__/`)
- ❌ Virtual environments (`venv/`, `.env`)
- ❌ IDE settings (`.vscode/`, `.idea/`)

**Model Files:**
- `video_exercise_model.pkl` - Generated after training
- Must be trained locally using `python train_video_model.py`
- Size: ~50MB (too large for Git)

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the backend directory:

```env
DATABASE_URL=sqlite:///./data/alpharep.db
SECRET_KEY=your-secret-key-here
CORS_ORIGINS=http://localhost:3000
```

### Model Parameters
Adjust in `backend/models/exercise_classifier.py`:

```python
# Random Forest parameters
n_estimators=100
max_depth=20
min_samples_split=5
```

## 📊 Supported Exercises

| Exercise | Rep Counting | Form Analysis | Difficulty |
|----------|-------------|---------------|------------|
| Push-ups | ✅ | ✅ | Beginner |
| Hammer curls | ✅ | ✅ | Beginner |
| Bicep curls | ✅ | ✅ | Beginner |
| Squats | ✅ | ✅ | Beginner |
| Shoulder Press | ✅ | ✅ | Intermediate |

## 🔒 Privacy & Security

- **Local Processing**: All AI analysis happens locally
- **Secure Storage**: Face encodings encrypted
- **No Video Recording**: Only pose landmarks stored
- **GDPR Compliant**: User data control

### Manual Deployment

1. **Backend (FastAPI)**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

2. **Frontend (React)**
```bash
npm run build
serve -s build -l 3000
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

