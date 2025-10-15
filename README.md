# 💪 AlphaRep - AI-Powered Personal Trainer

AlphaRep is a cutting-edge web application that serves as your AI-powered personal trainer. Using computer vision and machine learning, it provides real-time exercise detection, repetition counting, form correction, and workout analytics through your webcam.

## 🌟 Features

### 🤖 AI-Powered Exercise Detection
- **Real-time pose detection** using MediaPipe
- **9 exercise types supported**: Push-ups, Pull-ups, Squats, Lunges, Plank, Shoulder Press, Bent Over Row, Chest Dips, Glute Bridge
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
python -c "
import asyncio
from models.exercise_classifier import ExerciseClassifier
async def train():
    classifier = ExerciseClassifier()
    await classifier.train_model()
asyncio.run(train())
"
```

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

### Exercise Classification Model
- **Algorithm**: Random Forest Classifier
- **Features**: 132 pose landmarks (33 points × 4 coordinates)
- **Training Data**: 3,501 labeled exercise images
- **Accuracy**: 90%+ on test set
- **Classes**: 9 exercise types

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
│   ├── main.py                 # FastAPI server
│   ├── models/
│   │   ├── exercise_classifier.py
│   │   ├── pose_analyzer.py
│   │   ├── rep_counter.py
│   │   └── form_checker.py
│   ├── database/
│   │   └── db_manager.py
│   └── utils/
│       └── face_recognition_utils.py
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── context/
│   │   └── App.js
│   └── package.json
├── dataset/
│   └── imagePoses.csv          # Training data
└── requirements.txt
```

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
| Pull-ups | ✅ | ✅ | Advanced |
| Squats | ✅ | ✅ | Beginner |
| Lunges | ✅ | ✅ | Intermediate |
| Plank | ⏱️ Timer | ✅ | Beginner |
| Shoulder Press | ✅ | ✅ | Intermediate |
| Bent Over Row | ✅ | ✅ | Intermediate |
| Chest Dips | ✅ | ✅ | Advanced |
| Glute Bridge | ✅ | ✅ | Beginner |

## 🎯 Performance Metrics

- **Exercise Detection**: 90%+ accuracy
- **Rep Counting**: 95%+ accuracy
- **Form Analysis**: Real-time feedback
- **Latency**: <200ms processing time
- **Frame Rate**: 5 FPS analysis

## 🔒 Privacy & Security

- **Local Processing**: All AI analysis happens locally
- **Secure Storage**: Face encodings encrypted
- **No Video Recording**: Only pose landmarks stored
- **GDPR Compliant**: User data control

## 🚀 Deployment

### Using Docker (Recommended)

1. **Build and run with Docker Compose**
```bash
docker-compose up --build
```

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

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MediaPipe** for pose detection
- **FastAPI** for the backend framework
- **React.js** for the frontend
- **scikit-learn** for machine learning
- **OpenCV** for computer vision
- **Tailwind CSS** for styling

## 📞 Support

For support, email support@alpharep.ai or join our Discord community.

---

**Built with ❤️ by the AlphaRep Team**

*Transform your fitness journey with AI-powered personal training!* 💪🤖
