# 💪 AlphaReps - AI-Powered Gym Trainer

<div align="center">

![AlphaReps Banner](https://img.shields.io/badge/AlphaReps-AI%20Gym%20Trainer-FF3B3B?style=for-the-badge&logo=dumbbell)
![Accuracy](https://img.shields.io/badge/Accuracy-95%25+-14B8A6?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)

**Your AI-Powered Personal Gym Trainer with Real-Time Posture Correction**

[Features](#-features) • [Quick Start](#-quick-start) • [Tech Stack](#-tech-stack) • [Screenshots](#-screenshots) • [Documentation](#-documentation)

</div>

---

## 🌟 What is AlphaReps?

AlphaReps is a **cutting-edge AI gym training platform** that combines computer vision, machine learning, and modern web technologies to provide:

- 🎯 **Automatic Exercise Detection** - AI identifies what you're doing
- 🔢 **Smart Rep Counting** - Only counts perfect form reps
- ✅ **Real-Time Posture Correction** - Instant feedback on form
- 🚀 **Simple Login System** - Quick name-based authentication
- 📊 **Advanced Analytics** - Track progress with beautiful charts
- 👨‍💼 **Admin Dashboard** - Perfect for gym owners

Built for **gyms**, **trainers**, and **fitness enthusiasts** who want AI-powered workout guidance.

---

## ✨ Features

### 🤖 AI-Powered Exercise Detection
- **Real-time pose detection** using MediaPipe with GPU acceleration
- **5 exercise types supported**: Push-ups, Squats, Bicep Curls, Hammer Curls, Shoulder Press
- **95%+ accuracy** with premium ensemble model (RF + ET + GB + Ada + SVM + XGBoost)
- **Automatic classification** - Just start exercising!

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

### 👤 Simple Login System
- **Name-based authentication** - Just enter your name
- **Role selection** - User or Admin
- **Quick login** - Pre-filled demo accounts
- **No setup required** - Works immediately
- **Session persistence** - Stay logged in

### 📈 Comprehensive Analytics
- **Interactive charts** with Recharts
- **Weekly progress** tracking
- **Exercise distribution** visualization
- **Monthly goal** monitoring
- **Performance trends** analysis

### 🎯 Modern Gym-Themed UI
- **Vibrant color scheme** (Red/Teal/Orange)
- **Smooth animations** with Framer Motion
- **Responsive design** - Mobile to desktop
- **Dark theme** optimized for gyms
- **Real-time UI updates** with WebSockets ready

### 💪 User Dashboard
- **Personal workout stats**
- **Recent workout history**
- **Start workout** quick access
- **Profile management**
- **Progress analytics**

## 🏗️ System Architecture

```
┌──────────────────────────┐    ┌──────────────────────────┐
│   React 18 + Vite        │    │   Python Backend         │
│   Modern Frontend        │◄──►│   FastAPI Server         │
│                          │    │                          │
│ • Face Recognition       │    │ • MediaPipe Pose         │
│ • Real-Time Video        │    │ • ML Models (95%+ acc)   │
│ • TailwindCSS UI         │    │ • XGBoost Ensemble       │
│ • Framer Motion          │    │ • Rep Counters           │
│ • Recharts Analytics     │    │ • Posture Analysis       │
│ • Zustand State          │    │ • GPU Acceleration       │
└──────────────────────────┘    └──────────────────────────┘
         │                                   │
         └───────────── HTTP/WS ─────────────┘
```

### Tech Stack Details

#### Frontend
- **React 18** - Modern UI library
- **Vite** - Lightning-fast build tool
- **TailwindCSS** - Utility-first styling
- **Framer Motion** - Smooth animations
- **face-api.js** - Face recognition
- **React Webcam** - Camera access
- **Recharts** - Analytics visualization
- **Zustand** - State management
- **React Router** - Navigation
- **Axios** - HTTP client

#### Backend
- **Python 3.12** - Core language
- **FastAPI** - Modern web framework
- **MediaPipe** - Pose estimation
- **scikit-learn** - ML framework
- **XGBoost** - Gradient boosting
- **OpenCV** - Video processing
- **NumPy/Pandas** - Data handling

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+** (3.12 recommended)
- **Node.js 18+** and npm
- **Webcam** access
- **Modern browser** (Chrome/Firefox/Edge)
- **NVIDIA GPU** (optional, for faster training)

### 1️⃣ Clone Repository
```bash
git clone https://github.com/mpwin07/KPR_Hackathon.git
cd AlphaReps
```

### 2️⃣ Backend Setup

**Install Python Dependencies:**
```bash
pip install -r requirements.txt
```

**Install XGBoost (for 95%+ accuracy):**
```bash
pip install xgboost
```

**Prepare Dataset:**
Place exercise videos in `backend/dataset/`:
```
backend/dataset/
├── barbell_biceps_curl/  (10+ .mp4 videos)
├── hammer_curl/          (10+ .mp4 videos)
├── push_up/              (10+ .mp4 videos)
├── shoulder_press/       (10+ .mp4 videos)
└── squat/                (10+ .mp4 videos)
```

**Train the Premium Model:**
```bash
cd backend/scripts
python retrain_improved_model.py
```

This will:
- Process all videos (30 frames each)
- Extract 170 features per frame
- Apply 5x data augmentation
- Train 6-model ensemble (RF + ET + GB + Ada + SVM + XGBoost)
- Achieve **93-96% accuracy** (CPU) or **95-97%** (GPU)
- Takes 8-25 minutes depending on hardware

**Test the Model:**
```bash
cd backend
python start_workout.py
```

Or use the unified system directly:
```bash
cd backend
python unified_workout_system.py
```

Or use the integrated trainer:
```bash
cd backend
python integrated_trainer.py
```

### 3️⃣ Frontend Setup

**Install Dependencies:**
```bash
cd frontend
npm install
```

**Download Face Recognition Models:**

Create `frontend/public/models/` and download from:
https://github.com/justadudewhohacks/face-api.js-models

Or use the quick download script in `frontend/SETUP.md`

**Start Development Server:**
```bash
npm run dev
```

Frontend runs on: **http://localhost:3000**

### 4️⃣ Access the App

1. **Open**: http://localhost:3000
2. **Register**: Enter name and capture face
3. **Login**: Face recognition auto-login
4. **Start Workout**: Click "Start Workout" and begin!

---

## 📖 Documentation

- **Frontend Setup**: `frontend/SETUP.md` - Detailed frontend guide
- **Model Accuracy**: `MODEL_ACCURACY_GUIDE.md` - AI model details
- **Quick Start**: `QUICK_START_PREMIUM.md` - Fast setup guide
- **Integrated Trainer**: `INTEGRATED_TRAINER_GUIDE.md` - Usage guide

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
├── 📱 frontend/                         # React Frontend
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Landing.jsx              # 🏠 Landing page
│   │   │   ├── FaceLogin.jsx            # 🔐 Simple login
│   │   │   ├── admin/
│   │   │   │   └── AdminDashboard.jsx   # 👨‍💼 Gym owner dashboard
│   │   │   └── user/
│   │   │       ├── UserDashboard.jsx    # 💪 User home
│   │   │       ├── WorkoutSession.jsx   # 🎥 Main workout interface
│   │   │       ├── Profile.jsx          # 👤 User profile
│   │   │       └── Analytics.jsx        # 📊 Progress charts
│   │   ├── store/
│   │   │   └── authStore.js             # State management
│   │   ├── App.jsx                      # Router & routes
│   │   ├── main.jsx                     # Entry point
│   │   └── index.css                    # Styles
│   ├── package.json
│   ├── vite.config.js
│   └── SETUP.md                         # Frontend setup guide
│
├── 🐍 backend/                          # Python Backend
│   ├── start_workout.py                 # 🚀 Main entry point
│   ├── unified_workout_system.py        # ⭐ Core workout system
│   ├── main.py                          # FastAPI server
│   │
│   ├── models/
│   │   └── video_exercise_classifier.py # 🤖 AI exercise classifier
│   │
│   ├── rep_counters/
│   │   ├── __init__.py
│   │   ├── base_counter.py              # Base counter class
│   │   ├── pushup_counter.py            # ✅ With posture correction
│   │   ├── squat_counter.py             # Squat counter
│   │   ├── curl_counter.py              # Bicep/hammer curl counter
│   │   └── shoulder_press_counter.py    # Shoulder press counter
│   │
│   └── scripts/
│       ├── __init__.py
│       └── train_video_model.py         # 📚 Model training script
│
├── 📄 Documentation
│   ├── README.md                        # Main documentation
│   └── requirements.txt                 # Python dependencies
│
└── .gitignore                           # Git ignore rules
```

### 🎯 Key Files

| File | Purpose |
|------|---------|
| `backend/start_workout.py` | **Main entry point** - Start here! |
| `backend/unified_workout_system.py` | Complete workout system with AI |
| `backend/models/video_exercise_classifier.py` | Exercise detection AI |
| `backend/scripts/train_video_model.py` | Train the AI model |
| `frontend/src/pages/user/WorkoutSession.jsx` | Web workout interface |

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

## 📸 Screenshots

### Landing Page
- Vibrant hero section with gradient animations
- Feature highlights
- Responsive design

### Face Login
- Real-time face detection
- Visual feedback
- Smooth transitions

### Workout Session
- Live video feed with pose overlay
- Real-time exercise detection
- Rep counter and form feedback
- Angle visualization

### User Dashboard
- Personal statistics
- Recent workouts
- Quick actions

### Admin Dashboard
- Member management
- Performance analytics
- Top performers leaderboard

## 🔒 Privacy & Security

- **Local Processing**: All AI analysis happens locally
- **Secure Storage**: Face descriptors (not images) encrypted
- **No Video Recording**: Only pose landmarks stored
- **HTTPS Ready**: Secure webcam access
- **Role-Based Access**: Admin vs User permissions

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

## 🎯 Key Highlights

### For Gym Owners
- 👨‍💼 **Admin Dashboard** - Manage all members
- 📊 **Analytics** - Track gym-wide performance
- 🏆 **Leaderboards** - Motivate members
- 👤 **Face Auth** - No more lost gym cards

### For Gym Members
- 🎯 **Auto Detection** - Just start exercising
- ✅ **Form Correction** - Real-time feedback
- 📈 **Progress Tracking** - See improvements
- 🏅 **Achievements** - Stay motivated

### For Developers
- ⚡ **Modern Stack** - React + FastAPI
- 🎨 **Beautiful UI** - TailwindCSS + Framer Motion
- 🤖 **95%+ AI** - Production-ready models
- 📱 **Responsive** - Works on all devices

## 🚀 Performance

- **Exercise Detection**: < 100ms per frame
- **Face Recognition**: < 2 seconds
- **Model Inference**: Real-time (30+ FPS)
- **Frontend Load**: < 3 seconds
- **Model Accuracy**: 93-97%

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

MIT License - See LICENSE file for details

## 👥 Team

Built for **KPR Hackathon** by Team AlphaReps

## 🙏 Acknowledgments

- **MediaPipe** - For amazing pose detection
- **face-api.js** - For face recognition
- **TailwindCSS** - For beautiful styling
- **Framer Motion** - For smooth animations
- **XGBoost** - For ML excellence

---

<div align="center">

### ⭐ Star this repo if you found it helpful!

**Built with ❤️ and 💪 for the fitness community**

[Report Bug](https://github.com/mpwin07/KPR_Hackathon/issues) • [Request Feature](https://github.com/mpwin07/KPR_Hackathon/issues)

</div>
