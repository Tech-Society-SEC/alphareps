import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import sqlite3
import aiosqlite
from dataclasses import dataclass, asdict

@dataclass
class User:
    id: int
    name: str
    email: str
    created_at: str
    last_login: str

@dataclass
class WorkoutSession:
    id: int
    user_id: int
    exercise_type: str
    rep_count: int
    duration_seconds: int
    calories_burned: float
    form_score: float
    created_at: str

@dataclass
class AttendanceRecord:
    id: int
    user_name: str
    login_time: str
    logout_time: Optional[str]

class DatabaseManager:
    def __init__(self, db_path: str = "data/alpharep.db"):
        self.db_path = db_path
        
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    async def initialize(self):
        """Initialize database tables"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Users table
                await db.execute('''
                    CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL UNIQUE,
                        email TEXT UNIQUE,
                        created_at TEXT NOT NULL,
                        last_login TEXT
                    )
                ''')
                
                # Workout sessions table
                await db.execute('''
                    CREATE TABLE IF NOT EXISTS workout_sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER,
                        exercise_type TEXT NOT NULL,
                        rep_count INTEGER DEFAULT 0,
                        duration_seconds INTEGER DEFAULT 0,
                        calories_burned REAL DEFAULT 0.0,
                        form_score REAL DEFAULT 0.0,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (user_id) REFERENCES users (id)
                    )
                ''')
                
                # Attendance records table
                await db.execute('''
                    CREATE TABLE IF NOT EXISTS attendance_records (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_name TEXT NOT NULL,
                        login_time TEXT NOT NULL,
                        logout_time TEXT
                    )
                ''')
                
                # Exercise statistics table
                await db.execute('''
                    CREATE TABLE IF NOT EXISTS exercise_stats (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER,
                        exercise_type TEXT NOT NULL,
                        total_reps INTEGER DEFAULT 0,
                        total_sessions INTEGER DEFAULT 0,
                        total_duration_seconds INTEGER DEFAULT 0,
                        total_calories REAL DEFAULT 0.0,
                        average_form_score REAL DEFAULT 0.0,
                        best_form_score REAL DEFAULT 0.0,
                        last_updated TEXT NOT NULL,
                        FOREIGN KEY (user_id) REFERENCES users (id)
                    )
                ''')
                
                await db.commit()
                print("✅ Database initialized successfully")
                
        except Exception as e:
            print(f"❌ Error initializing database: {e}")
    
    async def create_user(self, name: str, email: str = None) -> Optional[int]:
        """Create a new user"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(
                    'INSERT INTO users (name, email, created_at) VALUES (?, ?, ?)',
                    (name, email, datetime.now().isoformat())
                )
                await db.commit()
                user_id = cursor.lastrowid
                print(f"✅ Created user: {name} (ID: {user_id})")
                return user_id
                
        except Exception as e:
            print(f"❌ Error creating user: {e}")
            return None
    
    async def get_user_by_name(self, name: str) -> Optional[User]:
        """Get user by name"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(
                    'SELECT * FROM users WHERE name = ?', (name,)
                )
                row = await cursor.fetchone()
                
                if row:
                    return User(
                        id=row[0],
                        name=row[1],
                        email=row[2],
                        created_at=row[3],
                        last_login=row[4]
                    )
                return None
                
        except Exception as e:
            print(f"❌ Error getting user: {e}")
            return None
    
    async def update_last_login(self, user_name: str):
        """Update user's last login time"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    'UPDATE users SET last_login = ? WHERE name = ?',
                    (datetime.now().isoformat(), user_name)
                )
                await db.commit()
                
        except Exception as e:
            print(f"❌ Error updating last login: {e}")
    
    async def log_attendance(self, user_name: str):
        """Log user attendance"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    'INSERT INTO attendance_records (user_name, login_time) VALUES (?, ?)',
                    (user_name, datetime.now().isoformat())
                )
                await db.commit()
                
                # Update user's last login
                await self.update_last_login(user_name)
                print(f"✅ Logged attendance for {user_name}")
                
        except Exception as e:
            print(f"❌ Error logging attendance: {e}")
    
    async def save_workout(self, user_id: int, workout_data: Dict[str, Any]):
        """Save workout session"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Insert workout session
                await db.execute('''
                    INSERT INTO workout_sessions 
                    (user_id, exercise_type, rep_count, duration_seconds, calories_burned, form_score, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    user_id,
                    workout_data.get('exercise_type', ''),
                    workout_data.get('rep_count', 0),
                    workout_data.get('duration_seconds', 0),
                    workout_data.get('calories_burned', 0.0),
                    workout_data.get('form_score', 0.0),
                    datetime.now().isoformat()
                ))
                
                # Update exercise statistics
                await self.update_exercise_stats(user_id, workout_data)
                
                await db.commit()
                print(f"✅ Saved workout for user {user_id}")
                
        except Exception as e:
            print(f"❌ Error saving workout: {e}")
    
    async def update_exercise_stats(self, user_id: int, workout_data: Dict[str, Any]):
        """Update exercise statistics"""
        try:
            exercise_type = workout_data.get('exercise_type', '')
            rep_count = workout_data.get('rep_count', 0)
            duration = workout_data.get('duration_seconds', 0)
            calories = workout_data.get('calories_burned', 0.0)
            form_score = workout_data.get('form_score', 0.0)
            
            async with aiosqlite.connect(self.db_path) as db:
                # Check if stats exist for this user and exercise
                cursor = await db.execute(
                    'SELECT * FROM exercise_stats WHERE user_id = ? AND exercise_type = ?',
                    (user_id, exercise_type)
                )
                existing_stats = await cursor.fetchone()
                
                if existing_stats:
                    # Update existing stats
                    new_total_reps = existing_stats[2] + rep_count
                    new_total_sessions = existing_stats[3] + 1
                    new_total_duration = existing_stats[4] + duration
                    new_total_calories = existing_stats[5] + calories
                    new_avg_form_score = (existing_stats[6] * existing_stats[3] + form_score) / new_total_sessions
                    new_best_form_score = max(existing_stats[7], form_score)
                    
                    await db.execute('''
                        UPDATE exercise_stats SET
                        total_reps = ?, total_sessions = ?, total_duration_seconds = ?,
                        total_calories = ?, average_form_score = ?, best_form_score = ?,
                        last_updated = ?
                        WHERE user_id = ? AND exercise_type = ?
                    ''', (
                        new_total_reps, new_total_sessions, new_total_duration,
                        new_total_calories, new_avg_form_score, new_best_form_score,
                        datetime.now().isoformat(), user_id, exercise_type
                    ))
                else:
                    # Create new stats
                    await db.execute('''
                        INSERT INTO exercise_stats
                        (user_id, exercise_type, total_reps, total_sessions, total_duration_seconds,
                         total_calories, average_form_score, best_form_score, last_updated)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        user_id, exercise_type, rep_count, 1, duration,
                        calories, form_score, form_score, datetime.now().isoformat()
                    ))
                
        except Exception as e:
            print(f"❌ Error updating exercise stats: {e}")
    
    async def get_user_stats(self, user_id: int) -> Dict[str, Any]:
        """Get comprehensive user statistics"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Get overall stats
                cursor = await db.execute('''
                    SELECT 
                        COUNT(*) as total_sessions,
                        SUM(rep_count) as total_reps,
                        SUM(duration_seconds) as total_duration,
                        SUM(calories_burned) as total_calories,
                        AVG(form_score) as avg_form_score
                    FROM workout_sessions 
                    WHERE user_id = ?
                ''', (user_id,))
                overall_stats = await cursor.fetchone()
                
                # Get exercise-specific stats
                cursor = await db.execute(
                    'SELECT * FROM exercise_stats WHERE user_id = ?', (user_id,)
                )
                exercise_stats = await cursor.fetchall()
                
                # Get recent sessions (last 10)
                cursor = await db.execute('''
                    SELECT * FROM workout_sessions 
                    WHERE user_id = ? 
                    ORDER BY created_at DESC 
                    LIMIT 10
                ''', (user_id,))
                recent_sessions = await cursor.fetchall()
                
                # Get weekly progress (last 7 days)
                week_ago = (datetime.now() - timedelta(days=7)).isoformat()
                cursor = await db.execute('''
                    SELECT 
                        DATE(created_at) as date,
                        COUNT(*) as sessions,
                        SUM(rep_count) as reps,
                        SUM(calories_burned) as calories
                    FROM workout_sessions 
                    WHERE user_id = ? AND created_at >= ?
                    GROUP BY DATE(created_at)
                    ORDER BY date
                ''', (user_id, week_ago))
                weekly_progress = await cursor.fetchall()
                
                return {
                    "overall_stats": {
                        "total_sessions": overall_stats[0] or 0,
                        "total_reps": overall_stats[1] or 0,
                        "total_duration_minutes": (overall_stats[2] or 0) / 60,
                        "total_calories": overall_stats[3] or 0.0,
                        "average_form_score": overall_stats[4] or 0.0
                    },
                    "exercise_stats": [
                        {
                            "exercise_type": stat[2],
                            "total_reps": stat[3],
                            "total_sessions": stat[4],
                            "total_duration_minutes": stat[5] / 60,
                            "total_calories": stat[6],
                            "average_form_score": stat[7],
                            "best_form_score": stat[8]
                        }
                        for stat in exercise_stats
                    ],
                    "recent_sessions": [
                        {
                            "exercise_type": session[2],
                            "rep_count": session[3],
                            "duration_minutes": session[4] / 60,
                            "calories_burned": session[5],
                            "form_score": session[6],
                            "date": session[7]
                        }
                        for session in recent_sessions
                    ],
                    "weekly_progress": [
                        {
                            "date": day[0],
                            "sessions": day[1],
                            "reps": day[2],
                            "calories": day[3]
                        }
                        for day in weekly_progress
                    ]
                }
                
        except Exception as e:
            print(f"❌ Error getting user stats: {e}")
            return {}
    
    async def get_leaderboard(self, exercise_type: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get leaderboard for exercises"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                if exercise_type:
                    cursor = await db.execute('''
                        SELECT u.name, es.total_reps, es.best_form_score, es.total_sessions
                        FROM exercise_stats es
                        JOIN users u ON es.user_id = u.id
                        WHERE es.exercise_type = ?
                        ORDER BY es.total_reps DESC
                        LIMIT ?
                    ''', (exercise_type, limit))
                else:
                    cursor = await db.execute('''
                        SELECT u.name, SUM(es.total_reps) as total_reps, 
                               AVG(es.average_form_score) as avg_form_score,
                               SUM(es.total_sessions) as total_sessions
                        FROM exercise_stats es
                        JOIN users u ON es.user_id = u.id
                        GROUP BY u.id
                        ORDER BY total_reps DESC
                        LIMIT ?
                    ''', (limit,))
                
                leaderboard = await cursor.fetchall()
                
                return [
                    {
                        "name": entry[0],
                        "total_reps": entry[1],
                        "form_score": entry[2],
                        "total_sessions": entry[3]
                    }
                    for entry in leaderboard
                ]
                
        except Exception as e:
            print(f"❌ Error getting leaderboard: {e}")
            return []
    
    async def cleanup_old_data(self, days_old: int = 90):
        """Clean up old data (older than specified days)"""
        try:
            cutoff_date = (datetime.now() - timedelta(days=days_old)).isoformat()
            
            async with aiosqlite.connect(self.db_path) as db:
                # Delete old workout sessions
                await db.execute(
                    'DELETE FROM workout_sessions WHERE created_at < ?', (cutoff_date,)
                )
                
                # Delete old attendance records
                await db.execute(
                    'DELETE FROM attendance_records WHERE login_time < ?', (cutoff_date,)
                )
                
                await db.commit()
                print(f"✅ Cleaned up data older than {days_old} days")
                
        except Exception as e:
            print(f"❌ Error cleaning up old data: {e}")

# Example usage and testing
if __name__ == "__main__":
    async def test_database():
        db = DatabaseManager()
        await db.initialize()
        
        # Create test user
        user_id = await db.create_user("Test User", "test@example.com")
        
        if user_id:
            # Save test workout
            workout_data = {
                "exercise_type": "pushup",
                "rep_count": 20,
                "duration_seconds": 120,
                "calories_burned": 15.5,
                "form_score": 85.0
            }
            
            await db.save_workout(user_id, workout_data)
            
            # Get user stats
            stats = await db.get_user_stats(user_id)
            print("User Stats:", json.dumps(stats, indent=2))
            
            # Get leaderboard
            leaderboard = await db.get_leaderboard()
            print("Leaderboard:", json.dumps(leaderboard, indent=2))
    
    # Run test
    asyncio.run(test_database())
