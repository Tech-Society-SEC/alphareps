"""
Frontend-friendly base class for exercise rep counters
Clean, stable, debounced output every frame
"""
import numpy as np
import mediapipe as mp
from abc import ABC, abstractmethod
from collections import deque
from scipy.signal import savgol_filter
import time

class BaseRepCounter(ABC):
    """
    Frontend-optimized rep counter with:
    - Clean JSON output every frame
    - Debounced rep increments
    - Stable state transitions
    - No console spam
    """
    
    def __init__(self, 
                 history_size=3,
                 min_transition_frames=1,
                 angle_smooth_window=3,
                 rep_debounce_seconds=0.1):
        """
        Args:
            history_size: Size of angle history buffer
            min_transition_frames: Minimum frames required to confirm state transition
            angle_smooth_window: Window size for angle smoothing (must be odd)
            rep_debounce_seconds: Minimum time between rep counts (prevents double counting)
        """
        self.counter = 0
        self.state = "idle"  # Simple string states: idle, up, down
        self.mp_pose = mp.solutions.pose
        
        # Smoothing parameters
        self.history_size = history_size
        self.min_transition_frames = min_transition_frames
        self.angle_smooth_window = angle_smooth_window if angle_smooth_window % 2 == 1 else angle_smooth_window + 1
        self.rep_debounce_seconds = rep_debounce_seconds
        
        # Buffers for smoothing
        self.angle_history = deque(maxlen=history_size)
        self.state_history = deque(maxlen=2)  # Minimal state smoothing for responsiveness
        
        # State tracking
        self.frames_in_current_state = 0
        self.last_rep_time = 0
        self.previous_reps = 0
        
        # Thresholds (to be set by subclasses)
        self.down_angle_threshold = None
        self.up_angle_threshold = None
        
    def calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle
    
    def smooth_angle(self, angle):
        """Smooth angle using rolling window average"""
        if angle is None:
            return None
            
        self.angle_history.append(angle)
        
        if len(self.angle_history) < 2:
            return angle
        
        # Use only last 2-3 values for fast response
        return np.mean(list(self.angle_history)[-3:])
    
    def smooth_state(self, state):
        """Smooth state using majority voting"""
        self.state_history.append(state)
        
        if len(self.state_history) < 2:
            return state
        
        # Majority voting
        from collections import Counter
        counts = Counter(self.state_history)
        return counts.most_common(1)[0][0]
    
    @abstractmethod
    def get_signature_angle(self, landmarks):
        """
        Get the signature angle for this exercise
        Must be implemented by subclasses
        
        Args:
            landmarks: MediaPipe pose landmarks
        
        Returns:
            angle (float): The signature angle for this exercise
        """
        pass
    
    @abstractmethod
    def get_angle_thresholds(self):
        """
        Get the angle thresholds for this exercise
        Must be implemented by subclasses
        
        Returns:
            tuple: (down_threshold, up_threshold)
        """
        pass
    
    def update_fsm(self, smoothed_angle):
        """
        Update FSM and return if rep was incremented
        
        Args:
            smoothed_angle: Current smoothed angle
        
        Returns:
            bool: True if rep was incremented, False otherwise
        """
        if smoothed_angle is None:
            return False
        
        # Get thresholds if not set
        if self.down_angle_threshold is None or self.up_angle_threshold is None:
            self.down_angle_threshold, self.up_angle_threshold = self.get_angle_thresholds()
        
        rep_incremented = False
        previous_state = self.state
        
        # Simplified 3-State FSM: idle -> up -> down -> up (rep!) -> down
        if self.state == "idle":
            # Initialize to UP state if angle is high enough
            if smoothed_angle > self.up_angle_threshold:
                self.state = "up"
                self.frames_in_current_state = 0
            # Or DOWN if angle is low
            elif smoothed_angle < self.down_angle_threshold:
                self.state = "down"
                self.frames_in_current_state = 0
                
        elif self.state == "up":
            # At top position, wait for downward movement
            if smoothed_angle < self.down_angle_threshold:
                self.frames_in_current_state += 1
                if self.frames_in_current_state >= self.min_transition_frames:
                    self.state = "down"
                    self.frames_in_current_state = 0
            else:
                # Still in UP range, reset counter
                self.frames_in_current_state = 0
                
        elif self.state == "down":
            # At bottom position, wait for upward movement
            if smoothed_angle > self.up_angle_threshold:
                self.frames_in_current_state += 1
                if self.frames_in_current_state >= self.min_transition_frames:
                    # Check debounce before counting rep
                    current_time = time.time()
                    if current_time - self.last_rep_time > self.rep_debounce_seconds:
                        # Complete rep!
                        self.counter += 1
                        self.last_rep_time = current_time
                        rep_incremented = True
                    
                    self.state = "up"
                    self.frames_in_current_state = 0
            else:
                # Still in DOWN range, reset counter
                self.frames_in_current_state = 0
        
        return rep_incremented
    
    def process_frame(self, landmarks):
        """
        Process a single frame and return frontend-friendly JSON
        
        Args:
            landmarks: MediaPipe pose landmarks
        
        Returns:
            dict: Clean JSON packet with all UI needs
        """
        try:
            # Get signature angle
            raw_angle = self.get_signature_angle(landmarks)
            
            # Smooth the angle
            smoothed_angle = self.smooth_angle(raw_angle)
            
            # Store previous state
            previous_state = self.state
            previous_reps = self.counter
            
            # Update FSM
            rep_incremented = self.update_fsm(smoothed_angle)
            
            # Smooth state for UI stability
            display_state = self.smooth_state(self.state)
            
            # Return clean JSON packet
            return {
                "success": True,
                "angle": round(smoothed_angle, 1) if smoothed_angle is not None else None,
                "raw_angle": round(raw_angle, 1) if raw_angle is not None else None,
                "state": display_state,
                "previous_state": previous_state,
                "reps": self.counter,
                "rep_incremented": rep_incremented,
                "frames_in_state": self.frames_in_current_state,
                "error": None
            }
            
        except Exception as e:
            # Return error state without console spam
            return {
                "success": False,
                "angle": None,
                "raw_angle": None,
                "state": self.state,
                "previous_state": self.state,
                "reps": self.counter,
                "rep_incremented": False,
                "frames_in_state": 0,
                "error": str(e)
            }
    
    # Legacy compatibility methods
    def count_rep(self, landmarks):
        """
        Legacy method for backward compatibility
        Returns: (rep_count, state, angle)
        """
        result = self.process_frame(landmarks)
        return result["reps"], result["state"], result["angle"]
    
    def reset(self):
        """Reset counter and state"""
        self.counter = 0
        self.state = "idle"
        self.frames_in_current_state = 0
        self.last_rep_time = 0
        self.previous_reps = 0
        self.angle_history.clear()
        self.state_history.clear()
    
    def get_count(self):
        """Get current count"""
        return self.counter
    
    def get_stage(self):
        """Get current stage"""
        return self.state
