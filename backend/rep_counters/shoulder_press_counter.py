"""
Rep counter for shoulder press
"""
import numpy as np
from .base_counter import BaseRepCounter

class ShoulderPressCounter(BaseRepCounter):
    """Enhanced counter for shoulder press with dual-arm detection"""
    
    def __init__(self):
        super().__init__()
        self.left_angle_history = []
        self.right_angle_history = []
        self.history_size = 5  # Increased for better smoothing
        self.min_down_angle = 85   # More lenient - was too strict at 90
        self.min_up_angle = 145    # More lenient - was too strict at 140
        self.debounce_frames = 2   # Slight debouncing to prevent double counting
        self.frames_in_position = 0
        self.left_stage = None
        self.right_stage = None
        self.primary_arm = 'right'  # Default to right arm
        self.confidence_threshold = 0.7  # Minimum visibility for reliable detection
        
    def count_rep(self, landmarks):
        """
        Enhanced shoulder press counting with dual-arm detection and better smoothing
        Returns: (rep_count, stage, angle)
        """
        try:
            # Get landmarks for both arms
            left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            left_elbow = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
            left_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
            
            right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            right_elbow = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            right_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
            
            # Check visibility and choose primary arm
            left_visible = all(lm.visibility > self.confidence_threshold for lm in [left_shoulder, left_elbow, left_wrist])
            right_visible = all(lm.visibility > self.confidence_threshold for lm in [right_shoulder, right_elbow, right_wrist])
            
            # Calculate angles for both arms
            left_angle = None
            right_angle = None
            
            if left_visible:
                left_angle = self.calculate_angle(
                    [left_shoulder.x, left_shoulder.y],
                    [left_elbow.x, left_elbow.y],
                    [left_wrist.x, left_wrist.y]
                )
                self.left_angle_history.append(left_angle)
                if len(self.left_angle_history) > self.history_size:
                    self.left_angle_history.pop(0)
            
            if right_visible:
                right_angle = self.calculate_angle(
                    [right_shoulder.x, right_shoulder.y],
                    [right_elbow.x, right_elbow.y],
                    [right_wrist.x, right_wrist.y]
                )
                self.right_angle_history.append(right_angle)
                if len(self.right_angle_history) > self.history_size:
                    self.right_angle_history.pop(0)
            
            # Choose the arm with better visibility or more movement
            primary_angle = None
            primary_history = []
            
            if right_visible and left_visible:
                # Use the arm with more variation (more likely to be exercising)
                right_variation = np.std(self.right_angle_history) if self.right_angle_history else 0
                left_variation = np.std(self.left_angle_history) if self.left_angle_history else 0
                
                if right_variation > left_variation:
                    primary_angle = right_angle
                    primary_history = self.right_angle_history
                    self.primary_arm = 'right'
                else:
                    primary_angle = left_angle
                    primary_history = self.left_angle_history
                    self.primary_arm = 'left'
            elif right_visible:
                primary_angle = right_angle
                primary_history = self.right_angle_history
                self.primary_arm = 'right'
            elif left_visible:
                primary_angle = left_angle
                primary_history = self.left_angle_history
                self.primary_arm = 'left'
            else:
                return self.counter, self.stage, None
            
            # Smooth the primary angle
            if primary_history:
                smoothed_angle = sum(primary_history) / len(primary_history)
            else:
                smoothed_angle = primary_angle
            
            # Enhanced rep counting logic with better state management
            if smoothed_angle < self.min_down_angle:
                if self.stage != "down":
                    self.stage = "down"
                    self.frames_in_position = 0
                else:
                    self.frames_in_position += 1
                    
            elif smoothed_angle > self.min_up_angle:
                if self.stage == "down" and self.frames_in_position >= self.debounce_frames:
                    self.stage = "up"
                    self.counter += 1
                    self.frames_in_position = 0
                elif self.stage != "up":
                    self.stage = "up"
                    self.frames_in_position = 0
            
            return self.counter, self.stage, int(smoothed_angle)
            
        except Exception as e:
            print(f"Error in enhanced shoulder press counter: {e}")
            return self.counter, self.stage, None
