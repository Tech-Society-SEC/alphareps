"""
Rep counter for squats
"""
from .base_counter import BaseRepCounter

class SquatCounter(BaseRepCounter):
    """Enhanced counter for squats with dual-leg detection"""
    
    def __init__(self):
        super().__init__()
        self.left_angle_history = []
        self.right_angle_history = []
        self.history_size = 5  # Increased for better smoothing
        self.min_down_angle = 95  # More lenient - was too strict at 90
        self.min_up_angle = 155   # More lenient - was too strict at 160
        self.debounce_frames = 2  # Slight debouncing to prevent double counting
        self.frames_in_position = 0
        self.primary_leg = 'right'  # Default to right leg
        self.confidence_threshold = 0.7  # Minimum visibility for reliable detection
        
    def count_rep(self, landmarks):
        """
        Enhanced squat counting with dual-leg detection and better smoothing
        Returns: (rep_count, stage, angle)
        """
        try:
            # Get landmarks for both legs
            left_hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP.value]
            left_knee = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
            left_ankle = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
            
            right_hip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
            right_knee = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
            right_ankle = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
            
            # Check visibility and choose primary leg
            left_visible = all(lm.visibility > self.confidence_threshold for lm in [left_hip, left_knee, left_ankle])
            right_visible = all(lm.visibility > self.confidence_threshold for lm in [right_hip, right_knee, right_ankle])
            
            # Calculate angles for both legs
            left_angle = None
            right_angle = None
            
            if left_visible:
                left_angle = self.calculate_angle(
                    [left_hip.x, left_hip.y],
                    [left_knee.x, left_knee.y],
                    [left_ankle.x, left_ankle.y]
                )
                self.left_angle_history.append(left_angle)
                if len(self.left_angle_history) > self.history_size:
                    self.left_angle_history.pop(0)
            
            if right_visible:
                right_angle = self.calculate_angle(
                    [right_hip.x, right_hip.y],
                    [right_knee.x, right_knee.y],
                    [right_ankle.x, right_ankle.y]
                )
                self.right_angle_history.append(right_angle)
                if len(self.right_angle_history) > self.history_size:
                    self.right_angle_history.pop(0)
            
            # Choose the leg with better visibility or use average of both
            primary_angle = None
            primary_history = []
            
            if right_visible and left_visible:
                # Use average of both legs for more stability
                left_smoothed = sum(self.left_angle_history) / len(self.left_angle_history) if self.left_angle_history else left_angle
                right_smoothed = sum(self.right_angle_history) / len(self.right_angle_history) if self.right_angle_history else right_angle
                primary_angle = (left_smoothed + right_smoothed) / 2
                primary_history = [(l + r) / 2 for l, r in zip(self.left_angle_history, self.right_angle_history) if l and r]
                self.primary_leg = 'both'
            elif right_visible:
                primary_angle = right_angle
                primary_history = self.right_angle_history
                self.primary_leg = 'right'
            elif left_visible:
                primary_angle = left_angle
                primary_history = self.left_angle_history
                self.primary_leg = 'left'
            else:
                return self.counter, self.stage, None
            
            # Smooth the primary angle
            if primary_history:
                smoothed_angle = sum(primary_history) / len(primary_history)
            else:
                smoothed_angle = primary_angle
            
            # Enhanced rep counting logic with better state management
            if smoothed_angle > self.min_up_angle:
                if self.stage != "up":
                    self.stage = "up"
                    self.frames_in_position = 0
                else:
                    self.frames_in_position += 1
                    
            elif smoothed_angle < self.min_down_angle:
                if self.stage == "up" and self.frames_in_position >= self.debounce_frames:
                    self.stage = "down"
                    self.counter += 1
                    self.frames_in_position = 0
                elif self.stage != "down":
                    self.stage = "down"
                    self.frames_in_position = 0
            
            return self.counter, self.stage, int(smoothed_angle)
            
        except Exception as e:
            print(f"Error in enhanced squat counter: {e}")
            return self.counter, self.stage, None
