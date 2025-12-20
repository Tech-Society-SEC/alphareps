"""
Improved pushup counter with FSM and angle-based logic
"""
from .base_counter import BaseRepCounter
import numpy as np

class PushupCounter(BaseRepCounter):
    """
    Pushup counter using:
    - Elbow angle as signature angle
    - FSM for stable counting
    - Form feedback (back angle, elbow flare)
    """
    
    def __init__(self):
        super().__init__(
            history_size=3,
            min_transition_frames=1,
            rep_debounce_seconds=0.05  # Very fast response
        )
        self.confidence_threshold = 0.2  # More permissive visibility
        self.mp_pose = __import__('mediapipe').solutions.pose
        self.feedback = ""
        self.form_quality = "GOOD"
        self.back_angle = 0
        
    def get_angle_thresholds(self):
        """
        Pushup thresholds (LIBERAL for responsiveness):
        - Down (bottom): < 120° (elbows bent - more forgiving)
        - Up (top): > 130° (arms extended - easier to trigger)
        """
        return 120, 130  # Liberal thresholds
    
    def get_signature_angle(self, landmarks):
        """
        Get elbow angle for pushups
        Uses both arms and averages
        """
        try:
            # Get landmarks for both arms
            left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            left_elbow = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
            left_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
            
            right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            right_elbow = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            right_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
            
            # Calculate visibility
            left_vis = (left_shoulder.visibility + left_elbow.visibility + left_wrist.visibility) / 3
            right_vis = (right_shoulder.visibility + right_elbow.visibility + right_wrist.visibility) / 3
            
            angles = {}
            
            if left_vis > self.confidence_threshold:
                angles["left"] = self.calculate_angle(
                    [left_shoulder.x, left_shoulder.y],
                    [left_elbow.x, left_elbow.y],
                    [left_wrist.x, left_wrist.y]
                )
            
            if right_vis > self.confidence_threshold:
                angles["right"] = self.calculate_angle(
                    [right_shoulder.x, right_shoulder.y],
                    [right_elbow.x, right_elbow.y],
                    [right_wrist.x, right_wrist.y]
                )
            
            if len(angles) == 0:
                return None
            
            # Choose better arm (higher visibility)
            if left_vis >= right_vis:
                return angles.get("left")
            return angles.get("right")
            
        except Exception as e:
            print(f"Error getting pushup angle: {e}")
            return None
    
    def check_form(self, landmarks):
        """
        Check pushup form and provide feedback
        Returns: (feedback, back_angle, form_quality)
        """
        try:
            # Get landmarks for back alignment
            left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            left_hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP.value]
            left_ankle = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
            
            right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            right_hip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
            right_ankle = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
            
            # Calculate back angle (should be straight ~170-180°)
            left_back = self.calculate_angle(
                [left_shoulder.x, left_shoulder.y],
                [left_hip.x, left_hip.y],
                [left_ankle.x, left_ankle.y]
            )
            right_back = self.calculate_angle(
                [right_shoulder.x, right_shoulder.y],
                [right_hip.x, right_hip.y],
                [right_ankle.x, right_ankle.y]
            )
            
            self.back_angle = (left_back + right_back) / 2
            
            # Form feedback
            issues = []
            
            if self.back_angle < 160:
                if self.back_angle < 145:
                    issues.append("STRAIGHTEN BACK - HIPS SAGGING")
                else:
                    issues.append("STRAIGHTEN BACK")
            elif self.back_angle > 175:
                issues.append("LOWER HIPS")
            
            if issues:
                self.feedback = " | ".join(issues)
                self.form_quality = "POOR"
            else:
                self.feedback = "PERFECT FORM"
                self.form_quality = "GOOD"
            
            return self.feedback, self.back_angle, self.form_quality
            
        except Exception as e:
            print(f"Error checking form: {e}")
            return "CHECK FORM", 0, "UNKNOWN"
    
    def process_frame(self, landmarks):
        """
        Process frame with form feedback
        Returns clean JSON with form data
        """
        # Get base counter result
        base_result = super().process_frame(landmarks)
        
        # Add form feedback
        feedback, back_angle, form_quality = self.check_form(landmarks)
        
        # Enhance result with pushup-specific data
        base_result["form_feedback"] = feedback
        base_result["back_angle"] = round(back_angle, 1) if back_angle else 0
        base_result["form_quality"] = form_quality
        
        return base_result
    
    def count_rep(self, landmarks):
        """
        Legacy method for backward compatibility
        Returns: (rep_count, stage, elbow_angle, feedback, back_angle, form_quality)
        """
        result = self.process_frame(landmarks)
        return (
            result["reps"],
            result["state"],
            result["angle"],
            result.get("form_feedback", ""),
            result.get("back_angle", 0),
            result.get("form_quality", "GOOD")
        )
