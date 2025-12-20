"""
Improved squat counter with FSM, angle-based logic, and form feedback
"""
from .base_counter import BaseRepCounter
import numpy as np

class SquatCounter(BaseRepCounter):
    """
    Squat counter using:
    - Knee angle as signature angle
    - FSM for stable counting
    - Dual-leg detection with fallback
    - Form feedback for proper squat technique
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
        self.knee_over_toe = False
        
    def get_angle_thresholds(self):
        """
        Squat thresholds (LIBERAL for responsiveness):
        - Down (bottom): < 130° (deep squat - more forgiving)
        - Up (top): > 150° (standing - easier to trigger)
        """
        return 130, 150  # Liberal thresholds
    
    def get_signature_angle(self, landmarks):
        """
        Get knee angle for squats
        Uses both legs and averages if both visible
        """
        try:
            # Get landmarks for both legs
            left_hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP.value]
            left_knee = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
            left_ankle = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
            
            right_hip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
            right_knee = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
            right_ankle = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
            
            # Calculate visibility
            left_vis = (left_hip.visibility + left_knee.visibility + left_ankle.visibility) / 3
            right_vis = (right_hip.visibility + right_knee.visibility + right_ankle.visibility) / 3
            
            angles = {}
            
            if left_vis > self.confidence_threshold:
                angles["left"] = self.calculate_angle(
                    [left_hip.x, left_hip.y],
                    [left_knee.x, left_knee.y],
                    [left_ankle.x, left_ankle.y]
                )
            
            if right_vis > self.confidence_threshold:
                angles["right"] = self.calculate_angle(
                    [right_hip.x, right_hip.y],
                    [right_knee.x, right_knee.y],
                    [right_ankle.x, right_ankle.y]
                )
            
            if len(angles) == 0:
                return None
            
            # For squats, average both legs if both visible (symmetric movement)
            if len(angles) == 2:
                return (angles["left"] + angles["right"]) / 2
            # Otherwise return whichever is visible
            return list(angles.values())[0]
            
        except Exception as e:
            return None
    
    def check_form(self, landmarks):
        """
        Check squat form and provide feedback
        Returns: (feedback, back_angle, form_quality)
        """
        try:
            # Get landmarks for form checking
            left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            left_hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP.value]
            left_knee = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
            left_ankle = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
            
            right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            right_hip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
            right_knee = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
            right_ankle = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
            
            issues = []
            
            # Calculate back angle (torso lean) - shoulder to hip relative to vertical
            # Average both sides
            left_back = self.calculate_angle(
                [left_shoulder.x, left_shoulder.y],
                [left_hip.x, left_hip.y],
                [left_hip.x, left_hip.y + 0.5]  # Vertical reference
            )
            right_back = self.calculate_angle(
                [right_shoulder.x, right_shoulder.y],
                [right_hip.x, right_hip.y],
                [right_hip.x, right_hip.y + 0.5]
            )
            self.back_angle = (left_back + right_back) / 2
            
            # Check if back is too far forward (excessive lean)
            if self.back_angle > 45:
                issues.append("KEEP CHEST UP")
            elif self.back_angle > 35:
                issues.append("SLIGHT CHEST LEAN")
            
            # Check knees going past toes (lateral view check)
            # Compare knee x position with ankle x position
            left_knee_over = left_knee.x > left_ankle.x + 0.05
            right_knee_over = right_knee.x > right_ankle.x + 0.05
            
            if left_knee_over and right_knee_over:
                issues.append("KNEES OVER TOES - SIT BACK")
                self.knee_over_toe = True
            else:
                self.knee_over_toe = False
            
            # Check depth - using current knee angle from state
            knee_angle = self.get_signature_angle(landmarks)
            if knee_angle and self.state == "down":
                if knee_angle > 110:
                    issues.append("GO DEEPER")
            
            # Check if knees are caving in (valgus)
            # Compare knee width to hip width
            hip_width = abs(left_hip.x - right_hip.x)
            knee_width = abs(left_knee.x - right_knee.x)
            
            if knee_width < hip_width * 0.8:
                issues.append("PUSH KNEES OUT")
            
            # Determine form quality
            if issues:
                self.feedback = " | ".join(issues)
                if len(issues) >= 2:
                    self.form_quality = "POOR"
                else:
                    self.form_quality = "FAIR"
            else:
                self.feedback = "PERFECT FORM"
                self.form_quality = "GOOD"
            
            return self.feedback, self.back_angle, self.form_quality
            
        except Exception as e:
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
        
        # Enhance result with squat-specific data
        base_result["form_feedback"] = feedback
        base_result["back_angle"] = round(back_angle, 1) if back_angle else 0
        base_result["form_quality"] = form_quality
        base_result["knee_over_toe"] = self.knee_over_toe
        
        return base_result
    
    def count_rep(self, landmarks):
        """
        Legacy method for backward compatibility
        Returns: (rep_count, stage, knee_angle, feedback, back_angle, form_quality)
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
