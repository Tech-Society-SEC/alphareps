"""
Improved shoulder press counter with FSM, angle-based logic, and form feedback
"""
from .base_counter import BaseRepCounter
import numpy as np
import mediapipe as mp

class ShoulderPressCounter(BaseRepCounter):
    """
    Shoulder press counter using:
    - Elbow angle as signature angle
    - FSM for stable counting
    - Better arm selection logic
    - Form feedback for proper overhead press technique
    """
    
    def __init__(self):
        super().__init__(
            history_size=3,
            min_transition_frames=1,
            rep_debounce_seconds=0.05  # Very fast response
        )
        self.confidence_threshold = 0.2  # More permissive visibility
        self.mp_pose = mp.solutions.pose
        self.feedback = ""
        self.form_quality = "GOOD"
        self.wrist_alignment = True
        self.core_engaged = True
    
    def get_angle_thresholds(self):
        """
        Shoulder press thresholds (LIBERAL for responsiveness):
        - Down (start): < 120° (arms bent - more forgiving)
        - Up (top): > 145° (arms extended - easier to trigger)
        """
        return 120, 145  # Liberal thresholds
    
    def get_signature_angle(self, landmarks):
        """
        Get elbow angle for shoulder press
        Uses better arm selection instead of averaging
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
            
            # For shoulder press, average both arms (bilateral movement)
            if len(angles) == 2:
                return (angles["left"] + angles["right"]) / 2
            
            # Choose better arm (higher visibility)
            if left_vis >= right_vis:
                return angles.get("left")
            return angles.get("right")
            
        except Exception as e:
            return None
    
    def check_form(self, landmarks):
        """
        Check shoulder press form and provide feedback
        Returns: (feedback, wrist_over_elbow, form_quality)
        """
        try:
            issues = []
            
            # Get landmarks
            left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            left_elbow = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
            left_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
            
            right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            right_elbow = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            right_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
            
            left_hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
            
            nose = landmarks.landmark[self.mp_pose.PoseLandmark.NOSE.value]
            
            # Check 1: Wrists should be directly over elbows (avoid flared wrists)
            left_wrist_alignment = abs(left_wrist.x - left_elbow.x)
            right_wrist_alignment = abs(right_wrist.x - right_elbow.x)
            
            if left_wrist_alignment > 0.08 or right_wrist_alignment > 0.08:
                issues.append("STACK WRISTS OVER ELBOWS")
                self.wrist_alignment = False
            else:
                self.wrist_alignment = True
            
            # Check 2: Elbows should be at ~90° at bottom position (not too low)
            if self.state == "down":
                left_elbow_angle = self.calculate_angle(
                    [left_shoulder.x, left_shoulder.y],
                    [left_elbow.x, left_elbow.y],
                    [left_wrist.x, left_wrist.y]
                )
                right_elbow_angle = self.calculate_angle(
                    [right_shoulder.x, right_shoulder.y],
                    [right_elbow.x, right_elbow.y],
                    [right_wrist.x, right_wrist.y]
                )
                
                avg_angle = (left_elbow_angle + right_elbow_angle) / 2
                if avg_angle < 70:
                    issues.append("DON'T GO TOO LOW")
            
            # Check 3: Elbows should not flare too wide
            shoulder_width = abs(left_shoulder.x - right_shoulder.x)
            elbow_width = abs(left_elbow.x - right_elbow.x)
            
            if elbow_width > shoulder_width * 1.5:
                issues.append("TUCK ELBOWS SLIGHTLY")
            
            # Check 4: Core engagement - back should not arch excessively
            # Check if shoulders are behind hips (arching back)
            avg_shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
            avg_hip_x = (left_hip.x + right_hip.x) / 2
            
            back_arch = avg_hip_x - avg_shoulder_x
            if back_arch > 0.05:
                issues.append("ENGAGE CORE - DON'T ARCH BACK")
                self.core_engaged = False
            else:
                self.core_engaged = True
            
            # Check 5: At top, arms should be fully extended
            if self.state == "up":
                # Check if wrists are above head (proper lockout)
                avg_wrist_y = (left_wrist.y + right_wrist.y) / 2
                if avg_wrist_y > nose.y:
                    issues.append("FULL EXTENSION OVERHEAD")
            
            # Check 6: Head position - avoid pushing head forward
            avg_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
            if nose.y > avg_shoulder_y + 0.1:
                issues.append("KEEP HEAD NEUTRAL")
            
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
            
            wrist_over_elbow = 1 - max(left_wrist_alignment, right_wrist_alignment)
            return self.feedback, wrist_over_elbow, self.form_quality
            
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
        feedback, wrist_over_elbow, form_quality = self.check_form(landmarks)
        
        # Enhance result with shoulder press-specific data
        base_result["form_feedback"] = feedback
        base_result["wrist_alignment"] = round(wrist_over_elbow * 100, 1) if wrist_over_elbow else 0
        base_result["form_quality"] = form_quality
        base_result["core_engaged"] = self.core_engaged
        
        return base_result
    
    def count_rep(self, landmarks):
        """
        Legacy method for backward compatibility
        Returns: (rep_count, stage, elbow_angle, feedback, wrist_alignment, form_quality)
        """
        result = self.process_frame(landmarks)
        return (
            result["reps"],
            result["state"],
            result["angle"],
            result.get("form_feedback", ""),
            result.get("wrist_alignment", 0),
            result.get("form_quality", "GOOD")
        )
