"""
Improved curl counter with FSM, angle-based logic, and form feedback
"""
from .base_counter import BaseRepCounter
import numpy as np

class CurlCounter(BaseRepCounter):
    """
    Curl counter (bicep curl, hammer curl) using:
    - Elbow angle as signature angle
    - FSM for stable counting
    - Dual-arm detection with automatic arm selection
    - Form feedback for proper curl technique
    """
    
    def __init__(self):
        super().__init__(
            history_size=3,
            min_transition_frames=1,
            rep_debounce_seconds=0.05  # Very fast response
        )
        self.confidence_threshold = 0.2  # More permissive visibility
        self.mp_pose = __import__('mediapipe').solutions.pose
        self.primary_arm = 'right'
        self.feedback = ""
        self.form_quality = "GOOD"
        self.elbow_movement = 0
        self.shoulder_stability = True
        
    def get_angle_thresholds(self):
        """
        Curl thresholds (LIBERAL for responsiveness):
        - Down (curled): < 100° (bicep contracted - more forgiving)
        - Up (extended): > 140° (arm extended - easier to trigger)
        """
        return 100, 140  # Liberal thresholds
    
    def get_signature_angle(self, landmarks):
        """
        Get elbow angle for curls
        Uses the arm with more movement (more variation)
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
            
            # For curls, choose the arm with smaller angle (more curled = actively exercising)
            if len(angles) == 2:
                if angles["left"] < angles["right"]:
                    self.primary_arm = 'left'
                    return angles["left"]
                else:
                    self.primary_arm = 'right'
                    return angles["right"]
            elif "right" in angles:
                self.primary_arm = 'right'
                return angles["right"]
            else:
                self.primary_arm = 'left'
                return angles["left"]
            
        except Exception as e:
            return None
    
    def check_form(self, landmarks):
        """
        Check curl form and provide feedback
        Returns: (feedback, elbow_movement, form_quality)
        """
        try:
            issues = []
            
            # Get the primary arm landmarks based on which arm is curling
            if self.primary_arm == 'left':
                shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                elbow = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
                wrist = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
                hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP.value]
            else:
                shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                elbow = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                wrist = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
                hip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
            
            # Check 1: Elbow should stay close to body (compare elbow x to hip x)
            elbow_drift = abs(elbow.x - hip.x)
            if elbow_drift > 0.15:
                issues.append("TUCK ELBOWS IN")
                self.shoulder_stability = False
            else:
                self.shoulder_stability = True
            
            # Check 2: Elbow should not move forward during curl
            # Compare elbow position to shoulder position
            elbow_forward = shoulder.x - elbow.x
            self.elbow_movement = elbow_forward
            
            if abs(elbow_forward) > 0.08:
                issues.append("KEEP ELBOW STATIONARY")
            
            # Check 3: Upper arm should stay vertical (shoulder to elbow angle)
            upper_arm_angle = self.calculate_angle(
                [shoulder.x, shoulder.y],
                [elbow.x, elbow.y],
                [elbow.x, elbow.y + 0.5]  # Vertical reference
            )
            
            if upper_arm_angle > 25:
                issues.append("KEEP UPPER ARM STILL")
            
            # Check 4: Wrist should not curl (check wrist alignment)
            # Wrist should be in line with forearm
            forearm_angle = self.calculate_angle(
                [elbow.x, elbow.y],
                [wrist.x, wrist.y],
                [wrist.x, wrist.y - 0.2]  # Straight extension
            )
            
            if forearm_angle > 30 and forearm_angle < 150:
                issues.append("KEEP WRIST STRAIGHT")
            
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
            
            return self.feedback, self.elbow_movement, self.form_quality
            
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
        feedback, elbow_movement, form_quality = self.check_form(landmarks)
        
        # Enhance result with curl-specific data
        base_result["form_feedback"] = feedback
        base_result["elbow_movement"] = round(elbow_movement * 100, 1) if elbow_movement else 0
        base_result["form_quality"] = form_quality
        base_result["primary_arm"] = self.primary_arm
        base_result["shoulder_stability"] = self.shoulder_stability
        
        return base_result
    
    def count_rep(self, landmarks):
        """
        Legacy method for backward compatibility
        Returns: (rep_count, stage, elbow_angle, feedback, elbow_movement, form_quality)
        """
        result = self.process_frame(landmarks)
        return (
            result["reps"],
            result["state"],
            result["angle"],
            result.get("form_feedback", ""),
            result.get("elbow_movement", 0),
            result.get("form_quality", "GOOD")
        )
