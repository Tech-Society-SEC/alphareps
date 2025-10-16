import cv2
import mediapipe as mp
import numpy as np
import time
import math
import winsound  # For beep sound (Windows only)

def find_angle(p1, p2, ref_pt):
    p1_ref = np.array(p1) - np.array(ref_pt)
    p2_ref = np.array(p2) - np.array(ref_pt)
    cos_theta = np.dot(p1_ref, p2_ref) / (np.linalg.norm(p1_ref) * np.linalg.norm(p2_ref) + 1e-6)
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return int(np.degrees(theta))

def get_mediapipe_pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5):
    return mp.solutions.pose.Pose(
        static_image_mode=static_image_mode,
        model_complexity=model_complexity,
        smooth_landmarks=smooth_landmarks,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence
    )

def draw_text(img, msg, pos, font_scale=0.8, text_color=(255, 255, 255),
              text_color_bg=(0, 0, 0), box_offset=(20, 10)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_thickness = 2
    x, y = pos
    text_size, _ = cv2.getTextSize(msg, font, font_scale, font_thickness)
    text_w, text_h = text_size
    rec_start = tuple(p - o for p, o in zip(pos, box_offset))
    rec_end = tuple(m + n - o for m, n, o in zip((x + text_w, y + text_h),
                                                  box_offset, (25, 0)))
    cv2.rectangle(img, rec_start, rec_end, text_color_bg, -1)
    cv2.putText(img, msg, (int(rec_start[0] + 6), int(y + text_h + font_scale - 1)),
                font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    return text_size

def draw_dotted_line(frame, pt1, pt2, color, thickness=2, gap=10):
    dist = int(math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1]))
    for i in range(0, dist, gap * 2):
        start = (int(pt1[0] + (pt2[0] - pt1[0]) * i / dist),
                 int(pt1[1] + (pt2[1] - pt1[1]) * i / dist))
        end = (int(pt1[0] + (pt2[0] - pt1[0]) * (i + gap) / dist),
               int(pt1[1] + (pt2[1] - pt1[1]) * (i + gap) / dist))
        if end[0] > frame.shape[1] or end[1] > frame.shape[0]:
            break
        cv2.line(frame, start, end, color, thickness)

def get_thresholds_beginner():
    return {
        'HIP_KNEE_VERT': {'NORMAL': (0, 32), 'TRANS': (35, 65), 'PASS': (70, 95)},
        'KNEE_THRESH': [50, 70, 95],
    }

def get_thresholds_pro():
    return {
        'HIP_KNEE_VERT': {'NORMAL': (0, 32), 'TRANS': (35, 65), 'PASS': (80, 95)},
        'KNEE_THRESH': [50, 80, 95],
    }

class ProcessFrame:
    def __init__(self, thresholds, flip_frame=False):
        self.flip_frame = flip_frame
        self.thresholds = thresholds
        self.COLORS = {
            'blue': (0, 127, 255), 'red': (255, 50, 50), 'green': (0, 255, 127),
            'yellow': (255, 255, 0), 'white': (255, 255, 255)
        }
        self.dict_features = {
            'left': {'hip': 23, 'knee': 25, 'ankle': 27},
            'right': {'hip': 24, 'knee': 26, 'ankle': 28}
        }
        self.state_tracker = {
            'prev_state': None, 'curr_state': None,
            'SQUAT_COUNT': 0, 'IMPROPER_SQUAT': 0,
            'INCORRECT_POSTURE': False
        }

    def _get_landmark_features(self, landmarks, side, frame_width, frame_height):
        features = self.dict_features[side]
        coords = {}
        for key, lm_id in features.items():
            lm = landmarks[lm_id]
            coords[key] = (int(lm.x * frame_width), int(lm.y * frame_height))
        return coords

    def _determine_side(self, landmarks, frame_width):
        left_hip = landmarks[self.dict_features['left']['hip']]
        right_hip = landmarks[self.dict_features['right']['hip']]
        return 'left' if left_hip.x < right_hip.x else 'right'

    def process(self, frame: np.ndarray, pose):
        play_sound = None
        frame_height, frame_width, _ = frame.shape
        results = pose.process(frame)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            side = self._determine_side(landmarks, frame_width)
            coords = self._get_landmark_features(landmarks, side, frame_width, frame_height)
            hip_knee_vert_angle = find_angle(coords['hip'], coords['knee'], (coords['knee'][0], 0))
            knee_angle = find_angle(coords['hip'], coords['knee'], coords['ankle'])

            if self.thresholds['HIP_KNEE_VERT']['NORMAL'][0] <= hip_knee_vert_angle <= self.thresholds['HIP_KNEE_VERT']['NORMAL'][1]:
                self.state_tracker['curr_state'] = 's1'
            elif self.thresholds['HIP_KNEE_VERT']['TRANS'][0] <= hip_knee_vert_angle <= self.thresholds['HIP_KNEE_VERT']['TRANS'][1]:
                self.state_tracker['curr_state'] = 's2'
            elif self.thresholds['HIP_KNEE_VERT']['PASS'][0] <= hip_knee_vert_angle <= self.thresholds['HIP_KNEE_VERT']['PASS'][1]:
                self.state_tracker['curr_state'] = 's3'

            if self.state_tracker['curr_state'] != self.state_tracker['prev_state']:
                if self.state_tracker['curr_state'] == 's3':
                    if not (self.thresholds['KNEE_THRESH'][1] < knee_angle < self.thresholds['KNEE_THRESH'][2]):
                        self.state_tracker['IMPROPER_SQUAT'] += 1
                        self.state_tracker['INCORRECT_POSTURE'] = True
                    else:
                        self.state_tracker['INCORRECT_POSTURE'] = False

                if self.state_tracker['prev_state'] == 's3' and self.state_tracker['curr_state'] == 's2':
                    if not self.state_tracker['INCORRECT_POSTURE']:
                        self.state_tracker['SQUAT_COUNT'] += 1
                        play_sound = True
                self.state_tracker['prev_state'] = self.state_tracker['curr_state']

            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(255,255,0), thickness=2, circle_radius=2),
                connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0,255,0), thickness=2)
            )

            draw_dotted_line(frame, coords['knee'], (coords['knee'][0], 0), self.COLORS['blue'])
            cv2.putText(frame, f"Knee Angle: {knee_angle}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLORS['white'], 2)
            cv2.putText(frame, f"Hip-Knee-Vert Angle: {hip_knee_vert_angle}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLORS['white'], 2)
            draw_text(frame, f"CORRECT SQUATS: {self.state_tracker['SQUAT_COUNT']}", pos=(frame_width - 300, 30), text_color_bg=self.COLORS['green'])
            draw_text(frame, f"IMPROPER SQUATS: {self.state_tracker['IMPROPER_SQUAT']}", pos=(frame_width - 300, 80), text_color_bg=self.COLORS['red'])
            draw_text(frame, f"STATE: {self.state_tracker['curr_state']}", pos=(10, frame_height - 50), text_color_bg=self.COLORS['blue'])

            if self.state_tracker['INCORRECT_POSTURE']:
                draw_text(frame, "IMPROPER FORM", pos=(frame_width // 2 - 100, frame_height // 2),
                          text_color_bg=self.COLORS['red'], font_scale=1.5)
        else:
            cv2.putText(frame, "No person detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, self.COLORS['red'], 2)
        return frame, play_sound

MODE = 'Beginner'
print(f"Selected Mode: {MODE}")
thresholds = get_thresholds_beginner() if MODE == 'Beginner' else get_thresholds_pro()
pose = get_mediapipe_pose()
process_frame = ProcessFrame(thresholds=thresholds)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
else:
    print("Press 'q' to quit the live demo.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frame, beep = process_frame.process(rgb_frame, pose)
        bgr_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Squat Analysis - Live", bgr_frame)

        if beep:
            winsound.Beep(800, 150)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nLive session ended.")
    print(f"Correct Squats: {process_frame.state_tracker['SQUAT_COUNT']}")
    print(f"Improper Squats: {process_frame.state_tracker['IMPROPER_SQUAT']}")
