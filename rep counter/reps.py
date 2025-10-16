import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

CURL_ANGLE_UP = 35
CURL_ANGLE_DOWN = 160
VISIBILITY_THRESHOLD = 0.7

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

left_counter = 0
right_counter = 0
left_stage = 'down'
right_stage = 'down'

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Blank frame grabbed.")
            break

        # Flip the camera feed horizontally (mirror view)
        frame = cv2.flip(frame, 1)

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        results = pose.process(image)
    
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        try:
            landmarks = results.pose_landmarks.landmark
            
            l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            l_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
            l_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
            
            if (l_shoulder.visibility > VISIBILITY_THRESHOLD and
                l_elbow.visibility > VISIBILITY_THRESHOLD and
                l_wrist.visibility > VISIBILITY_THRESHOLD):
                
                l_shoulder_coords = [l_shoulder.x, l_shoulder.y]
                l_elbow_coords = [l_elbow.x, l_elbow.y]
                l_wrist_coords = [l_wrist.x, l_wrist.y]
                
                left_angle = calculate_angle(l_shoulder_coords, l_elbow_coords, l_wrist_coords)
                
                cv2.putText(image, f"{int(left_angle)}", 
                               tuple(np.multiply(l_elbow_coords, [640, 480]).astype(int)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                
                if left_angle > CURL_ANGLE_DOWN:
                    left_stage = "down"
                if left_angle < CURL_ANGLE_UP and left_stage == 'down':
                    left_stage = "up"
                    left_counter += 1
                    print(f"Left Reps: {left_counter}")

            r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            r_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            r_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

            if (r_shoulder.visibility > VISIBILITY_THRESHOLD and
                r_elbow.visibility > VISIBILITY_THRESHOLD and
                r_wrist.visibility > VISIBILITY_THRESHOLD):

                r_shoulder_coords = [r_shoulder.x, r_shoulder.y]
                r_elbow_coords = [r_elbow.x, r_elbow.y]
                r_wrist_coords = [r_wrist.x, r_wrist.y]

                right_angle = calculate_angle(r_shoulder_coords, r_elbow_coords, r_wrist_coords)
                
                cv2.putText(image, f"{int(right_angle)}", 
                               tuple(np.multiply(r_elbow_coords, [640, 480]).astype(int)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                
                if right_angle > CURL_ANGLE_DOWN:
                    right_stage = "down"
                if right_angle < CURL_ANGLE_UP and right_stage == 'down':
                    right_stage = "up"
                    right_counter += 1
                    print(f"Right Reps: {right_counter}")

        except Exception as e:
            pass
        
        height, width, _ = image.shape
        
        cv2.rectangle(image, (0, 0), (250, 75), (245, 117, 16), -1)
        cv2.putText(image, 'LEFT REPS', (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(left_counter), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, 'STAGE', (130, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, left_stage, (130, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.rectangle(image, (width - 250, 0), (width, 75), (245, 117, 16), -1)
        cv2.putText(image, 'RIGHT REPS', (width - 235, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(right_counter), (width - 230, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, 'STAGE', (width - 120, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, right_stage, (width - 120, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
        
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               
        
        cv2.imshow('AI Curl Counter', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
