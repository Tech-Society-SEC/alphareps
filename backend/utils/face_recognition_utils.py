import face_recognition
import cv2
import numpy as np
import os
import pickle
from typing import List, Optional, Dict, Tuple
import json
from datetime import datetime

class FaceRecognitionManager:
    def __init__(self, encodings_file: str = "data/face_encodings.pkl"):
        self.encodings_file = encodings_file
        self.known_face_encodings = []
        self.known_face_names = []
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(encodings_file), exist_ok=True)
        
        # Load existing encodings
        self.load_encodings()
    
    def load_encodings(self) -> bool:
        """Load face encodings from file"""
        try:
            if os.path.exists(self.encodings_file):
                with open(self.encodings_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data.get('encodings', [])
                    self.known_face_names = data.get('names', [])
                print(f"✅ Loaded {len(self.known_face_names)} face encodings")
                return True
            else:
                print("📁 No existing face encodings found")
                return False
        except Exception as e:
            print(f"❌ Error loading face encodings: {e}")
            return False
    
    def save_encodings(self) -> bool:
        """Save face encodings to file"""
        try:
            data = {
                'encodings': self.known_face_encodings,
                'names': self.known_face_names,
                'created_at': datetime.now().isoformat()
            }
            
            with open(self.encodings_file, 'wb') as f:
                pickle.dump(data, f)
            
            print(f"💾 Saved {len(self.known_face_names)} face encodings")
            return True
            
        except Exception as e:
            print(f"❌ Error saving face encodings: {e}")
            return False
    
    def register_face(self, image: np.ndarray, name: str) -> bool:
        """Register a new face with a name"""
        try:
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image
            
            # Find face locations
            face_locations = face_recognition.face_locations(rgb_image)
            
            if not face_locations:
                print("❌ No face detected in the image")
                return False
            
            if len(face_locations) > 1:
                print("⚠️ Multiple faces detected, using the first one")
            
            # Get face encoding
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            
            if not face_encodings:
                print("❌ Could not encode the face")
                return False
            
            # Check if this person is already registered
            if name in self.known_face_names:
                # Update existing encoding
                index = self.known_face_names.index(name)
                self.known_face_encodings[index] = face_encodings[0]
                print(f"🔄 Updated face encoding for {name}")
            else:
                # Add new encoding
                self.known_face_encodings.append(face_encodings[0])
                self.known_face_names.append(name)
                print(f"➕ Added new face encoding for {name}")
            
            # Save encodings
            self.save_encodings()
            return True
            
        except Exception as e:
            print(f"❌ Error registering face: {e}")
            return False
    
    def recognize_face(self, image: np.ndarray, tolerance: float = 0.6) -> Optional[str]:
        """Recognize a face in the image"""
        try:
            if not self.known_face_encodings:
                print("❌ No registered faces found")
                return None
            
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image
            
            # Find face locations and encodings
            face_locations = face_recognition.face_locations(rgb_image)
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            
            if not face_encodings:
                print("❌ No face detected in the image")
                return None
            
            # Compare with known faces
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(
                    self.known_face_encodings, face_encoding, tolerance=tolerance
                )
                
                # Calculate face distances
                face_distances = face_recognition.face_distance(
                    self.known_face_encodings, face_encoding
                )
                
                # Find the best match
                if True in matches:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = 1 - face_distances[best_match_index]
                        print(f"✅ Recognized {name} with confidence {confidence:.2f}")
                        return name
            
            print("❌ Face not recognized")
            return None
            
        except Exception as e:
            print(f"❌ Error recognizing face: {e}")
            return None
    
    def recognize_faces_in_frame(self, frame: np.ndarray, tolerance: float = 0.6) -> List[Dict[str, any]]:
        """Recognize all faces in a frame and return their information"""
        try:
            if not self.known_face_encodings:
                return []
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize frame for faster processing
            small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.25, fy=0.25)
            
            # Find face locations and encodings
            face_locations = face_recognition.face_locations(small_frame)
            face_encodings = face_recognition.face_encodings(small_frame, face_locations)
            
            recognized_faces = []
            
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Scale back up face locations
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                # Compare with known faces
                matches = face_recognition.compare_faces(
                    self.known_face_encodings, face_encoding, tolerance=tolerance
                )
                face_distances = face_recognition.face_distance(
                    self.known_face_encodings, face_encoding
                )
                
                name = "Unknown"
                confidence = 0.0
                
                if True in matches:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = 1 - face_distances[best_match_index]
                
                recognized_faces.append({
                    'name': name,
                    'confidence': confidence,
                    'location': {
                        'top': top,
                        'right': right,
                        'bottom': bottom,
                        'left': left
                    }
                })
            
            return recognized_faces
            
        except Exception as e:
            print(f"❌ Error recognizing faces in frame: {e}")
            return []
    
    def delete_face(self, name: str) -> bool:
        """Delete a registered face"""
        try:
            if name in self.known_face_names:
                index = self.known_face_names.index(name)
                del self.known_face_encodings[index]
                del self.known_face_names[index]
                
                self.save_encodings()
                print(f"🗑️ Deleted face encoding for {name}")
                return True
            else:
                print(f"❌ Face {name} not found")
                return False
                
        except Exception as e:
            print(f"❌ Error deleting face: {e}")
            return False
    
    def get_registered_faces(self) -> List[str]:
        """Get list of all registered face names"""
        return self.known_face_names.copy()
    
    def get_face_count(self) -> int:
        """Get number of registered faces"""
        return len(self.known_face_names)
    
    def draw_face_boxes(self, frame: np.ndarray, recognized_faces: List[Dict[str, any]]) -> np.ndarray:
        """Draw bounding boxes and names on detected faces"""
        try:
            for face_info in recognized_faces:
                location = face_info['location']
                name = face_info['name']
                confidence = face_info['confidence']
                
                # Draw rectangle around face
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, 
                            (location['left'], location['top']), 
                            (location['right'], location['bottom']), 
                            color, 2)
                
                # Draw label
                label = f"{name} ({confidence:.2f})" if name != "Unknown" else name
                cv2.rectangle(frame, 
                            (location['left'], location['bottom'] - 35), 
                            (location['right'], location['bottom']), 
                            color, cv2.FILLED)
                
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, label, 
                          (location['left'] + 6, location['bottom'] - 6), 
                          font, 0.6, (255, 255, 255), 1)
            
            return frame
            
        except Exception as e:
            print(f"❌ Error drawing face boxes: {e}")
            return frame
    
    def create_demo_faces(self):
        """Create some demo face encodings for testing"""
        try:
            # Create dummy face encodings for demo purposes
            demo_names = ["John Doe", "Jane Smith", "Mike Johnson"]
            
            for name in demo_names:
                # Create a random face encoding (128 dimensions)
                dummy_encoding = np.random.rand(128)
                
                self.known_face_encodings.append(dummy_encoding)
                self.known_face_names.append(name)
            
            self.save_encodings()
            print(f"✅ Created {len(demo_names)} demo face encodings")
            
        except Exception as e:
            print(f"❌ Error creating demo faces: {e}")

# Example usage and testing
if __name__ == "__main__":
    # Initialize face recognition manager
    face_manager = FaceRecognitionManager()
    
    # Create demo faces if no faces are registered
    if face_manager.get_face_count() == 0:
        face_manager.create_demo_faces()
    
    print(f"Registered faces: {face_manager.get_registered_faces()}")
    
    # Test with webcam (uncomment to test)
    """
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Recognize faces
        recognized_faces = face_manager.recognize_faces_in_frame(frame)
        
        # Draw face boxes
        frame_with_boxes = face_manager.draw_face_boxes(frame, recognized_faces)
        
        # Display
        cv2.imshow('Face Recognition', frame_with_boxes)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    """
