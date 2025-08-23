import numpy as np
import math
from typing import Dict, List, Tuple

class AdvancedMetricsCalculator:
   """Advanced medical metrics calculation"""
   
   @staticmethod
   def calculate_facial_symmetry_score(landmarks: np.ndarray, facial_midline_x: float) -> float:
       """Calculate overall facial symmetry score (0-1, where 1 is perfectly symmetric)"""
       # Define paired landmark indices for symmetry analysis
       symmetry_pairs = [
           (33, 133),    # Eye corners
           (61, 291),    # Mouth corners
           (70, 300),    # Eyebrow inner
           (103, 332),   # Eyebrow outer
           (234, 454),   # Cheek points
       ]
       
       asymmetry_scores = []
       
       for left_idx, right_idx in symmetry_pairs:
           left_point = landmarks[left_idx]
           right_point = landmarks[right_idx]
           
           # Calculate expected symmetric positions
           left_expected_x = facial_midline_x - abs(left_point[0] - facial_midline_x)
           right_expected_x = facial_midline_x + abs(right_point[0] - facial_midline_x)
           
           # Calculate asymmetry
           left_deviation = abs(left_point[0] - left_expected_x)
           right_deviation = abs(right_point[0] - right_expected_x)
           
           # Normalize by facial width
           face_width = max(landmarks[:, 0]) - min(landmarks[:, 0])
           normalized_asymmetry = (left_deviation + right_deviation) / face_width
           
           asymmetry_scores.append(normalized_asymmetry)
       
       # Overall symmetry score (1 - average asymmetry)
       avg_asymmetry = np.mean(asymmetry_scores)
       symmetry_score = max(0, 1 - avg_asymmetry * 10)  # Scale factor for visibility
       
       return symmetry_score
   
   @staticmethod
   def calculate_eye_aspect_ratios(landmarks: np.ndarray) -> Tuple[float, float]:
       """Calculate Eye Aspect Ratio (EAR) for both eyes"""
       # Left eye landmarks
       left_eye_points = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
       # Right eye landmarks  
       right_eye_points = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
       
       def eye_aspect_ratio(eye_points):
           # Get eye landmarks
           eye_landmarks = landmarks[eye_points]
           
           # Calculate vertical distances
           A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
           B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
           
           # Calculate horizontal distance
           C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
           
           # Calculate EAR
           ear = (A + B) / (2.0 * C)
           return ear
       
       left_ear = eye_aspect_ratio(left_eye_points[:6])  # Use first 6 points for simplicity
       right_ear = eye_aspect_ratio(right_eye_points[:6])
       
       return left_ear, right_ear
   
   @staticmethod
   def calculate_mouth_aspect_ratio(landmarks: np.ndarray) -> float:
       """Calculate Mouth Aspect Ratio (MAR)"""
       # Mouth landmarks
       mouth_points = [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318]
       
       mouth_landmarks = landmarks[mouth_points]
       
       # Calculate vertical distance (mouth opening)
       vertical_dist = np.linalg.norm(mouth_landmarks[3] - mouth_landmarks[9])
       
       # Calculate horizontal distance (mouth width)
       horizontal_dist = np.linalg.norm(mouth_landmarks[0] - mouth_landmarks[6])
       
       # Calculate MAR
       mar = vertical_dist / horizontal_dist
       return mar
   
   @staticmethod
   def detect_head_pose(landmarks: np.ndarray, image_shape: Tuple[int, int]) -> Dict[str, float]:
       """Estimate head pose angles using facial landmarks"""
       h, w = image_shape[:2]
       
       # Define 3D model points (rough approximation)
       model_points = np.array([
           (0.0, 0.0, 0.0),             # Nose tip
           (0.0, -330.0, -65.0),        # Chin
           (-225.0, 170.0, -135.0),     # Left eye left corner
           (225.0, 170.0, -135.0),      # Right eye right corner
           (-150.0, -150.0, -125.0),    # Left mouth corner
           (150.0, -150.0, -125.0)      # Right mouth corner
       ])
       
       # Corresponding 2D image points
       image_points = np.array([
           landmarks[1][:2],    # Nose tip
           landmarks[152][:2],  # Chin
           landmarks[33][:2],   # Left eye left corner
           landmarks[133][:2],  # Right eye right corner
           landmarks[61][:2],   # Left mouth corner
           landmarks[291][:2]   # Right mouth corner
       ], dtype="double")
       
       # Camera matrix (approximation)
       focal_length = w
       center = (w/2, h/2)
       camera_matrix = np.array([
           [focal_length, 0, center[0]],
           [0, focal_length, center[1]],
           [0, 0, 1]
       ], dtype="double")
       
       # Distortion coefficients (assuming no lens distortion)
       dist_coeffs = np.zeros((4, 1))
       
       # Solve PnP
       success, rotation_vector, translation_vector = cv2.solvePnP(
           model_points, image_points, camera_matrix, dist_coeffs
       )
       
       if success:
           # Convert rotation vector to rotation matrix
           rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
           
           # Extract Euler angles
           sy = math.sqrt(rotation_matrix[0,0] * rotation_matrix[0,0] +  rotation_matrix[1,0] * rotation_matrix[1,0])
           
           singular = sy < 1e-6
           
           if not singular:
               x = math.atan2(rotation_matrix[2,1], rotation_matrix[2,2])
               y = math.atan2(-rotation_matrix[2,0], sy)
               z = math.atan2(rotation_matrix[1,0], rotation_matrix[0,0])
           else:
               x = math.atan2(-rotation_matrix[1,2], rotation_matrix[1,1])
               y = math.atan2(-rotation_matrix[2,0], sy)
               z = 0
           
           # Convert to degrees
           pitch = math.degrees(x)
           yaw = math.degrees(y)
           roll = math.degrees(z)
           
           return {
               "pitch": pitch,
               "yaw": yaw,
               "roll": roll
           }
       
       return {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
