import cv2
import numpy as np
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math

@dataclass
class LandmarkMetrics:
    """Structured container for medical landmark metrics"""
    mouth_corner_deviation_mm: float
    eyebrow_height_difference_mm: float
    eye_closure_left_percent: float
    eye_closure_right_percent: float
    facial_midline_deviation: float
    iris_left_center: Tuple[float, float]
    iris_right_center: Tuple[float, float]
    head_pose_angles: Tuple[float, float, float]  # pitch, yaw, roll
    symmetry_score: float
    ear_left: float
    ear_right: float

class MedicalLandmarkAnalyzer:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,  # Better for video streams
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # MediaPipe 478-point landmark indices for medical analysis
        self.FACIAL_LANDMARKS = {
            'left_eye_corner': 33,
            'right_eye_corner': 133,
            'left_mouth_corner': 61,
            'right_mouth_corner': 291,
            'nose_tip': 1,
            'left_eyebrow_inner': 70,
            'right_eyebrow_inner': 300,
            'left_iris_center': 468,
            'right_iris_center': 473,
            'left_eye_top': 159,
            'left_eye_bottom': 145,
            'right_eye_top': 386,
            'right_eye_bottom': 374,
            'chin': 152,
            'forehead_center': 9,
            'left_cheek': 234,
            'right_cheek': 454,
        }
        
        # Eye landmark groups for EAR calculation
        self.LEFT_EYE_POINTS = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE_POINTS = [33, 160, 158, 133, 153, 144]
    
    def extract_landmarks(self, image: np.ndarray) -> Optional[landmark_pb2.NormalizedLandmarkList]:
        """Extract landmarks using MediaPipe landmark_pb2"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        if results.multi_face_landmarks:
            return results.multi_face_landmarks[0]
        return None
    
    def get_landmark_coord(self, landmarks: landmark_pb2.NormalizedLandmarkList, 
                          idx: int, width: int, height: int) -> Tuple[float, float, float]:
        """Get single landmark coordinate"""
        if idx < len(landmarks.landmark):
            lm = landmarks.landmark[idx]
            return (lm.x * width, lm.y * height, lm.z * width)
        return (0.0, 0.0, 0.0)
    
    def calculate_ear(self, landmarks: landmark_pb2.NormalizedLandmarkList, 
                     eye_points: List[int], width: int, height: int) -> float:
        """Calculate Eye Aspect Ratio using landmark_pb2"""
        coords = []
        for idx in eye_points:
            x, y, z = self.get_landmark_coord(landmarks, idx, width, height)
            coords.append((x, y))
        
        if len(coords) < 6:
            return 0.0
        
        # Calculate vertical distances
        A = math.hypot(coords[1][0] - coords[5][0], coords[1][1] - coords[5][1])
        B = math.hypot(coords[2][0] - coords[4][0], coords[2][1] - coords[4][1])
        
        # Calculate horizontal distance
        C = math.hypot(coords[0][0] - coords[3][0], coords[0][1] - coords[3][1])
        
        if C == 0:
            return 0.0
        
        return (A + B) / (2.0 * C)
    
    def calculate_symmetry_score(self, landmarks: landmark_pb2.NormalizedLandmarkList,
                               width: int, height: int) -> float:
        """Calculate facial symmetry score (0-1, where 1 = perfectly symmetric)"""
        # Get nose tip for midline
        nose_x, _, _ = self.get_landmark_coord(landmarks, self.FACIAL_LANDMARKS['nose_tip'], width, height)
        
        # Paired landmarks for symmetry analysis
        pairs = [
            (self.FACIAL_LANDMARKS['left_eye_corner'], self.FACIAL_LANDMARKS['right_eye_corner']),
            (self.FACIAL_LANDMARKS['left_mouth_corner'], self.FACIAL_LANDMARKS['right_mouth_corner']),
            (self.FACIAL_LANDMARKS['left_eyebrow_inner'], self.FACIAL_LANDMARKS['right_eyebrow_inner']),
            (self.FACIAL_LANDMARKS['left_cheek'], self.FACIAL_LANDMARKS['right_cheek']),
        ]
        
        asymmetry_scores = []
        for left_idx, right_idx in pairs:
            left_x, left_y, _ = self.get_landmark_coord(landmarks, left_idx, width, height)
            right_x, right_y, _ = self.get_landmark_coord(landmarks, right_idx, width, height)
            
            # Calculate distance from midline
            left_dist = abs(left_x - nose_x)
            right_dist = abs(right_x - nose_x)
            
            # Asymmetry as relative difference
            if max(left_dist, right_dist) > 0:
                asymmetry = abs(left_dist - right_dist) / max(left_dist, right_dist)
                asymmetry_scores.append(asymmetry)
        
        if not asymmetry_scores:
            return 1.0
        
        avg_asymmetry = sum(asymmetry_scores) / len(asymmetry_scores)
        return max(0.0, 1.0 - avg_asymmetry)
    
    def calculate_medical_metrics(self, landmarks: landmark_pb2.NormalizedLandmarkList, 
                                image_shape: Tuple[int, int]) -> LandmarkMetrics:
        """Calculate comprehensive medical metrics from landmark_pb2 data"""
        height, width = image_shape[:2]
        
        # Get key landmark coordinates
        nose_x, nose_y, nose_z = self.get_landmark_coord(landmarks, self.FACIAL_LANDMARKS['nose_tip'], width, height)
        left_mouth_x, left_mouth_y, _ = self.get_landmark_coord(landmarks, self.FACIAL_LANDMARKS['left_mouth_corner'], width, height)
        right_mouth_x, right_mouth_y, _ = self.get_landmark_coord(landmarks, self.FACIAL_LANDMARKS['right_mouth_corner'], width, height)
        left_eyebrow_x, left_eyebrow_y, _ = self.get_landmark_coord(landmarks, self.FACIAL_LANDMARKS['left_eyebrow_inner'], width, height)
        right_eyebrow_x, right_eyebrow_y, _ = self.get_landmark_coord(landmarks, self.FACIAL_LANDMARKS['right_eyebrow_inner'], width, height)
        left_eye_top_x, left_eye_top_y, _ = self.get_landmark_coord(landmarks, self.FACIAL_LANDMARKS['left_eye_top'], width, height)
        left_eye_bot_x, left_eye_bot_y, _ = self.get_landmark_coord(landmarks, self.FACIAL_LANDMARKS['left_eye_bottom'], width, height)
        right_eye_top_x, right_eye_top_y, _ = self.get_landmark_coord(landmarks, self.FACIAL_LANDMARKS['right_eye_top'], width, height)
        right_eye_bot_x, right_eye_bot_y, _ = self.get_landmark_coord(landmarks, self.FACIAL_LANDMARKS['right_eye_bottom'], width, height)
        left_iris_x, left_iris_y, _ = self.get_landmark_coord(landmarks, self.FACIAL_LANDMARKS['left_iris_center'], width, height)
        right_iris_x, right_iris_y, _ = self.get_landmark_coord(landmarks, self.FACIAL_LANDMARKS['right_iris_center'], width, height)
        
        # Facial midline
        midline_x = nose_x
        
        # Mouth corner asymmetry
        left_mouth_to_midline = abs(left_mouth_x - midline_x)
        right_mouth_to_midline = abs(right_mouth_x - midline_x)
        mouth_asymmetry = abs(left_mouth_to_midline - right_mouth_to_midline)
        
        # Eyebrow height difference
        eyebrow_height_diff = abs(left_eyebrow_y - right_eyebrow_y)
        
        # Eye closure calculation
        left_eye_height = abs(left_eye_top_y - left_eye_bot_y)
        right_eye_height = abs(right_eye_top_y - right_eye_bot_y)
        max_eye_height = max(left_eye_height, right_eye_height, 1.0)
        
        left_eye_closure = max(0, min(100, (1 - left_eye_height / max_eye_height) * 100))
        right_eye_closure = max(0, min(100, (1 - right_eye_height / max_eye_height) * 100))
        
        # Eye Aspect Ratios
        ear_left = self.calculate_ear(landmarks, self.LEFT_EYE_POINTS, width, height)
        ear_right = self.calculate_ear(landmarks, self.RIGHT_EYE_POINTS, width, height)
        
        # Head pose (simplified)
        chin_x, chin_y, chin_z = self.get_landmark_coord(landmarks, self.FACIAL_LANDMARKS['chin'], width, height)
        forehead_x, forehead_y, forehead_z = self.get_landmark_coord(landmarks, self.FACIAL_LANDMARKS['forehead_center'], width, height)
        
        # Pitch calculation
        vertical_diff = forehead_y - chin_y
        depth_diff = forehead_z - chin_z if hasattr(landmarks.landmark[0], 'z') else 0
        pitch = math.degrees(math.atan2(depth_diff, abs(vertical_diff))) if vertical_diff != 0 else 0.0
        
        # Yaw calculation
        face_center_x = width / 2
        yaw = math.degrees(math.atan2(nose_x - face_center_x, width * 0.3))
        
        # Roll calculation
        left_eye_x, left_eye_y, _ = self.get_landmark_coord(landmarks, self.FACIAL_LANDMARKS['left_eye_corner'], width, height)
        right_eye_x, right_eye_y, _ = self.get_landmark_coord(landmarks, self.FACIAL_LANDMARKS['right_eye_corner'], width, height)
        eye_dx = right_eye_x - left_eye_x
        eye_dy = right_eye_y - left_eye_y
        roll = math.degrees(math.atan2(eye_dy, eye_dx)) if eye_dx != 0 else 0.0
        
        # Overall symmetry score
        symmetry_score = self.calculate_symmetry_score(landmarks, width, height)
        
        return LandmarkMetrics(
            mouth_corner_deviation_mm=mouth_asymmetry * 0.1,  # Rough pixel to mm conversion
            eyebrow_height_difference_mm=eyebrow_height_diff * 0.1,
            eye_closure_left_percent=left_eye_closure,
            eye_closure_right_percent=right_eye_closure,
            facial_midline_deviation=abs(midline_x - width/2) * 0.1,
            iris_left_center=(left_iris_x, left_iris_y),
            iris_right_center=(right_iris_x, right_iris_y),
            head_pose_angles=(pitch, yaw, roll),
            symmetry_score=symmetry_score,
            ear_left=ear_left,
            ear_right=ear_right
        )
    
    def create_annotated_image(self, image: np.ndarray, 
                             landmarks: landmark_pb2.NormalizedLandmarkList) -> np.ndarray:
        """Create annotated image with key medical landmarks"""
        annotated = image.copy()
        height, width = image.shape[:2]
        
        # Key medical landmarks with colors
        key_points = [
            (self.FACIAL_LANDMARKS['left_mouth_corner'], (0, 255, 0), "LM"),
            (self.FACIAL_LANDMARKS['right_mouth_corner'], (0, 255, 0), "RM"),
            (self.FACIAL_LANDMARKS['left_iris_center'], (255, 0, 0), "LI"),
            (self.FACIAL_LANDMARKS['right_iris_center'], (255, 0, 0), "RI"),
            (self.FACIAL_LANDMARKS['nose_tip'], (0, 0, 255), "N"),
            (self.FACIAL_LANDMARKS['left_eyebrow_inner'], (255, 255, 0), "LB"),
            (self.FACIAL_LANDMARKS['right_eyebrow_inner'], (255, 255, 0), "RB"),
        ]
        
        for landmark_idx, color, label in key_points:
            if landmark_idx < len(landmarks.landmark):
                x, y, _ = self.get_landmark_coord(landmarks, landmark_idx, width, height)
                cv2.circle(annotated, (int(x), int(y)), 4, color, -1)
                cv2.putText(annotated, label, (int(x)+8, int(y)-8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw facial midline
        nose_x, _, _ = self.get_landmark_coord(landmarks, self.FACIAL_LANDMARKS['nose_tip'], width, height)
        cv2.line(annotated, (int(nose_x), 0), (int(nose_x), height), (255, 255, 0), 2)
        
        # Draw eye regions for EAR visualization
        for eye_points, color in [(self.LEFT_EYE_POINTS, (0, 255, 255)), (self.RIGHT_EYE_POINTS, (0, 255, 255))]:
            pts = []
            for idx in eye_points:
                x, y, _ = self.get_landmark_coord(landmarks, idx, width, height)
                pts.append([int(x), int(y)])
            if len(pts) > 2:
                pts = np.array(pts, np.int32)
                cv2.polylines(annotated, [pts], True, color, 1)
        
        return annotated

def metrics_to_json(metrics: LandmarkMetrics) -> str:
    """Convert metrics to JSON string for Ollama prompt"""
    return json.dumps({
        "version": "1.0",
        "analysis_type": "facial_landmark_medical_assessment",
        "landmark_format": "mediapipe_478_with_pb2",
        "metrics": {
            "mouth_corner_horizontal_deviation_mm": round(metrics.mouth_corner_deviation_mm, 2),
            "eyebrow_height_difference_mm": round(metrics.eyebrow_height_difference_mm, 2),
            "eye_closure_left_percent": round(metrics.eye_closure_left_percent, 1),
            "eye_closure_right_percent": round(metrics.eye_closure_right_percent, 1),
            "facial_midline_deviation_mm": round(metrics.facial_midline_deviation, 2),
            "facial_symmetry_score": round(metrics.symmetry_score, 3),
            "eye_aspect_ratio": {
                "left": round(metrics.ear_left, 4),
                "right": round(metrics.ear_right, 4),
                "difference": round(abs(metrics.ear_left - metrics.ear_right), 4)
            },
            "iris_centers": {
                "left": [round(metrics.iris_left_center[0], 1), round(metrics.iris_left_center[1], 1)],
                "right": [round(metrics.iris_right_center[0], 1), round(metrics.iris_right_center[1], 1)]
            },
            "head_pose_degrees": {
                "pitch": round(metrics.head_pose_angles[0], 1),
                "yaw": round(metrics.head_pose_angles[1], 1),
                "roll": round(metrics.head_pose_angles[2], 1)
            }
        }
    }, indent=2)
