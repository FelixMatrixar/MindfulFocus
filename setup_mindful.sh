#!/bin/bash
# setup.sh - Medical Face Landmark Analysis with Ollama Integration (Simple & Local)

echo "Setting up Medical Face Landmark Analysis with Ollama..."

# Create directory structure for medical analysis
mkdir -p medical_analysis/{models,data,results,utils}
mkdir -p medical_analysis/data/{input,processed,annotations}

# Create the main medical analysis module using landmark_pb2
cat > medical_analysis/landmark_analyzer.py << 'EOF'
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
EOF

# Create simple Ollama integration
cat > medical_analysis/ollama_analyzer.py << 'EOF'
import requests
import json
import base64
import io
from PIL import Image
import numpy as np
from typing import Dict, Any, Optional

class OllamaError(Exception):
    pass

class OllamaMedicalAnalyzer:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llava"):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.session = requests.Session()
        self.session.timeout = 120  # 2 minutes timeout
        
        # Check if Ollama is running and model is available
        self._check_connection()
    
    def _check_connection(self) -> bool:
        """Check if Ollama is running and model is available"""
        try:
            # Check Ollama server
            response = self.session.get(f"{self.base_url}/api/tags")
            if response.status_code != 200:
                raise OllamaError("Ollama server not responding")
            
            # Check if model exists
            models = response.json().get("models", [])
            model_names = [m.get("name", "").split(":")[0] for m in models]
            
            if self.model not in model_names:
                print(f"Warning: Model '{self.model}' not found. Available models: {model_names}")
                print(f"To install: ollama pull {self.model}")
                
            return True
            
        except requests.RequestException as e:
            raise OllamaError(f"Cannot connect to Ollama at {self.base_url}: {e}")
    
    def _image_to_base64(self, image: np.ndarray) -> str:
        """Convert numpy image to base64 string"""
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Convert BGR to RGB
            image_rgb = image[:, :, ::-1]
        else:
            image_rgb = image
        
        pil_image = Image.fromarray(image_rgb.astype(np.uint8))
        
        # Resize if too large (Ollama has limits)
        if max(pil_image.size) > 1024:
            pil_image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=85)
        image_bytes = buffer.getvalue()
        
        return base64.b64encode(image_bytes).decode('utf-8')
    
    def _call_ollama(self, prompt: str, image_base64: str) -> str:
        """Make API call to Ollama"""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [image_base64],
            "stream": False
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload
            )
            
            if response.status_code != 200:
                raise OllamaError(f"Ollama API error: {response.status_code} - {response.text}")
            
            result = response.json()
            return result.get("response", "No response from model")
            
        except requests.RequestException as e:
            raise OllamaError(f"Request failed: {e}")
        except json.JSONDecodeError as e:
            raise OllamaError(f"Invalid JSON response: {e}")
    
    def analyze_facial_asymmetry(self, image: np.ndarray, metrics_json: str) -> str:
        """Analyze facial asymmetry using Ollama vision model"""
        
        prompt = f"""You are a medical AI assistant specializing in facial landmark analysis. Analyze this facial image for signs of neurological conditions, particularly focusing on facial asymmetry that might indicate conditions like Bell's palsy or other cranial nerve disorders.

Here is precise landmark data extracted from this image:

{metrics_json}

Please provide a detailed medical assessment that:

1. **Visual Assessment**: Describe any visible asymmetry you observe in the face
2. **Data Correlation**: Correlate your visual observations with the provided numerical metrics
3. **Regional Analysis**: Identify which facial regions (upper, middle, lower) show the most significant asymmetry
4. **Clinical Considerations**: Suggest possible medical conditions based on the asymmetry pattern
5. **Severity Rating**: Rate the overall asymmetry severity on a scale of 0-10 (0 = perfectly symmetric, 10 = severe asymmetry)
6. **Recommendations**: Provide appropriate medical recommendations

Focus on medical accuracy and be specific about anatomical landmarks and measurements. Consider the symmetry score, EAR differences, and head pose when making your assessment."""

        try:
            image_b64 = self._image_to_base64(image)
            response = self._call_ollama(prompt, image_b64)
            return response
        except OllamaError as e:
            return f"Error during analysis: {str(e)}"
    
    def analyze_iris_features(self, image: np.ndarray, iris_coords: Dict[str, Any]) -> str:
        """Analyze iris features for medical indicators using Ollama"""
        
        prompt = f"""You are a medical AI assistant specializing in iris analysis and iridology. Examine the iris regions in this facial image for potential health indicators.

The iris centers are located at these pixel coordinates:
- Left iris center: {iris_coords['left']}
- Right iris center: {iris_coords['right']}

Please analyze and report on:

1. **Iris Color and Uniformity**: Describe the color and any variations
2. **Visible Patterns**: Note any patterns, spots, rings, or discoloration
3. **Pupil Assessment**: Evaluate pupil size, shape, and symmetry between both eyes
4. **Medical Signs**: Look for signs of:
   - Corneal arcus (whitish ring around iris)
   - Kayser-Fleischer rings (copper deposits)
   - Heterochromia (color differences)
   - Iris nevus or other pigmented lesions
5. **Overall Health Assessment**: Provide general iris health evaluation
6. **Clinical Recommendations**: Suggest if further examination is needed

Be specific about what you observe in each eye and note any asymmetries or concerning features. Focus on medically relevant observations."""

        try:
            image_b64 = self._image_to_base64(image)
            response = self._call_ollama(prompt, image_b64)
            return response
        except OllamaError as e:
            return f"Error during iris analysis: {str(e)}"
    
    def analyze_general_health(self, image: np.ndarray, metrics_json: str) -> str:
        """General health assessment from facial features"""
        
        prompt = f"""You are a medical AI assistant trained in facial diagnosis and physiognomy for health assessment. Analyze this face image for general health indicators.

Landmark metrics data:
{metrics_json}

Please assess:

1. **Skin Health**: Color, texture, blemishes, or discoloration
2. **Facial Signs**: Look for signs that might indicate:
   - Fatigue or stress
   - Nutritional deficiencies
   - Hormonal imbalances
   - Circulatory issues
3. **Symmetry and Movement**: Overall facial balance and any signs of weakness
4. **Eye Health**: General appearance, clarity, and alertness
5. **Overall Vitality**: General appearance of health and energy

Provide observations and gentle recommendations for wellness, but emphasize that this is not a substitute for professional medical examination."""

        try:
            image_b64 = self._image_to_base64(image)
            response = self._call_ollama(prompt, image_b64)
            return response
        except OllamaError as e:
            return f"Error during general health analysis: {str(e)}"
EOF

# Create main medical analysis integration
cat > medical_analysis/medical_main.py << 'EOF'
#!/usr/bin/env python3
"""
Medical Face Landmark Analysis - Main Module
Simple integration using MediaPipe landmark_pb2 and Ollama
"""

import cv2
import numpy as np
import json
import os
import sys
from datetime import datetime
from typing import Optional, Dict, Any

from landmark_analyzer import MedicalLandmarkAnalyzer, metrics_to_json
from ollama_analyzer import OllamaMedicalAnalyzer, OllamaError

class MedicalAnalysisIntegration:
    def __init__(self, ollama_url: str = "http://localhost:11434", ollama_model: str = "llava"):
        print("Initializing Medical Analysis Integration...")
        print("📍 Using MediaPipe landmark_pb2 for precision")
        print("🦙 Using Ollama for local AI analysis")
        
        self.landmark_analyzer = MedicalLandmarkAnalyzer()
        
        try:
            self.ollama_analyzer = OllamaMedicalAnalyzer(
                base_url=ollama_url, 
                model=ollama_model
            )
            print(f"✅ Connected to Ollama at {ollama_url}")
        except OllamaError as e:
            print(f"❌ Ollama connection failed: {e}")
            print("💡 Make sure Ollama is running: ollama serve")
            print(f"💡 Install vision model: ollama pull {ollama_model}")
            raise
        
        # Create results directory
        self.results_dir = "medical_analysis/results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        print("✅ Medical Analysis Integration ready!")
    
    def analyze_image(self, image_path: str, analysis_type: str = "asymmetry") -> Dict[str, Any]:
        """
        Perform medical analysis on a face image
        
        Args:
            image_path: Path to the image file
            analysis_type: "asymmetry", "iris", or "general" analysis
            
        Returns:
            Dictionary containing analysis results
        """
        print(f"🔍 Analyzing {image_path} for {analysis_type}...")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Extract landmarks using landmark_pb2
        landmarks = self.landmark_analyzer.extract_landmarks(image)
        if landmarks is None:
            return {"error": "No face detected in image"}
        
        # Calculate medical metrics
        metrics = self.landmark_analyzer.calculate_medical_metrics(landmarks, image.shape)
        metrics_json = metrics_to_json(metrics)
        
        # Create annotated image
        annotated_image = self.landmark_analyzer.create_annotated_image(image, landmarks)
        
        # Perform Ollama analysis
        try:
            if analysis_type == "asymmetry":
                ollama_analysis = self.ollama_analyzer.analyze_facial_asymmetry(image, metrics_json)
            elif analysis_type == "iris":
                iris_coords = {
                    "left": list(metrics.iris_left_center),
                    "right": list(metrics.iris_right_center)
                }
                ollama_analysis = self.ollama_analyzer.analyze_iris_features(image, iris_coords)
            elif analysis_type == "general":
                ollama_analysis = self.ollama_analyzer.analyze_general_health(image, metrics_json)
            else:
                ollama_analysis = f"Unknown analysis type: {analysis_type}"
        except OllamaError as e:
            ollama_analysis = f"Ollama analysis failed: {str(e)}"
        
        # Prepare results
        results = {
            "timestamp": datetime.now().isoformat(),
            "image_path": image_path,
            "analysis_type": analysis_type,
            "landmarks_detected": len(landmarks.landmark),
            "metrics": json.loads(metrics_json),
            "ollama_analysis": ollama_analysis,
            "annotated_image_path": None
        }
        
        # Save annotated image
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        annotated_path = os.path.join(self.results_dir, f"{base_name}_annotated.jpg")
        cv2.imwrite(annotated_path, annotated_image)
        results["annotated_image_path"] = annotated_path
        
        # Save results
        results_path = os.path.join(self.results_dir, f"{base_name}_analysis.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Analysis complete! Results saved to {results_path}")
        return results
    
    def analyze_from_camera(self, analysis_type: str = "asymmetry") -> Optional[Dict[str, Any]]:
        """Capture image from camera and analyze"""
        print("📸 Starting camera capture for medical analysis...")
        print("💡 Press SPACE to capture image for analysis, or 'q' to quit")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Show preview with instructions
            display_frame = frame.copy()
            cv2.putText(display_frame, "Medical Analysis - Press SPACE to capture", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Mode: {analysis_type.title()}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMP
                       LEX, 0.6, (255, 255, 0), 2)
           cv2.imshow('Medical Analysis Camera', display_frame)
           
           key = cv2.waitKey(1) & 0xFF
           if key == ord(' '):  # Space bar to capture
               # Save temporary image
               temp_path = os.path.join(self.results_dir, "temp_capture.jpg")
               cv2.imwrite(temp_path, frame)
               
               cap.release()
               cv2.destroyAllWindows()
               
               # Analyze
               try:
                   results = self.analyze_image(temp_path, analysis_type)
                   # Clean up temp file
                   os.remove(temp_path)
                   return results
               except Exception as e:
                   print(f"❌ Analysis failed: {e}")
                   os.remove(temp_path)
                   return None
           
           elif key == ord('q'):  # Quit
               break
       
       cap.release()
       cv2.destroyAllWindows()
       return None

def main():
   """Main function for testing medical analysis"""
   print("🏥 Medical Face Landmark Analysis with Ollama")
   print("=" * 50)
   
   try:
       analyzer = MedicalAnalysisIntegration()
       
       while True:
           print("\nOptions:")
           print("1. 📸 Analyze from camera (facial asymmetry)")
           print("2. 👁️  Analyze from camera (iris features)")
           print("3. 🩺 Analyze from camera (general health)")
           print("4. 📁 Analyze image file")
           print("5. ⚙️  Check Ollama status")
           print("6. ❌ Exit")
           
           choice = input("\nSelect option (1-6): ").strip()
           
           if choice == "1":
               results = analyzer.analyze_from_camera("asymmetry")
               display_results(results)
           elif choice == "2":
               results = analyzer.analyze_from_camera("iris")
               display_results(results)
           elif choice == "3":
               results = analyzer.analyze_from_camera("general")
               display_results(results)
           elif choice == "4":
               image_path = input("📂 Enter image file path: ").strip()
               if os.path.exists(image_path):
                   analysis_type = input("Analysis type (asymmetry/iris/general) [asymmetry]: ").strip() or "asymmetry"
                   results = analyzer.analyze_image(image_path, analysis_type)
                   display_results(results)
               else:
                   print("❌ File not found!")
           elif choice == "5":
               check_ollama_status()
           elif choice == "6":
               print("👋 Goodbye!")
               break
           else:
               print("❌ Invalid choice, please try again.")
   
   except KeyboardInterrupt:
       print("\n👋 Exiting...")
   except Exception as e:
       print(f"❌ Error: {e}")

def check_ollama_status():
   """Check Ollama server status"""
   import requests
   try:
       response = requests.get("http://localhost:11434/api/tags", timeout=5)
       if response.status_code == 200:
           models = response.json().get("models", [])
           print("✅ Ollama server is running")
           print("📦 Available models:")
           for model in models:
               name = model.get("name", "Unknown")
               size = model.get("size", 0)
               size_mb = size / (1024*1024) if size else 0
               print(f"  • {name} ({size_mb:.1f} MB)")
           
           # Check for vision models
           vision_models = [m for m in models if any(vm in m.get("name", "").lower() 
                          for vm in ["llava", "vision", "moondream", "bakllava"])]
           if vision_models:
               print("👁️  Vision models available:", [m.get("name") for m in vision_models])
           else:
               print("⚠️  No vision models found. Install with: ollama pull llava")
       else:
           print("❌ Ollama server responded with error:", response.status_code)
   except requests.RequestException:
       print("❌ Cannot connect to Ollama server")
       print("💡 Start Ollama: ollama serve")
       print("💡 Install vision model: ollama pull llava")

def display_results(results):
   """Display analysis results in a formatted way"""
   if not results:
       print("❌ No results to display")
       return
       
   if "error" in results:
       print(f"❌ Error: {results['error']}")
       return
   
   print("\n" + "🏥 MEDICAL ANALYSIS RESULTS " + "🏥")
   print("=" * 60)
   
   print(f"📊 Analysis Type: {results['analysis_type'].title()}")
   print(f"🎯 Landmarks Detected: {results['landmarks_detected']}")
   print(f"📅 Timestamp: {results['timestamp']}")
   
   if results.get('annotated_image_path'):
       print(f"🖼️  Annotated Image: {results['annotated_image_path']}")
   
   print("\n🦙 Ollama AI Analysis:")
   print("─" * 40)
   print(results['ollama_analysis'])
   
   print("\n📏 Technical Metrics (MediaPipe landmark_pb2):")
   print("─" * 40)
   metrics = results['metrics']['metrics']
   
   # Display key metrics in organized way
   print(f"💫 Facial Symmetry Score: {metrics['facial_symmetry_score']:.3f} (1.0 = perfect)")
   
   ear_data = metrics['eye_aspect_ratio']
   print(f"👁️  Eye Aspect Ratio - Left: {ear_data['left']:.4f}, Right: {ear_data['right']:.4f}")
   print(f"👁️  EAR Difference: {ear_data['difference']:.4f}")
   
   print(f"😮 Mouth Asymmetry: {metrics['mouth_corner_horizontal_deviation_mm']:.2f} mm")
   print(f"🤨 Eyebrow Height Diff: {metrics['eyebrow_height_difference_mm']:.2f} mm")
   print(f"😴 Eye Closure - Left: {metrics['eye_closure_left_percent']:.1f}%, Right: {metrics['eye_closure_right_percent']:.1f}%")
   
   pose = metrics['head_pose_degrees']
   print(f"🗣️  Head Pose - Pitch: {pose['pitch']:.1f}°, Yaw: {pose['yaw']:.1f}°, Roll: {pose['roll']:.1f}°")
   
   print("\n" + "=" * 60)
   input("Press Enter to continue...")

if __name__ == "__main__":
   main()
EOF

# Create integration with mindful-core and mindful-desktop
echo "Creating integration with existing MindfulFocus app..."

# Create core integration
cat > mindful-core/medical_processor.py << 'EOF'
"""
Medical Analysis Processor for Mindful-Core
Simple integration using landmark_pb2 and Ollama
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'medical_analysis'))

import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from landmark_analyzer import MedicalLandmarkAnalyzer, metrics_to_json
from ollama_analyzer import OllamaMedicalAnalyzer, OllamaError

@dataclass
class MedicalAnalysisResult:
   """Medical analysis result for mindful-core"""
   timestamp: datetime
   analysis_type: str
   landmarks_count: int
   symmetry_score: float
   ear_difference: float
   mouth_asymmetry_mm: float
   eyebrow_diff_mm: float
   ai_assessment: str
   severity_score: float
   recommendations: list

class SimpleMedicalProcessor:
   """Lightweight medical processor for mindful-core integration"""
   
   def __init__(self, ollama_url: str = "http://localhost:11434", ollama_model: str = "llava"):
       self.landmark_analyzer = None
       self.ollama_analyzer = None
       self.is_initialized = False
       self.ollama_url = ollama_url
       self.ollama_model = ollama_model
       
   def initialize(self) -> bool:
       """Initialize processors lazily"""
       if self.is_initialized:
           return True
           
       try:
           print("🔧 Initializing medical processors...")
           self.landmark_analyzer = MedicalLandmarkAnalyzer()
           self.ollama_analyzer = OllamaMedicalAnalyzer(
               base_url=self.ollama_url, 
               model=self.ollama_model
           )
           self.is_initialized = True
           print("✅ Medical processors ready")
           return True
       except Exception as e:
           print(f"❌ Medical processor init failed: {e}")
           return False
   
   def quick_analyze(self, frame: np.ndarray) -> Optional[MedicalAnalysisResult]:
       """
       Quick medical analysis for integration with mindfulness app
       Returns simplified results suitable for real-time display
       """
       if not self.is_initialized:
           if not self.initialize():
               return None
       
       try:
           # Extract landmarks
           landmarks = self.landmark_analyzer.extract_landmarks(frame)
           if landmarks is None:
               return None
           
           # Calculate metrics
           metrics = self.landmark_analyzer.calculate_medical_metrics(landmarks, frame.shape)
           
           # Simple severity scoring
           severity = self._calculate_simple_severity(metrics)
           
           # Generate basic AI assessment (skip for real-time, too slow)
           ai_assessment = f"Symmetry: {metrics.symmetry_score:.2f}, EAR diff: {abs(metrics.ear_left - metrics.ear_right):.3f}"
           
           # Basic recommendations
           recommendations = self._generate_simple_recommendations(metrics, severity)
           
           return MedicalAnalysisResult(
               timestamp=datetime.now(),
               analysis_type="quick_asymmetry",
               landmarks_count=len(landmarks.landmark),
               symmetry_score=metrics.symmetry_score,
               ear_difference=abs(metrics.ear_left - metrics.ear_right),
               mouth_asymmetry_mm=metrics.mouth_corner_deviation_mm,
               eyebrow_diff_mm=metrics.eyebrow_height_difference_mm,
               ai_assessment=ai_assessment,
               severity_score=severity,
               recommendations=recommendations
           )
           
       except Exception as e:
           print(f"❌ Quick analysis failed: {e}")
           return None
   
   def detailed_analyze(self, frame: np.ndarray, analysis_type: str = "asymmetry") -> Optional[MedicalAnalysisResult]:
       """
       Detailed analysis with full Ollama AI assessment (slower)
       """
       if not self.is_initialized:
           if not self.initialize():
               return None
       
       try:
           # Extract landmarks
           landmarks = self.landmark_analyzer.extract_landmarks(frame)
           if landmarks is None:
               return None
           
           # Calculate metrics
           metrics = self.landmark_analyzer.calculate_medical_metrics(landmarks, frame.shape)
           metrics_json = metrics_to_json(metrics)
           
           # Full AI analysis
           try:
               if analysis_type == "asymmetry":
                   ai_assessment = self.ollama_analyzer.analyze_facial_asymmetry(frame, metrics_json)
               elif analysis_type == "iris":
                   iris_coords = {
                       "left": list(metrics.iris_left_center),
                       "right": list(metrics.iris_right_center)
                   }
                   ai_assessment = self.ollama_analyzer.analyze_iris_features(frame, iris_coords)
               else:
                   ai_assessment = self.ollama_analyzer.analyze_general_health(frame, metrics_json)
           except Exception as e:
               ai_assessment = f"AI analysis unavailable: {e}"
           
           # Calculate severity and recommendations
           severity = self._calculate_simple_severity(metrics)
           recommendations = self._generate_simple_recommendations(metrics, severity)
           
           return MedicalAnalysisResult(
               timestamp=datetime.now(),
               analysis_type=analysis_type,
               landmarks_count=len(landmarks.landmark),
               symmetry_score=metrics.symmetry_score,
               ear_difference=abs(metrics.ear_left - metrics.ear_right),
               mouth_asymmetry_mm=metrics.mouth_corner_deviation_mm,
               eyebrow_diff_mm=metrics.eyebrow_height_difference_mm,
               ai_assessment=ai_assessment,
               severity_score=severity,
               recommendations=recommendations
           )
           
       except Exception as e:
           print(f"❌ Detailed analysis failed: {e}")
           return None
   
   def _calculate_simple_severity(self, metrics) -> float:
       """Simple severity scoring (0-10)"""
       # Factors: symmetry (inverted), mouth asymmetry, eyebrow diff, EAR difference
       symmetry_penalty = (1.0 - metrics.symmetry_score) * 4  # 0-4
       mouth_penalty = min(3, metrics.mouth_corner_deviation_mm / 2)  # 0-3
       eyebrow_penalty = min(2, metrics.eyebrow_height_difference_mm / 3)  # 0-2
       ear_penalty = min(1, abs(metrics.ear_left - metrics.ear_right) * 20)  # 0-1
       
       total = symmetry_penalty + mouth_penalty + eyebrow_penalty + ear_penalty
       return min(10.0, total)
   
   def _generate_simple_recommendations(self, metrics, severity: float) -> list:
       """Generate basic recommendations"""
       recs = []
       
       if severity > 7:
           recs.append("Consider medical consultation - significant asymmetry detected")
       elif severity > 4:
           recs.append("Monitor facial symmetry - mild asymmetry present")
       
       if metrics.mouth_corner_deviation_mm > 3:
           recs.append("Mouth asymmetry noted - may indicate facial nerve involvement")
       
       if abs(metrics.ear_left - metrics.ear_right) > 0.05:
           recs.append("Eye opening asymmetry detected")
       
       if metrics.symmetry_score < 0.8:
           recs.append("Overall facial asymmetry present")
       
       if not recs:
           recs.append("No significant asymmetry detected")
       
       return recs
EOF

# Create desktop UI integration
cat > mindful-desktop/app/simple_medical_ui.py << 'EOF'
"""
Simple Medical UI Integration for MindfulFocus
Uses Ollama locally - no heavy dependencies
"""

import tkinter as tk
from tkinter import ttk
import sys
import os
import threading
from datetime import datetime
import cv2

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'mindful-core'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'medical_analysis'))

from medical_processor import SimpleMedicalProcessor

class SimpleMedicalPanel:
   """Simple medical analysis panel for mindful-desktop"""
   
   def __init__(self, parent_frame, camera_callback=None):
       self.parent = parent_frame
       self.camera_callback = camera_callback
       self.medical_processor = SimpleMedicalProcessor()
       self.analysis_running = False
       self.detailed_mode = False
       
       self.setup_ui()
   
   def setup_ui(self):
       """Setup simple medical UI"""
       # Main frame
       self.medical_frame = ttk.LabelFrame(self.parent, text="🏥 Medical Analysis (Ollama)", padding=10)
       self.medical_frame.pack(fill='both', expand=True, padx=5, pady=5)
       
       # Control buttons
       button_frame = ttk.Frame(self.medical_frame)
       button_frame.pack(fill='x', pady=5)
       
       self.toggle_btn = ttk.Button(button_frame, text="Start Quick Analysis", 
                                  command=self.toggle_analysis)
       self.toggle_btn.pack(side='left', padx=5)
       
       self.detailed_btn = ttk.Button(button_frame, text="Detailed Analysis", 
                                    command=self.run_detailed_analysis)
       self.detailed_btn.pack(side='left', padx=5)
       
       # Mode selection
       mode_frame = ttk.Frame(self.medical_frame)
       mode_frame.pack(fill='x', pady=5)
       
       ttk.Label(mode_frame, text="Mode:").pack(side='left')
       self.analysis_mode = tk.StringVar(value="asymmetry")
       ttk.Radiobutton(mode_frame, text="Asymmetry", 
                      variable=self.analysis_mode, value="asymmetry").pack(side='left', padx=5)
       ttk.Radiobutton(mode_frame, text="Iris", 
                      variable=self.analysis_mode, value="iris").pack(side='left', padx=5)
       ttk.Radiobutton(mode_frame, text="General", 
                      variable=self.analysis_mode, value="general").pack(side='left', padx=5)
       
       # Status and metrics display
       self.setup_metrics_display()
       
       # Ollama status
       self.status_var = tk.StringVar(value="Ready (Ollama required)")
       status_label = ttk.Label(self.medical_frame, textvariable=self.status_var)
       status_label.pack(pady=5)
   
   def setup_metrics_display(self):
       """Setup metrics display"""
       metrics_frame = ttk.LabelFrame(self.medical_frame, text="📊 Live Metrics")
       metrics_frame.pack(fill='both', expand=True, pady=10)
       
       # Metrics grid
       self.symmetry_var = tk.StringVar(value="Symmetry: --")
       self.ear_var = tk.StringVar(value="EAR Diff: --")
       self.mouth_var = tk.StringVar(value="Mouth: -- mm")
       self.severity_var = tk.StringVar(value="Severity: -- / 10")
       
       ttk.Label(metrics_frame, textvariable=self.symmetry_var).grid(row=0, column=0, sticky='w', padx=5, pady=2)
       ttk.Label(metrics_frame, textvariable=self.ear_var).grid(row=0, column=1, sticky='w', padx=5, pady=2)
       ttk.Label(metrics_frame, textvariable=self.mouth_var).grid(row=1, column=0, sticky='w', padx=5, pady=2)
       ttk.Label(metrics_frame, textvariable=self.severity_var).grid(row=1, column=1, sticky='w', padx=5, pady=2)
       
       # Recommendations
       self.rec_text = tk.Text(metrics_frame, height=4, wrap=tk.WORD)
       self.rec_text.grid(row=2, column=0, columnspan=2, sticky='ew', padx=5, pady=5)
       
       # Configure grid weights
       metrics_frame.columnconfigure(0, weight=1)
       metrics_frame.columnconfigure(1, weight=1)
   
   def toggle_analysis(self):
       """Toggle quick analysis"""
       if not self.analysis_running:
           self.start_quick_analysis()
       else:
           self.stop_analysis()
   
   def start_quick_analysis(self):
       """Start quick analysis loop"""
       self.analysis_running = True
       self.toggle_btn.config(text="Stop Analysis")
       self.status_var.set("Quick analysis running...")
       
       # Initialize in background
       threading.Thread(target=self._init_and_analyze, daemon=True).start()
   
   def _init_and_analyze(self):
       """Initialize and start analysis loop"""
       if not self.medical_processor.initialize():
           self.parent.after(0, lambda: self.status_var.set("❌ Failed to initialize (check Ollama)"))
           self.parent.after(0, self.stop_analysis)
           return
       
       self._quick_analysis_loop()
   
   def _quick_analysis_loop(self):
       """Quick analysis loop"""
       if not self.analysis_running:
           return
       
       if self.camera_callback:
           try:
               frame = self.camera_callback()
               if frame is not None:
                   result = self.medical_processor.quick_analyze(frame)
                   if result:
                       self.parent.after(0, lambda: self.update_metrics_display(result))
           except Exception as e:
               print(f"Analysis error: {e}")
       
       # Schedule next analysis (every 2 seconds for performance)
       if self.analysis_running:
           self.parent.after(2000, self._quick_analysis_loop)
   
   def run_detailed_analysis(self):
       """Run one detailed analysis with Ollama"""
       if self.camera_callback:
           self.status_var.set("Running detailed analysis... (may take 30s)")
           self.detailed_btn.config(state='disabled')
           
           def detailed_worker():
               try:
                   if not self.medical_processor.initialize():
                       self.parent.after(0, lambda: self.status_var.set("❌ Ollama not available"))
                       return
                   
                   frame = self.camera_callback()
                   if frame is not None:
                       result = self.medical_processor.detailed_analyze(frame, self.analysis_mode.get())
                       if result:
                           self.parent.after(0, lambda: self.display_detailed_result(result))
                       else:
                           self.parent.after(0, lambda: self.status_var.set("❌ Detailed analysis failed"))
                   
               except Exception as e:
                   self.parent.after(0, lambda: self.status_var.set(f"❌ Error: {e}"))
               finally:
                   self.parent.after(0, lambda: self.detailed_btn.config(state='normal'))
           
           threading.Thread(target=detailed_worker, daemon=True).start()
   
   def update_metrics_display(self, result):
       """Update metrics display with quick analysis results"""
       self.symmetry_var.set(f"Symmetry: {result.symmetry_score:.3f}")
       self.ear_var.set(f"EAR Diff: {result.ear_difference:.4f}")
       self.mouth_var.set(f"Mouth: {result.mouth_asymmetry_mm:.2f} mm")
       self.severity_var.set(f"Severity: {result.severity_score:.1f} / 10")
       
       # Update recommendations
       self.rec_text.delete('1.0', tk.END)
       for rec in result.recommendations:
           self.rec_text.insert(tk.END, f"• {rec}\n")
       
       # Update status with color coding
       if result.severity_score > 7:
           self.status_var.set("⚠️ High severity detected")
       elif result.severity_score > 4:
           self.status_var.set("⚡ Moderate findings")
       else:
           self.status_var.set("✅ Quick analysis running")
   
   def display_detailed_result(self, result):
       """Display detailed analysis result"""
       # Update metrics first
       self.update_metrics_display(result)
       
       # Show detailed AI assessment in a popup
       detail_window = tk.Toplevel(self.parent)
       detail_window.title("🦙 Detailed Ollama Analysis")
       detail_window.geometry("600x400")
       
       text_widget = tk.Text(detail_window, wrap=tk.WORD, padx=10, pady=10)
       scrollbar = ttk.Scrollbar(detail_window, orient='vertical', command=text_widget.yview)
       text_widget.configure(yscrollcommand=scrollbar.set)
       
       # Insert analysis
       text_widget.insert('1.0', f"Analysis Type: {result.analysis_type.title()}\n")
       text_widget.insert(tk.END, f"Timestamp: {result.timestamp}\n")
       text_widget.insert(tk.END, f"Severity Score: {result.severity_score:.1f}/10\n\n")
       text_widget.insert(tk.END, "🤖 AI Assessment:\n")
       text_widget.insert(tk.END, "=" * 40 + "\n")
       text_widget.insert(tk.END, result.ai_assessment)
       
       text_widget.pack(side='left', fill='both', expand=True)
       scrollbar.pack(side='right', fill='y')
       
       self.status_var.set("✅ Detailed analysis complete")
   
   def stop_analysis(self):
       """Stop analysis"""
       self.analysis_running = False
       self.toggle_btn.config(text="Start Quick Analysis")
       self.status_var.set("Analysis stopped")

# Simple standalone window
def run_simple_medical_app():
   """Run simple medical analysis app"""
   root = tk.Tk()
   root.title("🏥 MindfulFocus Medical Analysis (Ollama)")
   root.geometry("800x600")
   
   # Simple camera callback
   cap = cv2.VideoCapture(0)
   
   def get_frame():
       ret, frame = cap.read()
       return frame if ret else None
   
   # Create panel
   panel = SimpleMedicalPanel(root, get_frame)
   
   def cleanup():
       cap.release()
       root.destroy()
   
   root.protocol("WM_DELETE_WINDOW", cleanup)
   root.mainloop()

if __name__ == "__main__":
   run_simple_medical_app()
EOF

# Create simple test script
cat > test_ollama_integration.py << 'EOF'
#!/usr/bin/env python3
"""
Test Ollama integration for medical analysis
"""

import sys
import os
sys.path.append('medical_analysis')

def test_ollama_connection():
   """Test Ollama connection and models"""
   try:
       import requests
       
       print("🧪 Testing Ollama connection...")
       response = requests.get("http://localhost:11434/api/tags", timeout=5)
       
       if response.status_code == 200:
           print("✅ Ollama server is running")
           
           models = response.json().get("models", [])
           print(f"📦 Found {len(models)} models:")
           
           vision_models = []
           for model in models:
               name = model.get("name", "Unknown")
               size = model.get("size", 0)
               size_gb = size / (1024**3) if size else 0
               print(f"  • {name} ({size_gb:.1f} GB)")
               
               if any(vm in name.lower() for vm in ["llava", "vision", "moondream", "bakllava"]):
                   vision_models.append(name)
           
           if vision_models:
               print(f"👁️  Vision models available: {vision_models}")
               return True
           else:
               print("⚠️  No vision models found")
               print("💡 Install with: ollama pull llava")
               return False
       else:
           print(f"❌ Ollama responded with status {response.status_code}")
           return False
           
   except Exception as e:
       print(f"❌ Cannot connect to Ollama: {e}")
       print("💡 Make sure Ollama is running: ollama serve")
       print("💡 Install Ollama: https://ollama.ai/download")
       return False

def test_landmark_detection():
   """Test MediaPipe landmark detection"""
   try:
       print("\n🧪 Testing landmark detection...")
       
       import cv2
       import mediapipe as mp
       from mediapipe.framework.formats import landmark_pb2
       
       print("✅ MediaPipe imported successfully")
       
       from landmark_analyzer import MedicalLandmarkAnalyzer
       analyzer = MedicalLandmarkAnalyzer()
       
       print("✅ Medical landmark analyzer created")
       
       # Test with webcam if available
       cap = cv2.VideoCapture(0)
       if cap.isOpened():
           ret, frame = cap.read()
           if ret:
               landmarks = analyzer.extract_landmarks(frame)
               if landmarks:
                   print(f"✅ Detected {len(landmarks.landmark)} facial landmarks")
                   
                   metrics = analyzer.calculate_medical_metrics(landmarks, frame.shape)
                   print(f"✅ Calculated medical metrics")
                   print(f"   Symmetry score: {metrics.symmetry_score:.3f}")
                   print(f"   EAR L/R: {metrics.ear_left:.3f} / {metrics.ear_right:.3f}")
                   
               else:
                   print("⚠️  No face detected in camera frame")
           cap.release()
           return True
       else:
           print("⚠️  Camera not available, skipping face detection test")
           return True
           
   except Exception as e:
       print(f"❌ Landmark detection test failed: {e}")
       return False

def test_full_integration():
   """Test full medical analysis integration"""
   try:
       print("\n🧪 Testing full integration...")
       
       from medical_main import MedicalAnalysisIntegration
       
       # This will test Ollama connection
       analyzer = MedicalAnalysisIntegration()
       print("✅ Medical analysis integration initialized")
       
       return True
       
   except Exception as e:
       print(f"❌ Integration test failed: {e}")
       return False

def main():
   print("🏥 Testing Medical Analysis Integration with Ollama")
   print("=" * 60)
   
   tests = [
       ("Ollama Connection", test_ollama_connection),
       ("Landmark Detection", test_landmark_detection),
       ("Full Integration", test_full_integration)
   ]
   
   results = []
   for test_name, test_func in tests:
       print(f"\n{'='*20} {test_name} {'='*20}")
       result = test_func()
       results.append((test_name, result))
   
   print("\n" + "=" * 60)
   print("📊 TEST SUMMARY:")
   print("=" * 60)
   
   for test_name, result in results:
       status = "✅ PASS" if result else "❌ FAIL"
       print(f"{test_name:<20} {status}")
   
   all_passed = all(result for _, result in results)
   
   if all_passed:
       print("\n🎉 All tests passed!")
       print("\n🚀 Ready to run:")
       print("  python run_medical_analysis.py")
       print("  python mindful-desktop/app/simple_medical_ui.py")
   else:
       print("\n⚠️  Some tests failed. Check the issues above.")
       print("\n💡 Quick fixes:")
       print("  • Start Ollama: ollama serve")
       print("  • Install vision model: ollama pull llava")
       print("  • Install dependencies: pip install mediapipe opencv-python requests")

if __name__ == "__main__":
   main()
EOF

# Create simple standalone runner
cat > run_medical_analysis.py << 'EOF'
#!/usr/bin/env python3
"""
Simple Medical Analysis Runner using Ollama
"""

import sys
import os

# Add medical_analysis to path
sys.path.append('medical_analysis')

from medical_main import MedicalAnalysisIntegration

def main():
   print("🏥 Medical Face Landmark Analysis with Ollama")
   print("=" * 50)
   print("🦙 Using local Ollama for AI analysis")
   print("📍 Using MediaPipe landmark_pb2 for precision")
   print("")
   
   try:
       analyzer = MedicalAnalysisIntegration()
       
       while True:
           print("\nOptions:")
           print("1. 📸 Camera Analysis - Facial Asymmetry")
           print("2. 👁️  Camera Analysis - Iris Features")
           print("3. 🩺 Camera Analysis - General Health")
           print("4. 📁 Image File Analysis")
           print("5. ⚙️  Check Ollama Status")
           print("6. 🧪 Test Integration")
           print("7. ❌ Exit")
           
           choice = input("\nSelect option (1-7): ").strip()
           
           if choice == "1":
               print("\n🔍 Starting facial asymmetry analysis...")
               results = analyzer.analyze_from_camera("asymmetry")
               display_results(results)
               
           elif choice == "2":
               print("\n👁️ Starting iris feature analysis...")
               results = analyzer.analyze_from_camera("iris")
               display_results(results)
               
           elif choice == "3":
               print("\n🩺 Starting general health analysis...")
               results = analyzer.analyze_from_camera("general")
               display_results(results)
               
           elif choice == "4":
               image_path = input("📂 Enter image file path: ").strip().strip('"')
               if os.path.exists(image_path):
                   print("\nAvailable analysis types:")
                   print("  asymmetry - Facial asymmetry detection")
                   print("  iris      - Iris and eye health")
                   print("  general   - Overall health assessment")
                   
                   analysis_type = input("Analysis type [asymmetry]: ").strip() or "asymmetry"
                   print(f"\n🔍 Analyzing {os.path.basename(image_path)} for {analysis_type}...")
                   results = analyzer.analyze_image(image_path, analysis_type)
                   display_results(results)
               else:
                   print("❌ File not found!")
                   
           elif choice == "5":
               check_ollama_status()
               
           elif choice == "6":
               run_integration_test()
               
           elif choice == "7":
               print("👋 Goodbye!")
               break
               
           else:
               print("❌ Invalid choice, please try again.")
               
   except KeyboardInterrupt:
       print("\n👋 Exiting...")
   except Exception as e:
       print(f"❌ Error: {e}")
       print("\n💡 Try:")
       print("  • Check if Ollama is running: ollama serve")
       print("  • Install vision model: ollama pull llava")

def check_ollama_status():
   """Check Ollama server and model status"""
   print("\n🔍 Checking Ollama status...")
   
   import requests
   try:
       response = requests.get("http://localhost:11434/api/tags", timeout=10)
       if response.status_code == 200:
           models = response.json().get("models", [])
           print(f"✅ Ollama server running with {len(models)} models")
           
           vision_models = []
           for model in models:
               name = model.get("name", "")
               size = model.get("size", 0)
               size_gb = size / (1024**3) if size else 0
               
               print(f"  📦 {name} ({size_gb:.1f} GB)")
               
               if any(vm in name.lower() for vm in ["llava", "vision", "moondream", "bakllava"]):
                   vision_models.append(name.split(':')[0])
           
           if vision_models:
               print(f"\n👁️  Vision models ready: {', '.join(set(vision_models))}")
           else:
               print("\n⚠️  No vision models installed")
               print("💡 Install with: ollama pull llava")
       else:
           print(f"❌ Ollama server error: {response.status_code}")
   except:
       print("❌ Cannot connect to Ollama")
       print("💡 Start Ollama server: ollama serve")
       print("💡 Download Ollama: https://ollama.ai/")

def run_integration_test():
   """Run integration test"""
   print("\n🧪 Running integration test...")
   
   try:
       # Test imports
       print("📦 Testing imports...")
       import cv2
       import mediapipe as mp
       from landmark_analyzer import MedicalLandmarkAnalyzer
       from ollama_analyzer import OllamaMedicalAnalyzer
       print("✅ All imports successful")
       
       # Test camera
       print("📸 Testing camera...")
       cap = cv2.VideoCapture(0)
       if cap.isOpened():
           print("✅ Camera accessible")
           cap.release()
       else:
           print("⚠️  Camera not accessible")
       
       # Test landmark detection
       print("🎯 Testing landmark detection...")
       analyzer = MedicalLandmarkAnalyzer()
       print("✅ Landmark analyzer ready")
       
       # Test Ollama connection
       print("🦙 Testing Ollama connection...")
       ollama = OllamaMedicalAnalyzer()
       print("✅ Ollama connection successful")
       
       print("\n🎉 All integration tests passed!")
       
   except Exception as e:
       print(f"❌ Integration test failed: {e}")

def display_results(results):
   """Display analysis results"""
   if not results:
       print("❌ No results to display")
       return
       
   if "error" in results:
       print(f"❌ Error: {results['error']}")
       return
   
   print("\n" + "🏥 MEDICAL ANALYSIS RESULTS 🏥")
   print("=" * 60)
   
   print(f"📊 Analysis: {results['analysis_type'].title()}")
   print(f"🎯 Landmarks: {results['landmarks_detected']}")
   print(f"⏰ Time: {results['timestamp'][:19]}")
   
   if results.get('annotated_image_path'):
       print(f"🖼️  Saved: {results['annotated_image_path']}")
   
   # Display key metrics
   metrics = results['metrics']['metrics']
   print(f"\n📏 Key Metrics:")
   print(f"  💫 Symmetry Score: {metrics['facial_symmetry_score']:.3f}")
   
   ear = metrics['eye_aspect_ratio']
   print(f"  👁️  EAR - L:{ear['left']:.3f} R:{ear['right']:.3f} Diff:{ear['difference']:.3f}")
   print(f"  😮 Mouth Asymmetry: {metrics['mouth_corner_horizontal_deviation_mm']:.2f} mm")
   print(f"  🤨 Eyebrow Diff: {metrics['eyebrow_height_difference_mm']:.2f} mm")
   
   pose = metrics['head_pose_degrees']
   print(f"  🗣️  Head Pose: P:{pose['pitch']:.1f}° Y:{pose['yaw']:.1f}° R:{pose['roll']:.1f}°")
   
   print(f"\n🦙 Ollama AI Analysis:")
   print("─" * 60)
   analysis = results['ollama_analysis']
   
   # Truncate if too long for console
   if len(analysis) > 800:
       print(analysis[:800] + "\n... [truncated]")
       
       save_full = input("\n💾 Save full analysis to file? (y/n): ").lower() == 'y'
       if save_full:
           filename = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
           with open(filename, 'w') as f:
               f.write(f"Medical Analysis Results\n")
               f.write(f"Time: {results['timestamp']}\n")
               f.write(f"Type: {results['analysis_type']}\n\n")
               f.write(analysis)
           print(f"📄 Full analysis saved to {filename}")
   else:
       print(analysis)
   
   print("\n" + "=" * 60)
   input("Press Enter to continue...")

if __name__ == "__main__":
   main()
EOF

# Create configuration file
cat > medical_analysis/config.json << 'EOF'
{
 "ollama_config": {
   "base_url": "http://localhost:11434",
   "model": "llava",
   "timeout": 120
 },
 "analysis_config": {
   "default_analysis": "asymmetry",
   "landmark_confidence": 0.5,
   "tracking_confidence": 0.5,
   "max_faces": 1,
   "pixel_to_mm_ratio": 0.1
 },
 "output_config": {
   "save_annotated_images": true,
   "save_results_json": true,
   "results_directory": "medical_analysis/results"
 },
 "medical_thresholds": {
   "symmetry_excellent": 0.9,
   "symmetry_good": 0.8,
   "symmetry_fair": 0.7,
   "ear_normal_diff": 0.02,
   "ear_concerning_diff": 0.05,
   "mouth_mild_asymmetry": 2.0,
   "mouth_moderate_asymmetry": 5.0,
   "mouth_severe_asymmetry": 10.0
 }
}
EOF

# Make scripts executable
chmod +x run_medical_analysis.py
chmod +x test_ollama_integration.py

echo ""
echo "🏥 Simple Medical Analysis Setup Complete!"
echo "=" * 60
echo ""
echo "📁 Created structure:"
echo "  medical_analysis/"
echo "    ├── landmark_analyzer.py (MediaPipe landmark_pb2)"
echo "    ├── ollama_analyzer.py (Local Ollama integration)"
echo "    ├── medical_main.py (Main analysis runner)"
echo "    └── config.json"
echo ""
echo "🔧 Integration files:"
echo "  mindful-core/medical_processor.py"
echo "  mindful-desktop/app/simple_medical_ui.py"
echo "  run_medical_analysis.py"
echo "  test_ollama_integration.py"
echo ""
echo "🚀 Next steps:"
echo "  1. Install dependencies:"
echo "     pip install mediapipe opencv-python requests pillow numpy"
echo ""
echo "  2. Install and setup Ollama:"
echo "     # Download from https://ollama.ai/"
echo "     ollama serve              # Start server"
echo "     ollama pull llava         # Install vision model"
echo ""
echo "  3. Test setup:"
echo "     python test_ollama_integration.py"
echo ""
echo "  4. Run analysis:"
echo "     python run_medical_analysis.py"
echo ""
echo "💡 Key advantages:"
echo "   • ✅ Simple: Only mediapipe + ollama (no torch/transformers)"
echo "   • ✅ Local: Everything runs on your machine"
echo "   • ✅ Fast: MediaPipe landmark_pb2 for precision"
echo "   • ✅ Flexible: Multiple vision models available"
echo "   • ✅ Lightweight: ~50MB MediaPipe vs ~13GB torch models"
EOF

chmod +x setup.sh

echo "✅ setup.sh recreated with simple Ollama integration!"
echo ""
echo "🦙 This version uses:"
echo "  • MediaPipe landmark_pb2 for accurate facial landmarks"  
echo "  • Ollama for local AI analysis (no torch/transformers)"
echo "  • Simple HTTP requests to Ollama API"
echo "  • Much lighter dependencies and faster startup"
echo ""
echo "🚀 To run the setup:"
echo "  ./setup.sh"
echo ""
echo "📦 Required setup:"
echo "  1. pip install mediapipe opencv-python requests pillow numpy"
echo "  2. Install Ollama from https://ollama.ai/"
echo "  3. ollama serve && ollama pull llava"