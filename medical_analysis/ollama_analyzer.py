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
