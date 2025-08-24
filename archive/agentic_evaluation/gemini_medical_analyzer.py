"""
Medical Analysis using Gemini from Vertex AI
"""

import json
import base64
import io
from PIL import Image
import numpy as np
from typing import Dict, Any, Optional
import vertexai
from vertexai.generative_models import GenerativeModel, Part, SafetySetting

class GeminiMedicalAnalyzer:
    def __init__(self, project_id: str, location: str = "us-central1"):
        self.project_id = project_id
        self.location = location
        
        # Initialize Vertex AI
        vertexai.init(project=project_id, location=location)
        
        # Initialize Gemini model
        self.model = GenerativeModel("gemini-1.5-pro")
        
        # Safety settings for medical content
        self.safety_settings = [
            SafetySetting(
                category="HARM_CATEGORY_MEDICAL",
                threshold="BLOCK_NONE"
            ),
            SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT", 
                threshold="BLOCK_NONE"
            )
        ]
        
        print("✅ Gemini Medical Analyzer initialized")
    
    def _image_to_base64(self, image: np.ndarray) -> str:
        """Convert numpy image to base64 for Gemini"""
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Convert BGR to RGB
            image_rgb = image[:, :, ::-1]
        else:
            image_rgb = image
        
        pil_image = Image.fromarray(image_rgb.astype(np.uint8))
        
        # Resize if too large
        if max(pil_image.size) > 1024:
            pil_image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=85)
        image_bytes = buffer.getvalue()
        
        return base64.b64encode(image_bytes).decode('utf-8')
    
    def analyze_facial_asymmetry(self, image: np.ndarray, metrics_json: str) -> str:
        """Analyze facial asymmetry using Gemini vision model"""
        
        # Convert image to base64
        image_b64 = self._image_to_base64(image)
        image_part = Part.from_data(
            data=base64.b64decode(image_b64),
            mime_type="image/jpeg"
        )
        
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
            response = self.model.generate_content(
                [prompt, image_part],
                safety_settings=self.safety_settings
            )
            return response.text
        except Exception as e:
            return f"Error during Gemini analysis: {str(e)}"
    
    def analyze_iris_features(self, image: np.ndarray, iris_coords: Dict[str, Any]) -> str:
        """Analyze iris features for medical indicators using Gemini"""
        
        image_b64 = self._image_to_base64(image)
        image_part = Part.from_data(
            data=base64.b64decode(image_b64),
            mime_type="image/jpeg"
        )
        
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
   - Wilson's disease indicators
   - Horner's syndrome signs
5. **Overall Health Assessment**: Provide general iris health evaluation
6. **Clinical Recommendations**: Suggest if further examination is needed

Be specific about what you observe in each eye and note any asymmetries or concerning features. Focus on medically relevant observations."""

        try:
            response = self.model.generate_content(
                [prompt, image_part],
                safety_settings=self.safety_settings
            )
            return response.text
        except Exception as e:
            return f"Error during iris analysis: {str(e)}"
    
    def analyze_general_health(self, image: np.ndarray, metrics_json: str) -> str:
        """General health assessment from facial features using Gemini"""
        
        image_b64 = self._image_to_base64(image)
        image_part = Part.from_data(
            data=base64.b64decode(image_b64),
            mime_type="image/jpeg"
        )
        
        prompt = f"""You are a medical AI assistant trained in facial diagnosis and physiognomy for health assessment. Analyze this face image for general health indicators.

Landmark metrics data:
{metrics_json}

Please assess:

1. **Skin Health**: Analyze color, texture, blemishes, or discoloration that might indicate:
   - Circulatory issues
   - Nutritional deficiencies
   - Hormonal imbalances
   - Liver or kidney function

2. **Facial Signs**: Look for signs that might indicate:
   - Fatigue or chronic stress
   - Sleep disorders
   - Thyroid conditions
   - Cardiovascular health
   - Respiratory issues

3. **Symmetry and Neurological**: Overall facial balance and any signs of:
   - Neurological weakness
   - Stroke indicators
   - Bell's palsy
   - TMJ disorders

4. **Eye Health**: General appearance including:
   - Scleral color and clarity
   - Periorbital changes
   - Ptosis or drooping
   - Eye movement coordination

5. **Overall Vitality Assessment**: General appearance of health and energy level

6. **Wellness Recommendations**: Provide gentle lifestyle and health recommendations

Important: Emphasize that this analysis is for informational purposes only and is not a substitute for professional medical examination. Recommend consultation with healthcare providers for any concerning findings."""

        try:
            response = self.model.generate_content(
                [prompt, image_part],
                safety_settings=self.safety_settings
            )
            return response.text
        except Exception as e:
            return f"Error during general health analysis: {str(e)}"