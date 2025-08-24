"""
Updated Medical Analysis Processor using Gemini instead of Ollama
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'medical_analysis'))

import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from landmark_analyzer import MedicalLandmarkAnalyzer, metrics_to_json
from gemini_medical_analyzer import GeminiMedicalAnalyzer

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
    gemini_confidence: float

class GeminiMedicalProcessor:
    """Medical processor using Gemini from Vertex AI"""
    
    def __init__(self, project_id: str, location: str = "us-central1"):
        self.landmark_analyzer = None
        self.gemini_analyzer = None
        self.is_initialized = False
        self.project_id = project_id
        self.location = location
        
    def initialize(self) -> bool:
        """Initialize processors"""
        if self.is_initialized:
            return True
            
        try:
            print("🔧 Initializing Gemini medical processors...")
            self.landmark_analyzer = MedicalLandmarkAnalyzer()
            self.gemini_analyzer = GeminiMedicalAnalyzer(
                project_id=self.project_id,
                location=self.location
            )
            self.is_initialized = True
            print("✅ Gemini medical processors ready")
            return True
        except Exception as e:
            print(f"❌ Medical processor init failed: {e}")
            return False
    
    def quick_analyze(self, frame: np.ndarray) -> Optional[MedicalAnalysisResult]:
        """Quick analysis without Gemini for real-time performance"""
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
            
            # Basic AI assessment without Gemini for speed
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
                recommendations=recommendations,
                gemini_confidence=0.0  # No Gemini used in quick analysis
            )
            
        except Exception as e:
            print(f"❌ Quick analysis failed: {e}")
            return None
    
    def detailed_analyze(self, frame: np.ndarray, analysis_type: str = "asymmetry") -> Optional[MedicalAnalysisResult]:
        """Detailed analysis with full Gemini AI assessment"""
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
            
            # Full Gemini analysis
            try:
                if analysis_type == "asymmetry":
                    ai_assessment = self.gemini_analyzer.analyze_facial_asymmetry(frame, metrics_json)
                elif analysis_type == "iris":
                    iris_coords = {
                        "left": list(metrics.iris_left_center),
                        "right": list(metrics.iris_right_center)
                    }
                    ai_assessment = self.gemini_analyzer.analyze_iris_features(frame, iris_coords)
                else:
                    ai_assessment = self.gemini_analyzer.analyze_general_health(frame, metrics_json)
                
                # Extract confidence from Gemini response (simplified)
                gemini_confidence = self._extract_confidence_from_response(ai_assessment)
                
            except Exception as e:
                ai_assessment = f"Gemini analysis unavailable: {e}"
                gemini_confidence = 0.0
            
            # Calculate severity and recommendations
            severity = self._calculate_simple_severity(metrics)
            recommendations = self._generate_gemini_recommendations(ai_assessment, metrics, severity)
            
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
                recommendations=recommendations,
                gemini_confidence=gemini_confidence
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
    
    def _generate_gemini_recommendations(self, ai_assessment: str, metrics, severity: float) -> list:
        """Generate enhanced recommendations based on Gemini analysis"""
        recs = self._generate_simple_recommendations(metrics, severity)
        
        # Parse Gemini recommendations from the assessment
        if "recommend" in ai_assessment.lower():
            # Extract recommendations from Gemini response
            lines = ai_assessment.split('\n')
            for line in lines:
                if any(word in line.lower() for word in ['recommend', 'suggest', 'advise', 'consider']):
                    if len(line.strip()) > 10 and line.strip() not in recs:
                        recs.append(line.strip())
        
        return recs[:10]  # Limit to top 10 recommendations
    
    def _extract_confidence_from_response(self, response: str) -> float:
        """Extract confidence score from Gemini response (simplified)"""
        # Look for confidence indicators in the response
        confidence_indicators = {
            'high confidence': 0.9,
            'confident': 0.8,
            'likely': 0.7,
            'probable': 0.6,
            'possible': 0.5,
            'uncertain': 0.3,
            'unclear': 0.2
        }
        
        response_lower = response.lower()
        for indicator, score in confidence_indicators.items():
            if indicator in response_lower:
                return score
        
        # Default moderate confidence
        return 0.6