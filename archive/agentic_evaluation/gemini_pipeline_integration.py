"""
Integration layer with Gemini-powered evaluation
"""

import sys
import os
import time
import cv2
from datetime import datetime
from typing import Dict, Any

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'medical_analysis'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'mindful-core'))

from landmark_analyzer import MedicalLandmarkAnalyzer, metrics_to_json
from updated_medical_pipeline_evaluator import GeminiMedicalPipelineEvaluatorAgent
from updated_medical_processor import GeminiMedicalProcessor

class GeminiPipelineEvaluationOrchestrator:
    """Orchestrates evaluation using Gemini"""
    
    def __init__(self, project_id: str, location: str = "us-central1"):
        self.project_id = project_id
        self.location = location
        
        self.medical_processor = GeminiMedicalProcessor(project_id, location)
        self.evaluation_agent = GeminiMedicalPipelineEvaluatorAgent(project_id, location)
        
        self.evaluation_data = {
            'frame_results': [],
            'processing_times': [],
            'gemini_results': [],
            'resource_usage': {},
            'project_id': project_id,
            'location': location
        }
    
    async def run_comprehensive_gemini_evaluation(self, duration_seconds: int = 300, 
                                                 use_detailed_analysis: bool = False):
        """Run comprehensive evaluation with Gemini analysis"""
        print(f"🚀 Starting {duration_seconds}s Gemini-powered evaluation...")
        print(f"📍 Project: {self.project_id}, Location: {self.location}")
        print(f"🧠 Detailed Gemini Analysis: {'Enabled' if use_detailed_analysis else 'Quick Mode'}")
        
        start_time = time.time()
        frame_count = 0
        
        # Initialize medical processor
        if not self.medical_processor.initialize():
            print("❌ Failed to initialize Gemini medical processor")
            return None
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open camera")
            return None
        
        try:
            while time.time() - start_time < duration_seconds:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                frame_start_time = time.time()
                
                # Choose analysis type based on mode and frequency
                if use_detailed_analysis and frame_count % 10 == 0:  # Detailed every 10th frame
                    analysis_type = "asymmetry"
                    result = self.medical_processor.detailed_analyze(frame, analysis_type)
                else:
                    # Quick analysis for real-time performance
                    result = self.medical_processor.quick_analyze(frame)
                
                processing_time = time.time() - frame_start_time
                
                # Store results
                if result:
                    frame_result = {
                        'frame_id': frame_count,
                        'processing_time': processing_time,
                        'landmarks_detected': result.landmarks_count,
                        'symmetry_score': result.symmetry_score,
                        'severity_score': result.severity_score,
                        'ear_left': result.ear_difference / 2,  # Approximation
                        'ear_right': result.ear_difference / 2,
                        'mouth_asymmetry_mm': result.mouth_asymmetry_mm,
                        'eyebrow_diff_mm': result.eyebrow_diff_mm,
                        'analysis_type': result.analysis_type,
                        'gemini_confidence': result.gemini_confidence,
                        'ai_assessment_length': len(result.ai_assessment),
                        'recommendations_count': len(result.recommendations),
                        'timestamp': result.timestamp.isoformat()
                    }
                    
                    self.evaluation_data['frame_results'].append(frame_result)
                    self.evaluation_data['processing_times'].append(processing_time)
                    
                    # Store Gemini-specific results
                    if result.gemini_confidence > 0:
                        gemini_result = {
                            'frame_id': frame_count,
                            'confidence': result.gemini_confidence,
                            'analysis_type': result.analysis_type,
                            'assessment_preview': result.ai_assessment[:200],
                            'recommendations_count': len(result.recommendations),
                            'processing_time': processing_time,
                            'timestamp': result.timestamp.isoformat()
                        }
                        self.evaluation_data['gemini_results'].append(gemini_result)
                
                # Progress reporting
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    avg_confidence = np.mean([r.get('gemini_confidence', 0) for r in self.evaluation_data['frame_results'][-30:]])
                    print(f"📊 Progress: {frame_count} frames, {fps:.1f} FPS, Avg Gemini Confidence: {avg_confidence:.2f}")
        
        finally:
            cap.release()
        
        print(f"✅ Data collection complete: {len(self.evaluation_data['frame_results'])} frames analyzed")
        
        # Add comprehensive metadata
        total_time = time.time() - start_time
        self.evaluation_data.update({
            'version': '2.0-gemini',
            'model_used': 'gemini-1.5-pro',
            'duration_seconds': total_time,
            'total_frames': frame_count,
            'avg_fps': frame_count / total_time,
            'avg_processing_time': np.mean(self.evaluation_data['processing_times']),
            'detailed_analysis_frames': len([r for r in self.evaluation_data['frame_results'] if r.get('gemini_confidence', 0) > 0]),
            'avg_gemini_confidence': np.mean([r.get('gemini_confidence', 0) for r in self.evaluation_data['frame_results']]),
            'evaluation_timestamp': datetime.now().isoformat(),
            'use_detailed_analysis': use_detailed_analysis
        })
        
        # Run Gemini-powered evaluation
        print("🧠 Starting Gemini evaluation analysis...")
        evaluation_report = await self.evaluation_agent.evaluate_pipeline(self.evaluation_data)
        
        return evaluation_report

    async def run_quick_gemini_evaluation(self, duration_seconds: int = 60):
        """Run quick evaluation with minimal Gemini usage"""
        return await self.run_comprehensive_gemini_evaluation(
            duration_seconds=duration_seconds,
            use_detailed_analysis=False
        )

    async def run_deep_gemini_evaluation(self, duration_seconds: int = 300):
        """Run deep evaluation with extensive Gemini analysis"""
        return await self.run_comprehensive_gemini_evaluation(
            duration_seconds=duration_seconds,
            use_detailed_analysis=True
        )