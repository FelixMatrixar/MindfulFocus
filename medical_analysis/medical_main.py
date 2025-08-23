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
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
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
