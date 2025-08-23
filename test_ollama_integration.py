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
