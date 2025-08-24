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
