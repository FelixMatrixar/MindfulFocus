"""
Medical Analysis Integration for MindfulFocus
Adds medical face landmark analysis capabilities to the existing mindfulness app
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'medical_analysis'))

from medical_main import MedicalAnalysisIntegration
import threading
import time

class MindfulMedicalIntegration:
   def __init__(self):
       self.medical_analyzer = None
       self.medical_mode = False
       
   def initialize_medical_analysis(self):
       """Initialize medical analysis in a separate thread to avoid blocking UI"""
       if self.medical_analyzer is None:
           print("Initializing medical analysis capabilities...")
           try:
               self.medical_analyzer = MedicalAnalysisIntegration()
               print("Medical analysis ready!")
               return True
           except Exception as e:
               print(f"Error initializing medical analysis: {e}")
               return False
       return True
   
   def toggle_medical_mode(self):
       """Toggle between mindfulness and medical analysis modes"""
       if self.medical_analyzer is None:
           success = self.initialize_medical_analysis()
           if not success:
               return False
       
       self.medical_mode = not self.medical_mode
       mode = "Medical Analysis" if self.medical_mode else "Mindfulness"
       print(f"Switched to {mode} mode")
       return self.medical_mode
   
   def analyze_current_frame(self, frame, analysis_type="asymmetry"):
       """Analyze the current frame for medical indicators"""
       if self.medical_analyzer is None or not self.medical_mode:
           return None
       
       try:
           # Save frame temporarily
           temp_path = "temp_frame.jpg"
           import cv2
           cv2.imwrite(temp_path, frame)
           
           # Analyze
           results = self.medical_analyzer.analyze_image(temp_path, analysis_type)
           
           # Clean up
           os.remove(temp_path)
           
           return results
       except Exception as e:
           print(f"Error in medical analysis: {e}")
           return None
