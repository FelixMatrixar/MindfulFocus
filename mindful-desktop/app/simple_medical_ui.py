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
