# adk_medical_evaluation/ui/camera_hud.py
"""
Simple HUD Camera Preview UI for Medical Pipeline Evaluator
Project: mindfulfocus-470008
"""

import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import threading
import time
import os
import sys
from datetime import datetime
from typing import Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from main_evaluator import LocalMedicalPipelineEvaluator
    from config.agent_config import PROJECT_ID, MODEL_NAME
except ImportError as e:
    print(f"Warning: Could not import evaluator components: {e}")

class CameraHUD:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.setup_window()
        
        # Camera variables
        self.cap: Optional[cv2.VideoCapture] = None
        self.camera_thread: Optional[threading.Thread] = None
        self.camera_running = False
        self.current_frame = None
        
        # Evaluation system (optional)
        self.evaluator: Optional[LocalMedicalPipelineEvaluator] = None
        self.evaluation_active = False
        
        # UI Components
        self.setup_ui()
        self.setup_bindings()
        
        # Start camera automatically
        self.start_camera()
        
    def setup_window(self):
        """Setup main window properties"""
        self.root.title("Medical Pipeline - Camera HUD")
        self.root.geometry("1200x800")
        self.root.configure(bg='#0a0a0a')
        self.root.resizable(True, True)
        
        # Make window stay on top (HUD style)
        # self.root.wm_attributes("-topmost", True)
        
        # Icon setup (if logo exists)
        logo_path = os.path.join("assets", "logo.png")
        if os.path.exists(logo_path):
            try:
                icon = Image.open(logo_path)
                icon = icon.resize((32, 32), Image.Resampling.LANCZOS)
                self.icon_photo = ImageTk.PhotoImage(icon)
                self.root.iconphoto(False, self.icon_photo)
            except Exception as e:
                print(f"Could not load icon: {e}")
    
    def setup_ui(self):
        """Setup the HUD interface"""
        # Main container
        main_frame = tk.Frame(self.root, bg='#0a0a0a')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Header with logo and title
        self.setup_header(main_frame)
        
        # Camera preview area (main focus)
        self.setup_camera_area(main_frame)
        
        # Bottom status bar
        self.setup_status_bar(main_frame)
        
        # Side control panel (minimal)
        self.setup_controls(main_frame)
    
    def setup_header(self, parent):
        """Setup header with logo and title"""
        header_frame = tk.Frame(parent, bg='#0a0a0a', height=60)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        header_frame.pack_propagate(False)
        
        # Logo
        logo_path = os.path.join("assets", "logo.png")
        if os.path.exists(logo_path):
            try:
                logo_img = Image.open(logo_path)
                logo_img = logo_img.resize((50, 50), Image.Resampling.LANCZOS)
                self.logo_photo = ImageTk.PhotoImage(logo_img)
                
                logo_label = tk.Label(
                    header_frame, 
                    image=self.logo_photo, 
                    bg='#0a0a0a'
                )
                logo_label.pack(side=tk.LEFT, padx=(10, 15))
            except Exception as e:
                print(f"Could not load logo: {e}")
        
        # Title
        title_frame = tk.Frame(header_frame, bg='#0a0a0a')
        title_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        title_label = tk.Label(
            title_frame,
            text="Medical Pipeline Evaluator",
            font=('Arial', 18, 'bold'),
            fg='#00ff41',
            bg='#0a0a0a'
        )
        title_label.pack(anchor=tk.W)
        
        subtitle_label = tk.Label(
            title_frame,
            text=f"Project: {PROJECT_ID} • Model: {MODEL_NAME}",
            font=('Arial', 10),
            fg='#888888',
            bg='#0a0a0a'
        )
        subtitle_label.pack(anchor=tk.W)
        
        # Status indicator
        self.status_indicator = tk.Label(
            header_frame,
            text="●",
            font=('Arial', 20),
            fg='#ff4444',  # Red when not connected
            bg='#0a0a0a'
        )
        self.status_indicator.pack(side=tk.RIGHT, padx=10)
        
        status_text = tk.Label(
            header_frame,
            text="CAMERA",
            font=('Arial', 10, 'bold'),
            fg='#888888',
            bg='#0a0a0a'
        )
        status_text.pack(side=tk.RIGHT, padx=(10, 5))
    
    def setup_camera_area(self, parent):
        """Setup main camera preview area"""
        # Camera frame with border effect
        camera_container = tk.Frame(parent, bg='#001100', relief=tk.RAISED, bd=2)
        camera_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Camera display label
        self.camera_label = tk.Label(
            camera_container,
            text="INITIALIZING CAMERA...",
            font=('Arial', 16),
            fg='#00ff41',
            bg='#000000',
            width=80,
            height=25
        )
        self.camera_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Overlay info (bottom-left of camera)
        self.info_frame = tk.Frame(camera_container, bg='#000000')
        self.info_frame.place(x=10, rely=1.0, anchor=tk.SW)
        
        self.fps_label = tk.Label(
            self.info_frame,
            text="FPS: --",
            font=('Courier', 10, 'bold'),
            fg='#00ff41',
            bg='#000000'
        )
        self.fps_label.pack(anchor=tk.W)
        
        self.resolution_label = tk.Label(
            self.info_frame,
            text="Resolution: --",
            font=('Courier', 10, 'bold'),
            fg='#00ff41',
            bg='#000000'
        )
        self.resolution_label.pack(anchor=tk.W)
        
        self.timestamp_label = tk.Label(
            self.info_frame,
            text="Time: --",
            font=('Courier', 10, 'bold'),
            fg='#00ff41',
            bg='#000000'
        )
        self.timestamp_label.pack(anchor=tk.W)
    
    def setup_status_bar(self, parent):
        """Setup bottom status bar"""
        status_frame = tk.Frame(parent, bg='#0a0a0a', height=30)
        status_frame.pack(fill=tk.X, pady=(5, 0))
        status_frame.pack_propagate(False)
        
        self.status_text = tk.Label(
            status_frame,
            text="Ready",
            font=('Courier', 10),
            fg='#888888',
            bg='#0a0a0a'
        )
        self.status_text.pack(side=tk.LEFT, padx=5)
        
        # Frame counter
        self.frame_counter = tk.Label(
            status_frame,
            text="Frames: 0",
            font=('Courier', 10),
            fg='#888888',
            bg='#0a0a0a'
        )
        self.frame_counter.pack(side=tk.RIGHT, padx=5)
    
    def setup_controls(self, parent):
        """Setup minimal control panel"""
        # Control panel (right side, minimal)
        control_frame = tk.Frame(parent, bg='#111111', width=200, relief=tk.RAISED, bd=1)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        control_frame.pack_propagate(False)
        
        # Controls title
        tk.Label(
            control_frame,
            text="CONTROLS",
            font=('Arial', 12, 'bold'),
            fg='#00ff41',
            bg='#111111'
        ).pack(pady=(10, 5))
        
        # Camera controls
        self.start_btn = tk.Button(
            control_frame,
            text="START CAMERA",
            font=('Arial', 10, 'bold'),
            bg='#004400',
            fg='#00ff41',
            activebackground='#006600',
            activeforeground='#00ff41',
            relief=tk.FLAT,
            width=15,
            command=self.toggle_camera
        )
        self.start_btn.pack(pady=5)
        
        # Capture button
        capture_btn = tk.Button(
            control_frame,
            text="CAPTURE",
            font=('Arial', 10, 'bold'),
            bg='#444400',
            fg='#ffff00',
            activebackground='#666600',
            activeforeground='#ffff00',
            relief=tk.FLAT,
            width=15,
            command=self.capture_frame
        )
        capture_btn.pack(pady=5)
        
        # Separator
        separator = tk.Frame(control_frame, bg='#333333', height=2)
        separator.pack(fill=tk.X, padx=10, pady=10)
        
        # Evaluation button
        self.eval_btn = tk.Button(
            control_frame,
            text="START EVAL",
            font=('Arial', 10, 'bold'),
            bg='#440000',
            fg='#ff4444',
            activebackground='#660000',
            activeforeground='#ff4444',
            relief=tk.FLAT,
            width=15,
            command=self.toggle_evaluation
        )
        self.eval_btn.pack(pady=5)
        
        # Info display
        info_text = tk.Text(
            control_frame,
            height=10,
            width=25,
            bg='#000000',
            fg='#00ff41',
            font=('Courier', 8),
            relief=tk.FLAT,
            state=tk.DISABLED
        )
        info_text.pack(pady=(20, 10), padx=5, fill=tk.BOTH, expand=True)
        self.info_text = info_text
        
        # Add initial info
        self.update_info_display("System initialized\nCamera starting...\n")
    
    def setup_bindings(self):
        """Setup keyboard bindings"""
        self.root.bind('<Escape>', lambda e: self.root.quit())
        self.root.bind('<space>', lambda e: self.capture_frame())
        self.root.bind('<F1>', lambda e: self.toggle_camera())
        self.root.bind('<F2>', lambda e: self.toggle_evaluation())
    
    def start_camera(self):
        """Start camera capture"""
        if self.camera_running:
            return
            
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.update_status("ERROR: Could not open camera")
                return
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.camera_running = True
            self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
            self.camera_thread.start()
            
            self.status_indicator.config(fg='#00ff41')  # Green when connected
            self.start_btn.config(text="STOP CAMERA", bg='#440000', fg='#ff4444')
            self.update_status("Camera started")
            self.update_info_display("Camera connected\nStreaming...\n")
            
        except Exception as e:
            self.update_status(f"Camera error: {str(e)}")
            self.update_info_display(f"Camera error:\n{str(e)}\n")
    
    def stop_camera(self):
        """Stop camera capture"""
        self.camera_running = False
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.status_indicator.config(fg='#ff4444')  # Red when disconnected
        self.start_btn.config(text="START CAMERA", bg='#004400', fg='#00ff41')
        self.camera_label.config(image='', text="CAMERA STOPPED")
        self.update_status("Camera stopped")
        self.update_info_display("Camera stopped\n")
    
    def toggle_camera(self):
        """Toggle camera on/off"""
        if self.camera_running:
            self.stop_camera()
        else:
            self.start_camera()
    
    def camera_loop(self):
        """Main camera capture loop"""
        frame_count = 0
        fps_counter = 0
        fps_start = time.time()
        
        while self.camera_running and self.cap:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                self.current_frame = frame
                frame_count += 1
                fps_counter += 1
                
                # Convert frame for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize for display
                height, width = frame_rgb.shape[:2]
                display_width = self.camera_label.winfo_width()
                display_height = self.camera_label.winfo_height()
                
                if display_width > 1 and display_height > 1:
                    # Maintain aspect ratio
                    aspect_ratio = width / height
                    if display_width / display_height > aspect_ratio:
                        new_height = display_height
                        new_width = int(display_height * aspect_ratio)
                    else:
                        new_width = display_width
                        new_height = int(display_width / aspect_ratio)
                    
                    frame_resized = cv2.resize(frame_rgb, (new_width, new_height))
                    
                    # Convert to PhotoImage
                    img_pil = Image.fromarray(frame_resized)
                    img_tk = ImageTk.PhotoImage(img_pil)
                    
                    # Update display
                    self.camera_label.config(image=img_tk, text='')
                    self.camera_label.image = img_tk  # Keep reference
                
                # Update info overlay
                current_time = time.time()
                if current_time - fps_start >= 1.0:  # Update every second
                    fps = fps_counter / (current_time - fps_start)
                    
                    self.root.after(0, lambda: self.fps_label.config(text=f"FPS: {fps:.1f}"))
                    self.root.after(0, lambda: self.resolution_label.config(text=f"Resolution: {width}x{height}"))
                    self.root.after(0, lambda: self.timestamp_label.config(text=f"Time: {datetime.now().strftime('%H:%M:%S')}"))
                    self.root.after(0, lambda: self.frame_counter.config(text=f"Frames: {frame_count}"))
                    
                    fps_counter = 0
                    fps_start = current_time
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                print(f"Camera loop error: {e}")
                break
    
    def capture_frame(self):
        """Capture current frame to file"""
        if not self.current_frame is None:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"capture_{timestamp}.jpg"
                filepath = os.path.join("evaluation_data", "captures", filename)
                
                # Create directory if needed
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                
                # Save frame
                cv2.imwrite(filepath, self.current_frame)
                self.update_status(f"Frame captured: {filename}")
                self.update_info_display(f"Captured: {filename}\n")
                
            except Exception as e:
                self.update_status(f"Capture error: {str(e)}")
                self.update_info_display(f"Capture error: {str(e)}\n")
        else:
            self.update_status("No frame to capture")
    
    def toggle_evaluation(self):
        """Toggle evaluation system (placeholder)"""
        if not self.evaluation_active:
            self.evaluation_active = True
            self.eval_btn.config(text="STOP EVAL", bg='#004400', fg='#00ff41')
            self.update_status("Evaluation started (simulated)")
            self.update_info_display("Evaluation started\n(Simulated mode)\n")
        else:
            self.evaluation_active = False
            self.eval_btn.config(text="START EVAL", bg='#440000', fg='#ff4444')
            self.update_status("Evaluation stopped")
            self.update_info_display("Evaluation stopped\n")
    
    def update_status(self, message: str):
        """Update status bar"""
        self.status_text.config(text=message)
    
    def update_info_display(self, message: str):
        """Update info display"""
        self.info_text.config(state=tk.NORMAL)
        self.info_text.insert(tk.END, message)
        self.info_text.see(tk.END)
        self.info_text.config(state=tk.DISABLED)
    
    def cleanup(self):
        """Cleanup resources"""
        self.camera_running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


def main():
    """Main entry point"""
    # Create main window
    root = tk.Tk()
    
    # Create HUD interface
    hud = CameraHUD(root)
    
    # Setup cleanup on window close
    def on_closing():
        hud.cleanup()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    print("🖥️ Medical Pipeline Camera HUD Started")
    print("📹 Camera should start automatically")
    print("⌨️ Keyboard shortcuts:")
    print("   ESC - Exit")
    print("   SPACE - Capture frame")
    print("   F1 - Toggle camera")
    print("   F2 - Toggle evaluation")
    
    # Start the GUI
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\n👋 HUD shutdown")
    finally:
        hud.cleanup()


if __name__ == "__main__":
    main()