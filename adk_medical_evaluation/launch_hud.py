# adk_medical_evaluation/launch_hud.py
"""
Simple launcher for Camera HUD
"""

import os
import sys

def main():
    print("🏥 Medical Pipeline - Camera HUD Launcher")
    print("=" * 50)
    
    # Check for required dependencies
    try:
        import cv2
        import tkinter as tk
        from PIL import Image, ImageTk
        print("✅ All dependencies available")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Install with: pip install opencv-python pillow")
        return
    
    # Check for logo
    logo_path = os.path.join("assets", "logo.png")
    if os.path.exists(logo_path):
        print(f"✅ Logo found: {logo_path}")
    else:
        print(f"⚠️ Logo not found: {logo_path}")
        print("   HUD will work without logo")
    
    # Launch HUD
    print("🚀 Launching Camera HUD...")
    
    try:
        from ui.camera_hud import main as hud_main
        hud_main()
    except ImportError:
        print("❌ Could not import HUD module")
        print("Make sure you're running from the adk_medical_evaluation directory")
    except Exception as e:
        print(f"❌ Error launching HUD: {e}")


if __name__ == "__main__":
    main()