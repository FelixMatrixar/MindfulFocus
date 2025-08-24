# launcher.py
"""
Medical Pipeline Evaluator Launcher
Project: mindfulfocus-470008
"""

import os
import sys
import asyncio
from datetime import datetime

# Add ADK evaluation path
sys.path.append("adk_medical_evaluation")

def display_banner():
    """Display welcome banner"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║              MEDICAL PIPELINE EVALUATOR                     ║
║                   with Agent Development Kit                ║
║                                                              ║
║  Project: mindfulfocus-470008                               ║
║  Model: Gemini 2.5 Pro                                     ║
║  Storage: Local SQLite + Files                             ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")

def check_requirements():
    """Check system requirements"""
    print("🔍 Checking system requirements...")
    
    requirements = {
        "opencv-python": "cv2",
        "mediapipe": "mediapipe", 
        "numpy": "numpy",
        "pandas": "pandas",
        "google-adk": "google.adk",
        "sqlite3": "sqlite3"
    }
    
    missing = []
    
    for package, import_name in requirements.items():
        try:
            __import__(import_name)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
            missing.append(package)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Run: pip install " + " ".join(missing))
        return False
    
    print("✅ All requirements satisfied")
    return True

def check_directories():
    """Check and create necessary directories"""
    print("📁 Checking directories...")
    
    directories = [
        "evaluation_data",
        "evaluation_data/images",
        "evaluation_data/reports", 
        "evaluation_data/metrics",
        "evaluation_data/sessions",
        "evaluation_data/logs",
        "evaluation_data/exports",
        "local_database"
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"  📁 Created: {directory}")
        else:
            print(f"  ✅ Exists: {directory}")
    
    return True

def check_camera():
    """Check camera availability"""
    print("📹 Checking camera...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print("  ✅ Camera working")
                cap.release()
                return True
        cap.release()
        print("  ⚠️ Camera not accessible")
        return False
    except Exception as e:
        print(f"  ❌ Camera error: {e}")
        return False

def setup_api_key():
    """Setup Gemini API key"""
    if os.getenv("GOOGLE_API_KEY"):
        print("✅ Gemini API key already set")
        return True
    
    print("🔑 Gemini API key not found")
    print("Get your key from: https://aistudio.google.com/app/apikey")
    
    api_key = input("Enter your Gemini API key: ").strip()
    
    if not api_key:
        print("❌ API key required for operation")
        return False
    
    os.environ["GOOGLE_API_KEY"] = api_key
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"
    
    print("✅ API key configured")
    return True

async def launch_evaluator():
    """Launch the main evaluator"""
    try:
        from main_evaluator import main
        await main()
    except ImportError as e:
        print(f"❌ Failed to import evaluator: {e}")
        return False
    except Exception as e:
        print(f"❌ Evaluator error: {e}")
        return False
    
    return True

def show_quick_start_guide():
    """Show quick start guide"""
    print("""
🚀 QUICK START GUIDE
═══════════════════════════════════════════════════════════════

1. 📹 Camera Evaluation (Recommended first run):
   - Select option 1 for 5-minute evaluation
   - Ensure good lighting and face visibility
   - System will analyze facial landmarks in real-time

2. 📊 Results Review:
   - Check evaluation_data/reports/ for detailed reports
   - View database statistics for historical data
   - Export options available in multiple formats

3. 🔧 Performance Testing:
   - Option 4 for quick 1-minute performance test
   - Monitor processing speed and accuracy
   - Identify system bottlenecks

4. 🏥 Medical Assessment:
   - Automatic generation of clinical assessments
   - Facial asymmetry detection and scoring
   - Severity classifications and recommendations

5. 📁 File Management:
   - Automatic organization of images and data
   - Cleanup tools for storage management
   - Export capabilities for external analysis

⚠️  IMPORTANT MEDICAL DISCLAIMER:
This system is for research and development purposes.
Results should be reviewed by qualified medical professionals
before any clinical decisions are made.

Press Enter to continue...
""")
    input()

async def main():
    """Main launcher function"""
    display_banner()
    
    # System checks
    if not check_requirements():
        print("\n❌ Please install missing requirements and try again")
        return
    
    if not check_directories():
        print("\n❌ Failed to create directories")
        return
    
    camera_ok = check_camera()
    if not camera_ok:
        print("⚠️ Camera issues detected - image directory evaluation will still work")
    
    if not setup_api_key():
        print("\n❌ API key required for operation")
        return
    
    print("\n✅ System ready!")
    
    # Show options
    while True:
        print("\n🎯 Launch Options:")
        print("1. 🚀 Start Medical Pipeline Evaluator")
        print("2. 📖 Quick Start Guide") 
        print("3. 🔍 System Information")
        print("4. ❌ Exit")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == "1":
            print("\n🚀 Starting Medical Pipeline Evaluator...")
            await launch_evaluator()
            break
            
        elif choice == "2":
            show_quick_start_guide()
            
        elif choice == "3":
            display_system_info(camera_ok)
            
        elif choice == "4":
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please select 1-4.")

def display_system_info(camera_ok: bool):
    """Display system information"""
    print(f"""
🖥️  SYSTEM INFORMATION
═══════════════════════════════════════════════════════════════

📊 Project Configuration:
   • Project ID: mindfulfocus-470008
   • Model: Gemini 2.5 Pro (via API)
   • Storage: Local SQLite + File System
   • Framework: Google Agent Development Kit (ADK)

🔧 System Status:
   • Python Version: {sys.version.split()[0]}
   • Camera Available: {'✅ Yes' if camera_ok else '❌ No'}
   • API Key Configured: {'✅ Yes' if os.getenv('GOOGLE_API_KEY') else '❌ No'}
   • Storage Ready: ✅ Yes

📁 Data Directories:
   • Images: evaluation_data/images/
   • Reports: evaluation_data/reports/
   • Database: local_database/medical_evaluation.db
   • Exports: evaluation_data/exports/

🛠️  Available Tools:
   • Real-time camera analysis
   • Batch image processing
   • Performance metrics calculation
   • Medical assessment generation
   • Clinical report generation
   • Data export in multiple formats

⚡ Performance Notes:
   • Recommended: 8GB+ RAM for optimal performance
   • Camera resolution: 1280x720 for best balance
   • Processing: ~10 FPS for real-time analysis
   • Storage: ~1MB per analyzed frame with metadata

Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
═══════════════════════════════════════════════════════════════
""")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Launcher shutdown complete")
    except Exception as e:
        print(f"\n❌ Launcher error: {e}")