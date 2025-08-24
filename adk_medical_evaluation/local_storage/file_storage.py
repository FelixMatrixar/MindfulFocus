# adk_medical_evaluation/local_storage/file_storage.py
"""
Local file storage utilities for medical evaluation
Project: mindfulfocus-470008
"""

import os
import json
import shutil
from datetime import datetime
from typing import Dict, Any, List, Optional
import cv2
import numpy as np

class LocalFileStorage:
    """Local file storage manager for medical evaluation data"""
    
    def __init__(self, base_dir: str = "evaluation_data"):
        self.base_dir = base_dir
        self.images_dir = os.path.join(base_dir, "images")
        self.reports_dir = os.path.join(base_dir, "reports")
        self.metrics_dir = os.path.join(base_dir, "metrics")
        self.sessions_dir = os.path.join(base_dir, "sessions")
        self.logs_dir = os.path.join(base_dir, "logs")
        
        # Create directories
        self._create_directories()
        
        print(f"✅ LocalFileStorage initialized: {base_dir}")
    
    def _create_directories(self):
        """Create all necessary directories"""
        directories = [
            self.base_dir,
            self.images_dir,
            self.reports_dir, 
            self.metrics_dir,
            self.sessions_dir,
            self.logs_dir
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def save_frame(self, frame: np.ndarray, frame_id: int, session_id: str) -> str:
        """Save camera frame to local storage"""
        try:
            # Create session-specific directory
            session_images_dir = os.path.join(self.images_dir, session_id)
            os.makedirs(session_images_dir, exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"frame_{frame_id:06d}_{timestamp}.jpg"
            filepath = os.path.join(session_images_dir, filename)
            
            # Save image
            success = cv2.imwrite(filepath, frame)
            
            if success:
                print(f"📸 Frame saved: {filename}")
                return filepath
            else:
                raise Exception("Failed to save image with cv2.imwrite")
                
        except Exception as e:
            print(f"❌ Error saving frame {frame_id}: {e}")
            raise
    
    def save_metrics(self, metrics_data: Dict[str, Any], session_id: str, 
                    frame_id: int = None) -> str:
        """Save metrics data to JSON file"""
        try:
            # Create session-specific directory
            session_metrics_dir = os.path.join(self.metrics_dir, session_id)
            os.makedirs(session_metrics_dir, exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if frame_id is not None:
                filename = f"metrics_frame_{frame_id:06d}_{timestamp}.json"
            else:
                filename = f"metrics_summary_{timestamp}.json"
                
            filepath = os.path.join(session_metrics_dir, filename)
            
            # Add metadata
            metrics_with_meta = {
                "session_id": session_id,
                "frame_id": frame_id,
                "timestamp": datetime.now().isoformat(),
                "project_id": "mindfulfocus-470008",
                "model": "gemini-1.5-pro",
                "metrics": metrics_data
            }
            
            # Save JSON
            with open(filepath, 'w') as f:
                json.dump(metrics_with_meta, f, indent=2, default=str)
            
            print(f"📊 Metrics saved: {filename}")
            return filepath
            
        except Exception as e:
            print(f"❌ Error saving metrics: {e}")
            raise
    
    def save_report(self, report_data: Dict[str, Any], session_id: str) -> str:
        """Save evaluation report to JSON file"""
        try:
            # Create session-specific directory
            session_reports_dir = os.path.join(self.reports_dir, session_id)
            os.makedirs(session_reports_dir, exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_report_{timestamp}.json"
            filepath = os.path.join(session_reports_dir, filename)
            
            # Add metadata
            report_with_meta = {
                "report_id": f"medical_eval_{session_id}_{timestamp}",
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "project_id": "mindfulfocus-470008",
                "model": "gemini-1.5-pro",
                "report": report_data
            }
            
            # Save JSON
            with open(filepath, 'w') as f:
                json.dump(report_with_meta, f, indent=2, default=str)
            
            print(f"📋 Report saved: {filename}")
            return filepath
            
        except Exception as e:
            print(f"❌ Error saving report: {e}")
            raise
    
    def save_session_data(self, session_data: Dict[str, Any], session_id: str) -> str:
        """Save session data to file"""
        try:
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"session_{session_id}_{timestamp}.json"
            filepath = os.path.join(self.sessions_dir, filename)
            
            # Add metadata
            session_with_meta = {
                "session_id": session_id,
                "saved_at": datetime.now().isoformat(),
                "project_id": "mindfulfocus-470008",
                "session_data": session_data
            }
            
            # Save JSON
            with open(filepath, 'w') as f:
                json.dump(session_with_meta, f, indent=2, default=str)
            
            print(f"💾 Session data saved: {filename}")
            return filepath
            
        except Exception as e:
            print(f"❌ Error saving session data: {e}")
            raise
    
    def load_session_data(self, filepath: str) -> Optional[Dict[str, Any]]:
        """Load session data from file"""
        try:
            if not os.path.exists(filepath):
                return None
                
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            return data.get('session_data', data)
            
        except Exception as e:
            print(f"❌ Error loading session data: {e}")
            return None
    
    def get_session_files(self, session_id: str) -> Dict[str, List[str]]:
        """Get all files for a specific session"""
        session_files = {
            "images": [],
            "metrics": [],
            "reports": [],
            "sessions": []
        }
        
        try:
            # Images
            session_images_dir = os.path.join(self.images_dir, session_id)
            if os.path.exists(session_images_dir):
                session_files["images"] = [
                    os.path.join(session_images_dir, f) 
                    for f in os.listdir(session_images_dir) 
                    if f.endswith(('.jpg', '.png', '.jpeg'))
                ]
            
            # Metrics
            session_metrics_dir = os.path.join(self.metrics_dir, session_id)
            if os.path.exists(session_metrics_dir):
                session_files["metrics"] = [
                    os.path.join(session_metrics_dir, f)
                    for f in os.listdir(session_metrics_dir)
                    if f.endswith('.json')
                ]
            
            # Reports
            session_reports_dir = os.path.join(self.reports_dir, session_id)
            if os.path.exists(session_reports_dir):
                session_files["reports"] = [
                    os.path.join(session_reports_dir, f)
                    for f in os.listdir(session_reports_dir)
                    if f.endswith('.json')
                ]
            
            # Session files
            session_files["sessions"] = [
                os.path.join(self.sessions_dir, f)
                for f in os.listdir(self.sessions_dir)
                if f.startswith(f"session_{session_id}") and f.endswith('.json')
            ]
            
            return session_files
            
        except Exception as e:
            print(f"❌ Error getting session files: {e}")
            return session_files
    
    def cleanup_session(self, session_id: str, keep_reports: bool = True) -> bool:
        """Clean up session files (optionally keep reports)"""
        try:
            deleted_count = 0
            
            # Delete images
            session_images_dir = os.path.join(self.images_dir, session_id)
            if os.path.exists(session_images_dir):
                shutil.rmtree(session_images_dir)
                deleted_count += 1
                print(f"🗑️ Deleted images directory: {session_images_dir}")
            
            # Delete metrics
            session_metrics_dir = os.path.join(self.metrics_dir, session_id)
            if os.path.exists(session_metrics_dir):
                shutil.rmtree(session_metrics_dir)
                deleted_count += 1
                print(f"🗑️ Deleted metrics directory: {session_metrics_dir}")
            
            # Optionally delete reports
            if not keep_reports:
                session_reports_dir = os.path.join(self.reports_dir, session_id)
                if os.path.exists(session_reports_dir):
                    shutil.rmtree(session_reports_dir)
                    deleted_count += 1
                    print(f"🗑️ Deleted reports directory: {session_reports_dir}")
            
            # Delete session files
            for f in os.listdir(self.sessions_dir):
                if f.startswith(f"session_{session_id}"):
                    os.remove(os.path.join(self.sessions_dir, f))
                    deleted_count += 1
            
            print(f"✅ Cleaned up session {session_id} ({deleted_count} items)")
            return True
            
        except Exception as e:
            print(f"❌ Error cleaning up session: {e}")
            return False
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        stats = {
            "base_directory": self.base_dir,
            "total_size_mb": 0,
            "file_counts": {},
            "directory_sizes": {}
        }
        
        try:
            # Calculate directory sizes and file counts
            for subdir_name in ["images", "reports", "metrics", "sessions", "logs"]:
                subdir_path = getattr(self, f"{subdir_name}_dir")
                
                if os.path.exists(subdir_path):
                    size_bytes = 0
                    file_count = 0
                    
                    for root, dirs, files in os.walk(subdir_path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            if os.path.exists(file_path):
                                size_bytes += os.path.getsize(file_path)
                                file_count += 1
                    
                    stats["directory_sizes"][subdir_name] = round(size_bytes / (1024*1024), 2)  # MB
                    stats["file_counts"][subdir_name] = file_count
                    stats["total_size_mb"] += stats["directory_sizes"][subdir_name]
            
            stats["total_size_mb"] = round(stats["total_size_mb"], 2)
            
        except Exception as e:
            print(f"❌ Error calculating storage stats: {e}")
        
        return stats