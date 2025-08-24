# adk_medical_evaluation/tools/file_manager.py
"""
File management tools for medical evaluation
Project: mindfulfocus-470008
"""

import os
import shutil
import json
import cv2
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from google.adk.tools.tool_context import ToolContext

def save_frame_locally(frame_data: Dict[str, Any], tool_context: ToolContext) -> Dict[str, Any]:
    """
    Save camera frame to local storage with metadata
    
    Args:
        frame_data: Frame data including numpy array and metadata
        tool_context: ADK tool context
        
    Returns:
        Save result with file path
    """
    print("--- 💾 Tool: save_frame_locally called ---")
    
    try:
        frame = frame_data.get('frame')
        frame_id = frame_data.get('frame_id', 0)
        session_id = tool_context.state.get('session_id', 'unknown')
        
        if frame is None:
            return {
                "status": "error",
                "message": "No frame data provided"
            }
        
        # Create session directory
        session_dir = f"evaluation_data/images/{session_id}"
        os.makedirs(session_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"frame_{frame_id:06d}_{timestamp}.jpg"
        filepath = os.path.join(session_dir, filename)
        
        # Save image
        if isinstance(frame, np.ndarray):
            success = cv2.imwrite(filepath, frame)
            if not success:
                return {
                    "status": "error",
                    "message": f"Failed to save frame {frame_id}"
                }
        else:
            return {
                "status": "error", 
                "message": "Frame must be numpy array"
            }
        
        # Save metadata
        metadata = {
            "frame_id": frame_id,
            "filename": filename,
            "filepath": filepath,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "frame_shape": frame.shape,
            "file_size_bytes": os.path.getsize(filepath),
            "additional_metadata": frame_data.get('metadata', {})
        }
        
        metadata_file = filepath.replace('.jpg', '_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update context
        frames_saved = tool_context.state.get('frames_saved_locally', 0) + 1
        tool_context.state['frames_saved_locally'] = frames_saved
        tool_context.state['last_saved_frame'] = filepath
        
        return {
            "status": "success",
            "frame_id": frame_id,
            "filepath": filepath,
            "metadata_file": metadata_file,
            "file_size_bytes": metadata["file_size_bytes"],
            "total_frames_saved": frames_saved
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Frame save failed: {str(e)}",
            "error_type": type(e).__name__
        }

def load_frame_data(file_path: str, tool_context: ToolContext) -> Dict[str, Any]:
    """
    Load frame data from file with metadata
    
    Args:
        file_path: Path to image file
        tool_context: ADK tool context
        
    Returns:
        Loaded frame data
    """
    print(f"--- 📖 Tool: load_frame_data called for {os.path.basename(file_path)} ---")
    
    try:
        if not os.path.exists(file_path):
            return {
                "status": "error",
                "message": f"File not found: {file_path}"
            }
        
        # Load image
        frame = cv2.imread(file_path)
        if frame is None:
            return {
                "status": "error",
                "message": f"Could not load image: {file_path}"
            }
        
        # Try to load metadata
        metadata_file = file_path.replace('.jpg', '_metadata.json')
        metadata = {}
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            except:
                pass
        
        # Get file stats
        file_stats = os.stat(file_path)
        
        frame_data = {
            "status": "success",
            "frame": frame,
            "filepath": file_path,
            "frame_shape": frame.shape,
            "file_size_bytes": file_stats.st_size,
            "modification_time": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
            "metadata": metadata
        }
        
        return frame_data
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Frame load failed: {str(e)}",
            "error_type": type(e).__name__
        }

def manage_session_files(management_config: Dict[str, Any], tool_context: ToolContext) -> Dict[str, Any]:
    """
    Manage files for a session (organize, compress, cleanup)
    
    Args:
        management_config: File management configuration
        tool_context: ADK tool context
        
    Returns:
        Management results
    """
    print("--- 📁 Tool: manage_session_files called ---")
    
    try:
        session_id = tool_context.state.get('session_id', 'unknown')
        action = management_config.get('action', 'organize')  # organize, compress, cleanup
        
        session_dirs = {
            "images": f"evaluation_data/images/{session_id}",
            "metrics": f"evaluation_data/metrics/{session_id}",
            "reports": f"evaluation_data/reports/{session_id}",
            "exports": f"evaluation_data/exports/{session_id}"
        }
        
        management_results = {
            "session_id": session_id,
            "action": action,
            "processed_directories": [],
            "total_files_processed": 0,
            "total_size_bytes": 0
        }
        
        if action == "organize":
            # Organize files by type and date
            for dir_type, dir_path in session_dirs.items():
                if os.path.exists(dir_path):
                    result = organize_directory(dir_path, dir_type)
                    management_results["processed_directories"].append({
                        "directory": dir_path,
                        "type": dir_type,
                        "files_organized": result["files_organized"],
                        "subdirectories_created": result.get("subdirectories_created", 0)
                    })
                    management_results["total_files_processed"] += result["files_organized"]
        
        elif action == "compress":
            # Compress session data
            compress_result = compress_session_data(session_id, session_dirs)
            management_results.update(compress_result)
        
        elif action == "cleanup":
            # Cleanup old or unnecessary files
            cleanup_config = management_config.get('cleanup_config', {})
            cleanup_result = cleanup_session_files(session_id, session_dirs, cleanup_config)
            management_results.update(cleanup_result)
        
        # Calculate total directory size
        for dir_path in session_dirs.values():
            if os.path.exists(dir_path):
                dir_size = calculate_directory_size(dir_path)
                management_results["total_size_bytes"] += dir_size
        
        management_results["total_size_mb"] = round(management_results["total_size_bytes"] / (1024*1024), 2)
        
        # Update context
        tool_context.state['last_file_management'] = {
            "action": action,
            "timestamp": datetime.now().isoformat(),
            "results": management_results
        }
        
        print(f"--- ✅ File management complete: {action} action processed {management_results['total_files_processed']} files ---")
       
        return {
            "status": "success",
            "management_results": management_results
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"File management failed: {str(e)}",
            "error_type": type(e).__name__
        }

def cleanup_old_files(cleanup_config: Dict[str, Any], tool_context: ToolContext) -> Dict[str, Any]:
    """
    Clean up old files based on age and size criteria
    
    Args:
        cleanup_config: Cleanup configuration
        tool_context: ADK tool context
        
    Returns:
        Cleanup results
    """
    print("--- 🧹 Tool: cleanup_old_files called ---")
    
    try:
        # Default cleanup settings
        max_age_days = cleanup_config.get('max_age_days', 30)
        max_size_gb = cleanup_config.get('max_size_gb', 10)
        keep_reports = cleanup_config.get('keep_reports', True)
        
        base_dir = "evaluation_data"
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        cleanup_results = {
            "files_deleted": 0,
            "directories_deleted": 0,
            "space_freed_mb": 0,
            "sessions_cleaned": [],
            "preserved_files": 0
        }
        
        # Walk through all session directories
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                file_path = os.path.join(root, file)
                
                try:
                    file_stat = os.stat(file_path)
                    file_age = datetime.fromtimestamp(file_stat.st_mtime)
                    file_size = file_stat.st_size
                    
                    should_delete = False
                    
                    # Check age criteria
                    if file_age < cutoff_date:
                        should_delete = True
                    
                    # Skip reports if configured to keep them
                    if keep_reports and ('reports' in file_path or 'exports' in file_path):
                        should_delete = False
                        cleanup_results["preserved_files"] += 1
                    
                    # Skip metadata files of preserved files
                    if file.endswith('_metadata.json') and not should_delete:
                        should_delete = False
                    
                    if should_delete:
                        os.remove(file_path)
                        cleanup_results["files_deleted"] += 1
                        cleanup_results["space_freed_mb"] += file_size / (1024*1024)
                        
                        # Track which session was cleaned
                        session_id = extract_session_id_from_path(file_path)
                        if session_id and session_id not in cleanup_results["sessions_cleaned"]:
                            cleanup_results["sessions_cleaned"].append(session_id)
                
                except Exception as e:
                    print(f"Warning: Could not process file {file_path}: {e}")
                    continue
        
        # Remove empty directories
        for root, dirs, files in os.walk(base_dir, topdown=False):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                try:
                    if not os.listdir(dir_path):  # Directory is empty
                        os.rmdir(dir_path)
                        cleanup_results["directories_deleted"] += 1
                except:
                    continue
        
        cleanup_results["space_freed_mb"] = round(cleanup_results["space_freed_mb"], 2)
        
        # Update context
        tool_context.state['last_cleanup'] = {
            "timestamp": datetime.now().isoformat(),
            "results": cleanup_results
        }
        
        print(f"--- ✅ Cleanup complete: {cleanup_results['files_deleted']} files, {cleanup_results['space_freed_mb']} MB freed ---")
        
        return {
            "status": "success",
            "cleanup_results": cleanup_results
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Cleanup failed: {str(e)}",
            "error_type": type(e).__name__
        }

def organize_directory(dir_path: str, dir_type: str) -> Dict[str, Any]:
    """Organize files in a directory by date"""
    
    if not os.path.exists(dir_path):
        return {"files_organized": 0}
    
    files_organized = 0
    subdirectories_created = 0
    
    for file in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file)
        
        if os.path.isfile(file_path):
            try:
                # Get file modification date
                file_stat = os.stat(file_path)
                file_date = datetime.fromtimestamp(file_stat.st_mtime)
                date_str = file_date.strftime("%Y-%m-%d")
                
                # Create date subdirectory
                date_dir = os.path.join(dir_path, date_str)
                if not os.path.exists(date_dir):
                    os.makedirs(date_dir)
                    subdirectories_created += 1
                
                # Move file to date directory
                new_path = os.path.join(date_dir, file)
                if not os.path.exists(new_path):
                    shutil.move(file_path, new_path)
                    files_organized += 1
                
            except Exception as e:
                print(f"Warning: Could not organize file {file}: {e}")
                continue
    
    return {
        "files_organized": files_organized,
        "subdirectories_created": subdirectories_created
    }

def compress_session_data(session_id: str, session_dirs: Dict[str, str]) -> Dict[str, Any]:
    """Compress session data into archives"""
    import zipfile
    
    compressed_files = []
    total_compressed_size = 0
    
    for dir_type, dir_path in session_dirs.items():
        if os.path.exists(dir_path):
            archive_name = f"evaluation_data/archives/{session_id}_{dir_type}_{datetime.now().strftime('%Y%m%d')}.zip"
            os.makedirs(os.path.dirname(archive_name), exist_ok=True)
            
            try:
                with zipfile.ZipFile(archive_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for root, dirs, files in os.walk(dir_path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arc_path = os.path.relpath(file_path, dir_path)
                            zipf.write(file_path, arc_path)
                
                archive_size = os.path.getsize(archive_name)
                compressed_files.append({
                    "archive": archive_name,
                    "directory_type": dir_type,
                    "size_mb": round(archive_size / (1024*1024), 2)
                })
                total_compressed_size += archive_size
                
            except Exception as e:
                print(f"Warning: Could not compress {dir_path}: {e}")
                continue
    
    return {
        "compressed_files": compressed_files,
        "total_archives": len(compressed_files),
        "total_compressed_size_mb": round(total_compressed_size / (1024*1024), 2)
    }

def cleanup_session_files(session_id: str, session_dirs: Dict[str, str], cleanup_config: Dict[str, Any]) -> Dict[str, Any]:
    """Clean up specific session files"""
    
    keep_reports = cleanup_config.get('keep_reports', True)
    keep_latest_n = cleanup_config.get('keep_latest_frames', 100)
    
    cleanup_results = {
        "files_deleted": 0,
        "space_freed_mb": 0
    }
    
    for dir_type, dir_path in session_dirs.items():
        if not os.path.exists(dir_path):
            continue
            
        if dir_type == "reports" and keep_reports:
            continue
        
        if dir_type == "images" and keep_latest_n > 0:
            # Keep only the latest N frames
            files = []
            for file in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file)
                if os.path.isfile(file_path) and file.endswith('.jpg'):
                    file_stat = os.stat(file_path)
                    files.append((file_path, file_stat.st_mtime, file_stat.st_size))
            
            # Sort by modification time (newest first)
            files.sort(key=lambda x: x[1], reverse=True)
            
            # Delete files beyond the keep limit
            for file_path, mtime, size in files[keep_latest_n:]:
                try:
                    os.remove(file_path)
                    cleanup_results["files_deleted"] += 1
                    cleanup_results["space_freed_mb"] += size / (1024*1024)
                    
                    # Also remove metadata file if it exists
                    metadata_file = file_path.replace('.jpg', '_metadata.json')
                    if os.path.exists(metadata_file):
                        os.remove(metadata_file)
                        cleanup_results["files_deleted"] += 1
                
                except Exception as e:
                    print(f"Warning: Could not delete {file_path}: {e}")
                    continue
    
    cleanup_results["space_freed_mb"] = round(cleanup_results["space_freed_mb"], 2)
    
    return cleanup_results

def calculate_directory_size(dir_path: str) -> int:
    """Calculate total size of directory in bytes"""
    total_size = 0
    
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                total_size += os.path.getsize(file_path)
            except:
                continue
    
    return total_size

def extract_session_id_from_path(file_path: str) -> Optional[str]:
    """Extract session ID from file path"""
    path_parts = file_path.split(os.sep)
    
    # Look for session ID in path structure
    for part in path_parts:
        if part.startswith('eval_') or part.startswith('session_'):
            return part
    
    return None