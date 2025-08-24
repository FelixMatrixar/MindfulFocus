# adk_medical_evaluation/tools/data_processor.py
"""
Data processing tools for medical evaluation
Project: mindfulfocus-470008
"""

import cv2
import numpy as np
import json
import os
import time
import threading
from datetime import datetime
from typing import Dict, Any, List, Generator, Optional
from google.adk.tools.tool_context import ToolContext
from queue import Queue, Empty

def process_camera_stream(stream_config: Dict[str, Any], tool_context: ToolContext) -> Dict[str, Any]:
    """
    Process real-time camera stream for medical analysis
    
    Args:
        stream_config: Camera and processing configuration
        tool_context: ADK tool context
        
    Returns:
        Stream processing results
    """
    print("--- 📹 Tool: process_camera_stream called ---")
    
    try:
        # Extract configuration
        duration_seconds = stream_config.get('duration_seconds', 300)
        camera_index = stream_config.get('camera_index', 0)
        fps_target = stream_config.get('fps_target', 10)  # Process every nth frame
        frame_skip = max(1, int(30 / fps_target))  # Skip frames to achieve target FPS
        
        # Initialize camera
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            return {
                "status": "error",
                "message": f"Cannot open camera {camera_index}"
            }
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, stream_config.get('width', 1280))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, stream_config.get('height', 720))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"--- 📹 Camera initialized: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)} @ {actual_fps} FPS ---")
        
        # Setup storage
        session_id = tool_context.state.get('session_id', 'unknown')
        images_dir = f"evaluation_data/images/{session_id}/stream"
        os.makedirs(images_dir, exist_ok=True)
        
        # Processing variables
        frame_count = 0
        processed_count = 0
        start_time = time.time()
        frame_queue = Queue(maxsize=100)
        
        # Update context state
        tool_context.state['stream_active'] = True
        tool_context.state['stream_start_time'] = start_time
        
        try:
            # Capture frames
            while (time.time() - start_time) < duration_seconds:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Skip frames to achieve target FPS
                if frame_count % frame_skip != 0:
                    continue
                
                processed_count += 1
                
                # Save frame
                frame_filename = f"stream_frame_{processed_count:06d}.jpg"
                frame_path = os.path.join(images_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                
                # Prepare frame data for analysis queue
                frame_data = {
                    "frame_id": processed_count,
                    "image_path": frame_path,
                    "timestamp": time.time(),
                    "original_frame_number": frame_count
                }
                
                # Add to processing queue
                try:
                    frame_queue.put(frame_data, timeout=1.0)
                except:
                    print(f"⚠️ Frame queue full, dropping frame {processed_count}")
                
                # Update progress in state
                if processed_count % 30 == 0:
                    elapsed = time.time() - start_time
                    actual_fps = processed_count / elapsed
                    tool_context.state['stream_progress'] = {
                        'frames_captured': frame_count,
                        'frames_processed': processed_count,
                        'elapsed_seconds': elapsed,
                        'actual_fps': actual_fps
                    }
                    print(f"--- 📊 Stream progress: {processed_count} frames, {actual_fps:.1f} FPS ---")
        
        finally:
            cap.release()
            tool_context.state['stream_active'] = False
        
        # Final statistics
        total_time = time.time() - start_time
        actual_processing_fps = processed_count / total_time
        
        stream_results = {
            "status": "success",
            "total_frames_captured": frame_count,
            "frames_processed": processed_count,
            "duration_seconds": round(total_time, 2),
            "target_fps": fps_target,
            "actual_fps": round(actual_processing_fps, 2),
            "frames_directory": images_dir,
            "frame_queue_size": frame_queue.qsize()
        }
        
        # Store frame queue in context for batch processing
        tool_context.state['frame_queue'] = frame_queue
        tool_context.state['stream_results'] = stream_results
        
        print(f"--- ✅ Stream processing complete: {processed_count} frames in {total_time:.1f}s ---")
        
        return stream_results
        
    except Exception as e:
        if 'cap' in locals():
            cap.release()
        
        return {
            "status": "error",
            "message": f"Stream processing failed: {str(e)}",
            "error_type": type(e).__name__
        }

def batch_process_frames(batch_config: Dict[str, Any], tool_context: ToolContext) -> Dict[str, Any]:
    """
    Process frames in batches for efficient analysis
    
    Args:
        batch_config: Batch processing configuration
        tool_context: ADK tool context
        
    Returns:
        Batch processing results
    """
    print("--- 📦 Tool: batch_process_frames called ---")
    
    try:
        # Get frame queue from context
        frame_queue = tool_context.state.get('frame_queue')
        if not frame_queue:
            return {
                "status": "error",
                "message": "No frame queue found in context. Run process_camera_stream first."
            }
        
        batch_size = batch_config.get('batch_size', 10)
        max_workers = batch_config.get('max_workers', 2)
        
        # Import analysis tool
        from .pipeline_analyzer import analyze_pipeline_frame
        
        # Process frames in batches
        batch_results = []
        batch_count = 0
        current_batch = []
        
        print(f"--- 📦 Processing {frame_queue.qsize()} frames in batches of {batch_size} ---")
        
        while True:
            try:
                # Get frame from queue
                frame_data = frame_queue.get(timeout=1.0)
                current_batch.append(frame_data)
                
                # Process batch when full
                if len(current_batch) >= batch_size:
                    batch_count += 1
                    print(f"--- 📦 Processing batch {batch_count} ({len(current_batch)} frames) ---")
                    
                    batch_start = time.time()
                    
                    # Process each frame in the batch
                    for frame_data in current_batch:
                        try:
                            result = analyze_pipeline_frame(frame_data, tool_context)
                            batch_results.append(result)
                        except Exception as e:
                            print(f"⚠️ Frame {frame_data.get('frame_id', 'unknown')} failed: {e}")
                            batch_results.append({
                                "status": "error",
                                "frame_id": frame_data.get('frame_id', 0),
                                "message": str(e)
                            })
                    
                    batch_time = time.time() - batch_start
                    avg_time_per_frame = batch_time / len(current_batch)
                    
                    print(f"--- ✅ Batch {batch_count} completed in {batch_time:.1f}s ({avg_time_per_frame*1000:.1f}ms/frame) ---")
                    
                    # Clear current batch
                    current_batch = []
                
            except Empty:
                # Queue is empty, process remaining frames
                if current_batch:
                    batch_count += 1
                    print(f"--- 📦 Processing final batch {batch_count} ({len(current_batch)} frames) ---")
                    
                    for frame_data in current_batch:
                        try:
                            result = analyze_pipeline_frame(frame_data, tool_context)
                            batch_results.append(result)
                        except Exception as e:
                            batch_results.append({
                                "status": "error",
                                "frame_id": frame_data.get('frame_id', 0),
                                "message": str(e)
                            })
                
                break  # No more frames to process
        
        # Calculate batch processing statistics
        successful_results = [r for r in batch_results if r.get('status') == 'success']
        failed_results = [r for r in batch_results if r.get('status') == 'error']
        
        processing_summary = {
            "status": "success",
            "total_frames": len(batch_results),
            "successful_frames": len(successful_results),
            "failed_frames": len(failed_results),
            "success_rate": len(successful_results) / len(batch_results) if batch_results else 0,
            "batches_processed": batch_count,
            "average_batch_size": len(batch_results) / batch_count if batch_count > 0 else 0
        }
        
        # Store results in context
        tool_context.state['batch_results'] = batch_results
        tool_context.state['batch_processing_summary'] = processing_summary
        
        print(f"--- ✅ Batch processing complete: {len(successful_results)}/{len(batch_results)} frames successful ---")
        
        return {
            "status": "success",
            "processing_summary": processing_summary,
            "batch_results": batch_results
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Batch processing failed: {str(e)}",
            "error_type": type(e).__name__
        }

def aggregate_results(aggregation_config: Dict[str, Any], tool_context: ToolContext) -> Dict[str, Any]:
    """
    Aggregate analysis results from multiple frames
    
    Args:
        aggregation_config: Aggregation configuration
        tool_context: ADK tool context
        
    Returns:
        Aggregated results
    """
    print("--- 🔄 Tool: aggregate_results called ---")
    
    try:
        # Get batch results from context
        batch_results = tool_context.state.get('batch_results', [])
        if not batch_results:
            return {
                "status": "error",
                "message": "No batch results found in context. Run batch_process_frames first."
            }
        
        # Filter successful results
        successful_results = [r for r in batch_results if r.get('status') == 'success']
        
        if not successful_results:
            return {
                "status": "error",
                "message": "No successful frame analyses to aggregate"
            }
        
        print(f"--- 🔄 Aggregating {len(successful_results)} successful results ---")
        
        # Initialize aggregation arrays
        landmarks_counts = []
        symmetry_scores = []
        ear_left_values = []
        ear_right_values = []
        ear_differences = []
        mouth_asymmetries = []
        eyebrow_differences = []
        severity_scores = []
        processing_times = []
        
        # Extract values from results
        for result in successful_results:
            landmarks_counts.append(result.get('landmarks_detected', 0))
            symmetry_scores.append(result.get('symmetry_score', 0))
            ear_left_values.append(result.get('ear_left', 0))
            ear_right_values.append(result.get('ear_right', 0))
            ear_differences.append(result.get('ear_difference', 0))
            mouth_asymmetries.append(result.get('mouth_asymmetry_mm', 0))
            eyebrow_differences.append(result.get('eyebrow_diff_mm', 0))
            severity_scores.append(result.get('severity_score', 0))
            processing_times.append(result.get('processing_time_ms', 0))
        
        # Calculate comprehensive statistics
        def calc_stats(values):
            if not values:
                return {"mean": 0, "std": 0, "min": 0, "max": 0, "median": 0}
            
            return {
                "mean": round(np.mean(values), 4),
                "std": round(np.std(values), 4),
                "min": round(np.min(values), 4),
                "max": round(np.max(values), 4),
                "median": round(np.median(values), 4),
                "percentile_25": round(np.percentile(values, 25), 4),
                "percentile_75": round(np.percentile(values, 75), 4)
            }
        
        # Aggregate all metrics
        aggregated_metrics = {
            "landmarks_detected": calc_stats(landmarks_counts),
            "symmetry_score": calc_stats(symmetry_scores),
            "ear_left": calc_stats(ear_left_values),
            "ear_right": calc_stats(ear_right_values),
            "ear_difference": calc_stats(ear_differences),
            "mouth_asymmetry_mm": calc_stats(mouth_asymmetries),
            "eyebrow_difference_mm": calc_stats(eyebrow_differences),
            "severity_score": calc_stats(severity_scores),
            "processing_time_ms": calc_stats(processing_times)
        }
        
        # Calculate derived metrics
        total_frames = len(batch_results)
        successful_frames = len(successful_results)
        success_rate = successful_frames / total_frames if total_frames > 0 else 0
        
        # Temporal analysis
        frame_ids = [r.get('frame_id', 0) for r in successful_results]
        if len(frame_ids) > 1:
            temporal_span = max(frame_ids) - min(frame_ids)
            temporal_coverage = len(frame_ids) / (temporal_span + 1) if temporal_span > 0 else 1.0
        else:
            temporal_span = 0
            temporal_coverage = 1.0
        
        # Quality assessment
        quality_indicators = {
            "adequate_landmarks": sum(1 for c in landmarks_counts if c >= 400) / len(landmarks_counts) if landmarks_counts else 0,
            "good_symmetry": sum(1 for s in symmetry_scores if s >= 0.7) / len(symmetry_scores) if symmetry_scores else 0,
            "normal_ear_difference": sum(1 for e in ear_differences if e <= 0.05) / len(ear_differences) if ear_differences else 0,
            "low_severity": sum(1 for s in severity_scores if s <= 5.0) / len(severity_scores) if severity_scores else 0
        }
        
        overall_quality = np.mean(list(quality_indicators.values()))
        
        # Clinical flags based on aggregated data
        clinical_flags = {
            "persistent_asymmetry": aggregated_metrics["symmetry_score"]["mean"] < 0.7,
            "significant_ear_asymmetry": aggregated_metrics["ear_difference"]["mean"] > 0.05,
            "mouth_asymmetry_concern": aggregated_metrics["mouth_asymmetry_mm"]["mean"] > 3.0,
            "elevated_severity": aggregated_metrics["severity_score"]["mean"] > 6.0,
            "high_variability": (aggregated_metrics["symmetry_score"]["std"] > 0.1 or 
                                aggregated_metrics["ear_difference"]["std"] > 0.02),
            "processing_delays": aggregated_metrics["processing_time_ms"]["mean"] > 500
        }
        
        active_flags = sum(clinical_flags.values())
        
        # Performance assessment
        performance_metrics = {
            "total_frames_analyzed": total_frames,
            "successful_analyses": successful_frames,
            "success_rate": round(success_rate, 4),
            "average_processing_time_ms": aggregated_metrics["processing_time_ms"]["mean"],
            "estimated_fps": round(1000 / aggregated_metrics["processing_time_ms"]["mean"], 2) if aggregated_metrics["processing_time_ms"]["mean"] > 0 else 0,
            "temporal_span_frames": temporal_span,
            "temporal_coverage": round(temporal_coverage, 4)
        }
        
        # Generate summary assessment
        if overall_quality >= 0.8 and active_flags <= 1:
            summary_assessment = "excellent"
            deployment_recommendation = "ready_for_deployment"
        elif overall_quality >= 0.6 and active_flags <= 2:
            summary_assessment = "good"
            deployment_recommendation = "minor_improvements_needed"
        elif overall_quality >= 0.4 and active_flags <= 3:
            summary_assessment = "fair"
            deployment_recommendation = "significant_improvements_needed"
        else:
            summary_assessment = "poor"
            deployment_recommendation = "major_revisions_required"
        
        # Compile comprehensive aggregation result
        aggregation_result = {
            "status": "success",
            "aggregation_timestamp": datetime.now().isoformat(),
            "input_summary": {
                "total_frames": total_frames,
                "successful_frames": successful_frames,
                "failed_frames": total_frames - successful_frames,
                "success_rate": round(success_rate, 4)
            },
            "aggregated_metrics": aggregated_metrics,
            "quality_indicators": quality_indicators,
            "overall_quality_score": round(overall_quality, 4),
            "clinical_flags": clinical_flags,
            "active_clinical_flags": active_flags,
            "performance_metrics": performance_metrics,
            "summary_assessment": summary_assessment,
            "deployment_recommendation": deployment_recommendation
        }
        
        # Store in context
        tool_context.state['aggregated_results'] = aggregation_result
        tool_context.state['aggregation_complete'] = True
        
        # Save aggregation to file
        session_id = tool_context.state.get('session_id', 'unknown')
        aggregation_file = f"evaluation_data/reports/{session_id}/aggregated_results.json"
        os.makedirs(os.path.dirname(aggregation_file), exist_ok=True)
        
        with open(aggregation_file, 'w') as f:
            json.dump(aggregation_result, f, indent=2)
        
        print(f"--- ✅ Aggregation complete: {summary_assessment} quality, {active_flags} clinical flags ---")
        
        return aggregation_result
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Results aggregation failed: {str(e)}",
            "error_type": type(e).__name__
        }

def export_results(export_config: Dict[str, Any], tool_context: ToolContext) -> Dict[str, Any]:
    """
    Export analysis results in various formats
    
    Args:
        export_config: Export configuration
        tool_context: ADK tool context
        
    Returns:
        Export results
    """
    print("--- 📤 Tool: export_results called ---")
    
    try:
        # Get aggregated results
        aggregated_results = tool_context.state.get('aggregated_results')
        if not aggregated_results:
            return {
                "status": "error",
                "message": "No aggregated results found. Run aggregate_results first."
            }
        
        session_id = tool_context.state.get('session_id', 'unknown')
        export_formats = export_config.get('formats', ['json', 'csv', 'txt'])
        export_dir = f"evaluation_data/exports/{session_id}"
        os.makedirs(export_dir, exist_ok=True)
        
        exported_files = {}
        
        # JSON export (comprehensive)
        if 'json' in export_formats:
            json_file = os.path.join(export_dir, f"medical_evaluation_{session_id}.json")
            
            export_data = {
                "export_metadata": {
                    "session_id": session_id,
                    "export_timestamp": datetime.now().isoformat(),
                    "project_id": "mindfulfocus-470008",
                    "model": "gemini-1.5-pro",
                    "export_version": "1.0"
                },
                "session_summary": {
                    "frames_analyzed": tool_context.state.get('frames_analyzed', 0),
                    "session_duration": tool_context.state.get('evaluation_duration', 0),
                    "analysis_start": tool_context.state.get('session_created', ''),
                    "analysis_end": datetime.now().isoformat()
                },
                "aggregated_results": aggregated_results,
                "performance_metrics": tool_context.state.get('latest_performance_metrics', {}),
                "medical_assessment": tool_context.state.get('medical_assessment', {})
            }
            
            with open(json_file, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            exported_files['json'] = json_file
            print(f"--- 📄 JSON export: {os.path.basename(json_file)} ---")
        
        # CSV export (metrics only)
        if 'csv' in export_formats:
            import pandas as pd
            
            csv_file = os.path.join(export_dir, f"medical_metrics_{session_id}.csv")
            
            # Extract batch results for CSV
            batch_results = tool_context.state.get('batch_results', [])
            successful_results = [r for r in batch_results if r.get('status') == 'success']
            
            if successful_results:
                # Create DataFrame
                csv_data = []
                for result in successful_results:
                    row = {
                        'frame_id': result.get('frame_id', 0),
                        'landmarks_detected': result.get('landmarks_detected', 0),
                        'symmetry_score': result.get('symmetry_score', 0),
                        'ear_left': result.get('ear_left', 0),
                        'ear_right': result.get('ear_right', 0),
                        'ear_difference': result.get('ear_difference', 0),
                        'mouth_asymmetry_mm': result.get('mouth_asymmetry_mm', 0),
                        'eyebrow_diff_mm': result.get('eyebrow_diff_mm', 0),
                        'severity_score': result.get('severity_score', 0),
                        'processing_time_ms': result.get('processing_time_ms', 0),
                        'timestamp': result.get('analysis_timestamp', '')
                    }
                    csv_data.append(row)
                
                df = pd.DataFrame(csv_data)
                df.to_csv(csv_file, index=False)
                
                exported_files['csv'] = csv_file
                print(f"--- 📊 CSV export: {os.path.basename(csv_file)} ({len(csv_data)} rows) ---")
        
        # Text summary export
        if 'txt' in export_formats:
            txt_file = os.path.join(export_dir, f"medical_summary_{session_id}.txt")
            
            # Generate human-readable summary
            summary_text = generate_summary_text(aggregated_results, tool_context)
            
            with open(txt_file, 'w') as f:
                f.write(summary_text)
            
            exported_files['txt'] = txt_file
            print(f"--- 📝 Text export: {os.path.basename(txt_file)} ---")
        
        # Medical report export (if requested)
        if 'medical_report' in export_formats:
            report_file = os.path.join(export_dir, f"clinical_report_{session_id}.txt")
            
            clinical_report = generate_clinical_report(aggregated_results, tool_context)
            
            with open(report_file, 'w') as f:
                f.write(clinical_report)
            
            exported_files['medical_report'] = report_file
            print(f"--- 🏥 Medical report: {os.path.basename(report_file)} ---")
        
        # Update context
        tool_context.state['export_complete'] = True
        tool_context.state['exported_files'] = exported_files
        
        export_summary = {
            "status": "success",
            "exported_files": exported_files,
            "export_directory": export_dir,
            "formats_exported": list(exported_files.keys()),
            "export_timestamp": datetime.now().isoformat()
        }
        
        print(f"--- ✅ Export complete: {len(exported_files)} files in {len(export_formats)} formats ---")
        
        return export_summary
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Results export failed: {str(e)}",
            "error_type": type(e).__name__
        }

def generate_summary_text(aggregated_results: Dict[str, Any], tool_context: ToolContext) -> str:
    """Generate human-readable summary text"""
    
    session_id = tool_context.state.get('session_id', 'unknown')
    
    summary = f"""
    MEDICAL FACE ANALYSIS SUMMARY REPORT
    ═══════════════════════════════════════════════════════════════

    Session ID: {session_id}
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    Project: mindfulfocus-470008
    Model: Gemini 1.5 Pro

    OVERALL ASSESSMENT
    ──────────────────────────────────────────────────────────────
    Quality Score: {aggregated_results.get('overall_quality_score', 0):.3f}/1.0
    Assessment: {aggregated_results.get('summary_assessment', 'unknown').upper()}
    Deployment Status: {aggregated_results.get('deployment_recommendation', 'unknown').replace('_', ' ').title()}

    ANALYSIS SUMMARY
    ──────────────────────────────────────────────────────────────
    Total Frames: {aggregated_results.get('input_summary', {}).get('total_frames', 0)}
    Successful Analyses: {aggregated_results.get('input_summary', {}).get('successful_frames', 0)}
    Success Rate: {aggregated_results.get('input_summary', {}).get('success_rate', 0):.1%}

    KEY METRICS (Average Values)
    ──────────────────────────────────────────────────────────────
    """
    
    metrics = aggregated_results.get('aggregated_metrics', {})
    
    summary += f"Facial Symmetry Score: {metrics.get('symmetry_score', {}).get('mean', 0):.4f}\n"
    summary += f"Eye Aspect Ratio Difference: {metrics.get('ear_difference', {}).get('mean', 0):.4f}\n"
    summary += f"Mouth Asymmetry: {metrics.get('mouth_asymmetry_mm', {}).get('mean', 0):.2f} mm\n"
    summary += f"Severity Score: {metrics.get('severity_score', {}).get('mean', 0):.2f}/10.0\n"
    summary += f"Processing Time: {metrics.get('processing_time_ms', {}).get('mean', 0):.1f} ms/frame\n"
    
    summary += f"""
    QUALITY INDICATORS
    ──────────────────────────────────────────────────────────────
    """
    
    quality = aggregated_results.get('quality_indicators', {})
    for indicator, value in quality.items():
        summary += f"{indicator.replace('_', ' ').title()}: {value:.1%}\n"
    
    summary += f"""
    CLINICAL FLAGS
    ──────────────────────────────────────────────────────────────
    """
    
    flags = aggregated_results.get('clinical_flags', {})
    active_flags = [flag.replace('_', ' ').title() for flag, active in flags.items() if active]
    
    if active_flags:
        for flag in active_flags:
            summary += f"⚠️ {flag}\n"
    else:
        summary += "✅ No significant clinical flags detected\n"
    
    summary += f"""
    RECOMMENDATIONS
    ──────────────────────────────────────────────────────────────
    Based on this analysis:

    """
    
    recommendation = aggregated_results.get('deployment_recommendation', '')
    if recommendation == 'ready_for_deployment':
        summary += "✅ System appears ready for clinical deployment\n"
        summary += "✅ All quality metrics within acceptable ranges\n"
        summary += "✅ Minimal clinical flags detected\n"
    elif recommendation == 'minor_improvements_needed':
        summary += "⚡ System shows good performance with minor areas for improvement\n"
        summary += "⚡ Consider addressing identified clinical flags\n"
        summary += "⚡ Monitor consistency metrics\n"
    else:
        summary += "❌ System requires significant improvements before deployment\n"
        summary += "❌ Address multiple clinical flags\n"
        summary += "❌ Improve accuracy and consistency metrics\n"
    
    summary += f"""
    Generated by MindfulFocus Medical Pipeline Evaluator
    ═══════════════════════════════════════════════════════════════
    """
    
    return summary

def generate_clinical_report(aggregated_results: Dict[str, Any], tool_context: ToolContext) -> str:
    """Generate clinical assessment report"""
    
    session_id = tool_context.state.get('session_id', 'unknown')
    medical_assessment = tool_context.state.get('medical_assessment', {})
    
    report = f"""
    CLINICAL ASSESSMENT REPORT
    ═══════════════════════════════════════════════════════════════

    Patient/Session ID: {session_id}
    Assessment Date: {datetime.now().strftime('%Y-%m-%d')}
    Analysis System: MindfulFocus Medical Pipeline v1.0
    Model: Gemini 1.5 Pro

    CLINICAL FINDINGS
    ──────────────────────────────────────────────────────────────
    """
    
    if medical_assessment:
        severity = medical_assessment.get('severity_level', 'unknown')
        report += f"Severity Level: {severity.upper()}\n\n"
        
        clinical_summary = medical_assessment.get('clinical_summary', '')
        report += f"{clinical_summary}\n"
        
        report += "\nPATTERN ANALYSIS\n"
        report += "──────────────────────────────────────────────────────────────\n"
        
        patterns = medical_assessment.get('pattern_analysis', {})
        for pattern, value in patterns.items():
            report += f"{pattern.replace('_', ' ').title()}: {value}%\n"
        
        report += "\nCLINICAL RECOMMENDATIONS\n"
        report += "──────────────────────────────────────────────────────────────\n"
        
        recommendations = medical_assessment.get('recommendations', [])
        for i, rec in enumerate(recommendations, 1):
            report += f"{i}. {rec}\n"
        
        report += "\nNEXT STEPS\n"
        report += "──────────────────────────────────────────────────────────────\n"
        
        next_steps = medical_assessment.get('next_steps', [])
        for i, step in enumerate(next_steps, 1):
            report += f"{i}. {step}\n"
    
    else:
        report += "No detailed medical assessment available.\n"
        report += "Consider running generate_medical_assessment tool for clinical evaluation.\n"
    
    disclaimer = medical_assessment.get('disclaimer', 
        "This assessment is for informational purposes only and does not constitute medical advice.")
    
    report += f"""
    IMPORTANT DISCLAIMER
    ──────────────────────────────────────────────────────────────
    {disclaimer}

    This report should be reviewed by qualified medical professionals
    before any clinical decisions are made.

    Report Generated: {datetime.now().isoformat()}
    System: MindfulFocus Medical Pipeline Evaluator
    Project ID: mindfulfocus-470008
    ═══════════════════════════════════════════════════════════════
    """
    
    return report