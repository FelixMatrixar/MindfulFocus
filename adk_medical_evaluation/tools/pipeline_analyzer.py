# adk_medical_evaluation/tools/pipeline_analyzer.py
"""
Pipeline analysis tools for medical evaluation
Project: mindfulfocus-470008
Model: Gemini 1.5 Pro
"""

import json
import cv2
import numpy as np
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from google.adk.tools.tool_context import ToolContext

# Import existing medical analysis components
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'medical_analysis'))

try:
    from landmark_analyzer import MedicalLandmarkAnalyzer, metrics_to_json
    print("✅ Successfully imported medical analysis components")
except ImportError as e:
    print(f"❌ Error importing medical analysis components: {e}")
    print("Please ensure medical_analysis directory exists with landmark_analyzer.py")

def analyze_pipeline_frame(frame_data: Dict[str, Any], tool_context: ToolContext) -> Dict[str, Any]:
    """
    Analyze a single frame from the medical pipeline using existing components
    
    Args:
        frame_data: Dictionary containing frame information and image path
        tool_context: ADK tool context for accessing state and session info
        
    Returns:
        Dictionary with comprehensive analysis results
    """
    print(f"--- 🔬 Tool: analyze_pipeline_frame called for frame {frame_data.get('frame_id', 'unknown')} ---")
    
    start_time = time.time()
    
    try:
        # Initialize or get analyzer from state
        if 'landmark_analyzer' not in tool_context.state:
            tool_context.state['landmark_analyzer'] = MedicalLandmarkAnalyzer()
            print("--- 🔧 Tool: Initialized MedicalLandmarkAnalyzer ---")
        
        analyzer = tool_context.state['landmark_analyzer']
        
        # Validate frame data
        frame_id = frame_data.get('frame_id', 0)
        image_path = frame_data.get('image_path', '')
        
        if not image_path or not os.path.exists(image_path):
            return {
                "status": "error",
                "message": f"Invalid or missing image path: {image_path}",
                "frame_id": frame_id
            }
        
        # Load and validate image
        image = cv2.imread(image_path)
        if image is None:
            return {
                "status": "error", 
                "message": f"Could not load image from: {image_path}",
                "frame_id": frame_id
            }
        
        print(f"--- 📸 Tool: Loaded image {os.path.basename(image_path)}, shape: {image.shape} ---")
        
        # Extract facial landmarks
        landmarks = analyzer.extract_landmarks(image)
        
        if landmarks is None:
            return {
                "status": "partial_success",
                "message": "No face detected in frame",
                "frame_id": frame_id,
                "landmarks_detected": 0,
                "processing_time_ms": (time.time() - start_time) * 1000
            }
        
        # Calculate comprehensive metrics
        metrics = analyzer.calculate_medical_metrics(landmarks, image.shape)
        metrics_json = metrics_to_json(metrics)
        metrics_dict = json.loads(metrics_json)
        
        # Extract key values for database storage
        landmarks_count = len(landmarks.landmark)
        symmetry_score = metrics.symmetry_score
        ear_left = metrics.ear_left
        ear_right = metrics.ear_right
        ear_difference = abs(ear_left - ear_right)
        mouth_asymmetry = metrics.mouth_corner_deviation_mm
        eyebrow_diff = metrics.eyebrow_height_difference_mm
        
        # Calculate severity score
        severity_score = calculate_severity_score(metrics)
        
        # Create annotated image for debugging/visualization
        annotated_image = analyzer.create_annotated_image(image, landmarks)
        
        # Save annotated image
        session_id = tool_context.state.get('session_id', 'unknown')
        annotated_dir = f"evaluation_data/images/{session_id}/annotated"
        os.makedirs(annotated_dir, exist_ok=True)
        
        annotated_path = os.path.join(annotated_dir, f"annotated_frame_{frame_id:06d}.jpg")
        cv2.imwrite(annotated_path, annotated_image)
        
        # Store detailed metrics locally
        metrics_dir = f"evaluation_data/metrics/{session_id}"
        os.makedirs(metrics_dir, exist_ok=True)
        
        frame_metrics_file = os.path.join(metrics_dir, f"frame_{frame_id:06d}_detailed.json")
        with open(frame_metrics_file, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Update session state
        frames_analyzed = tool_context.state.get('frames_analyzed', 0) + 1
        tool_context.state['frames_analyzed'] = frames_analyzed
        tool_context.state['last_frame_analyzed'] = frame_id
        tool_context.state['last_analysis_time'] = datetime.now().isoformat()
        
        # Prepare comprehensive result
        analysis_result = {
            "status": "success",
            "frame_id": frame_id,
            "image_path": image_path,
            "annotated_path": annotated_path,
            "landmarks_detected": landmarks_count,
            "symmetry_score": round(symmetry_score, 4),
            "ear_left": round(ear_left, 4),
            "ear_right": round(ear_right, 4),
            "ear_difference": round(ear_difference, 4),
            "mouth_asymmetry_mm": round(mouth_asymmetry, 2),
            "eyebrow_diff_mm": round(eyebrow_diff, 2),
            "severity_score": round(severity_score, 2),
            "processing_time_ms": round(processing_time_ms, 2),
            "full_metrics": metrics_dict,
            "metrics_file": frame_metrics_file,
            "total_frames_analyzed": frames_analyzed,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        # Log to database if available
        if 'sqlite_storage' in tool_context.state:
            storage = tool_context.state['sqlite_storage']
            evaluation_run_id = tool_context.state.get('current_evaluation_run_id')
            
            if evaluation_run_id:
                storage.save_frame_analysis_detailed(evaluation_run_id, analysis_result)
        
        print(f"--- ✅ Tool: Frame {frame_id} analyzed successfully in {processing_time_ms:.1f}ms ---")
        print(f"--- 📊 Landmarks: {landmarks_count}, Symmetry: {symmetry_score:.3f}, Severity: {severity_score:.1f} ---")
        
        return analysis_result
        
    except Exception as e:
        processing_time_ms = (time.time() - start_time) * 1000
        error_msg = f"Analysis failed for frame {frame_data.get('frame_id', 'unknown')}: {str(e)}"
        
        print(f"--- ❌ Tool: {error_msg} ---")
        
        return {
            "status": "error",
            "message": error_msg,
            "frame_id": frame_data.get('frame_id', 0),
            "processing_time_ms": processing_time_ms,
            "error_type": type(e).__name__
        }

def calculate_performance_metrics(results_list: List[Dict], tool_context: ToolContext) -> Dict[str, Any]:
    """
    Calculate comprehensive performance metrics from frame analysis results
    
    Args:
        results_list: List of frame analysis results
        tool_context: ADK tool context
        
    Returns:
        Performance metrics dictionary
    """
    print(f"--- 📊 Tool: calculate_performance_metrics called with {len(results_list)} results ---")
    
    if not results_list:
        return {"status": "error", "message": "No results provided for analysis"}
    
    start_time = time.time()
    
    try:
        # Separate successful and failed analyses
        successful_results = [r for r in results_list if r.get('status') == 'success']
        partial_results = [r for r in results_list if r.get('status') == 'partial_success']
        failed_results = [r for r in results_list if r.get('status') == 'error']
        
        total_frames = len(results_list)
        successful_frames = len(successful_results)
        partial_frames = len(partial_results)
        failed_frames = len(failed_results)
        
        # Basic success metrics
        success_rate = successful_frames
        success_rate = successful_frames / total_frames if total_frames > 0 else 0
        detection_rate = (successful_frames + partial_frames) / total_frames if total_frames > 0 else 0
        
        # Performance timing metrics
        processing_times = [r.get('processing_time_ms', 0) for r in results_list if 'processing_time_ms' in r]
        avg_processing_time = np.mean(processing_times) if processing_times else 0
        max_processing_time = np.max(processing_times) if processing_times else 0
        min_processing_time = np.min(processing_times) if processing_times else 0
        
        # Medical accuracy metrics from successful frames only
        if successful_results:
            landmarks_counts = [r.get('landmarks_detected', 0) for r in successful_results]
            symmetry_scores = [r.get('symmetry_score', 0) for r in successful_results]
            ear_differences = [r.get('ear_difference', 0) for r in successful_results]
            mouth_asymmetries = [r.get('mouth_asymmetry_mm', 0) for r in successful_results]
            eyebrow_diffs = [r.get('eyebrow_diff_mm', 0) for r in successful_results]
            severity_scores = [r.get('severity_score', 0) for r in successful_results]
            
            # Calculate averages and standard deviations
            avg_landmarks = np.mean(landmarks_counts)
            avg_symmetry = np.mean(symmetry_scores)
            std_symmetry = np.std(symmetry_scores)
            avg_ear_diff = np.mean(ear_differences)
            std_ear_diff = np.std(ear_differences)
            avg_mouth_asymmetry = np.mean(mouth_asymmetries)
            avg_eyebrow_diff = np.mean(eyebrow_diffs)
            avg_severity = np.mean(severity_scores)
            std_severity = np.std(severity_scores)
            
            # Consistency scores (higher is better)
            symmetry_consistency = max(0, 1 - std_symmetry) if std_symmetry < 1 else 0
            ear_consistency = max(0, 1 - (std_ear_diff * 10)) if std_ear_diff < 0.1 else 0
            severity_consistency = max(0, 1 - (std_severity / 10)) if std_severity < 10 else 0
            
            overall_consistency = (symmetry_consistency + ear_consistency + severity_consistency) / 3
            
        else:
            # No successful results
            avg_landmarks = avg_symmetry = std_symmetry = 0
            avg_ear_diff = std_ear_diff = avg_mouth_asymmetry = 0
            avg_eyebrow_diff = avg_severity = std_severity = 0
            symmetry_consistency = ear_consistency = severity_consistency = 0
            overall_consistency = 0
        
        # Clinical relevance assessment
        clinical_flags = {
            "high_asymmetry_frames": len([r for r in successful_results if r.get('symmetry_score', 1) < 0.7]),
            "significant_ear_diff_frames": len([r for r in successful_results if r.get('ear_difference', 0) > 0.05]),
            "high_severity_frames": len([r for r in successful_results if r.get('severity_score', 0) > 7.0]),
            "mouth_asymmetry_frames": len([r for r in successful_results if r.get('mouth_asymmetry_mm', 0) > 3.0])
        }
        
        # Calculate overall quality score (0-1)
        quality_factors = [
            success_rate * 0.3,  # 30% weight for detection success
            (avg_symmetry if successful_results else 0) * 0.2,  # 20% for symmetry quality
            (1 - avg_ear_diff * 10 if avg_ear_diff < 0.1 else 0) * 0.15,  # 15% for EAR quality
            overall_consistency * 0.2,  # 20% for consistency
            (1 - avg_processing_time / 1000 if avg_processing_time < 1000 else 0) * 0.15  # 15% for speed
        ]
        
        overall_score = sum(quality_factors)
        
        # Deployment readiness assessment
        deployment_criteria = {
            "min_success_rate": success_rate >= 0.8,
            "consistent_detection": std_symmetry < 0.1 if successful_results else False,
            "reasonable_processing_time": avg_processing_time < 500,  # < 500ms per frame
            "adequate_landmark_detection": avg_landmarks >= 400 if successful_results else False
        }
        
        deployment_ready = all(deployment_criteria.values())
        
        # Create comprehensive metrics dictionary
        performance_metrics = {
            # Basic counts and rates
            "total_frames": total_frames,
            "successful_frames": successful_frames,
            "partial_success_frames": partial_frames,
            "failed_frames": failed_frames,
            "success_rate": round(success_rate, 4),
            "detection_rate": round(detection_rate, 4),
            
            # Performance timing
            "average_processing_time_ms": round(avg_processing_time, 2),
            "max_processing_time_ms": round(max_processing_time, 2),
            "min_processing_time_ms": round(min_processing_time, 2),
            "estimated_fps": round(1000 / avg_processing_time, 2) if avg_processing_time > 0 else 0,
            
            # Medical accuracy metrics
            "average_landmarks_detected": round(avg_landmarks, 1),
            "average_symmetry_score": round(avg_symmetry, 4),
            "symmetry_score_std": round(std_symmetry, 4),
            "average_ear_difference": round(avg_ear_diff, 4),
            "ear_difference_std": round(std_ear_diff, 4),
            "average_mouth_asymmetry_mm": round(avg_mouth_asymmetry, 2),
            "average_eyebrow_diff_mm": round(avg_eyebrow_diff, 2),
            "average_severity_score": round(avg_severity, 2),
            "severity_score_std": round(std_severity, 2),
            
            # Consistency metrics
            "symmetry_consistency": round(symmetry_consistency, 4),
            "ear_consistency": round(ear_consistency, 4),
            "severity_consistency": round(severity_consistency, 4),
            "overall_consistency": round(overall_consistency, 4),
            
            # Clinical flags
            "clinical_flags": clinical_flags,
            
            # Overall assessment
            "overall_quality_score": round(overall_score, 4),
            "deployment_ready": deployment_ready,
            "deployment_criteria": deployment_criteria,
            
            # Metadata
            "analysis_timestamp": datetime.now().isoformat(),
            "calculation_time_ms": round((time.time() - start_time) * 1000, 2)
        }
        
        # Store metrics in session state
        tool_context.state['latest_performance_metrics'] = performance_metrics
        tool_context.state['last_metrics_calculation'] = datetime.now().isoformat()
        
        # Save detailed metrics to file
        session_id = tool_context.state.get('session_id', 'unknown')
        metrics_file = f"evaluation_data/reports/{session_id}/performance_metrics_detailed.json"
        os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
        
        with open(metrics_file, 'w') as f:
            json.dump({
                "performance_metrics": performance_metrics,
                "frame_details": results_list,
                "project_id": "mindfulfocus-470008",
                "model": "gemini-1.5-pro"
            }, f, indent=2)
        
        # Log to database if available
        if 'sqlite_storage' in tool_context.state:
            storage = tool_context.state['sqlite_storage']
            evaluation_run_id = tool_context.state.get('current_evaluation_run_id')
            
            if evaluation_run_id:
                storage.save_performance_metrics(evaluation_run_id, performance_metrics)
                storage.update_evaluation_run_status(evaluation_run_id, "completed", performance_metrics)
        
        processing_time = (time.time() - start_time) * 1000
        print(f"--- ✅ Tool: Performance metrics calculated in {processing_time:.1f}ms ---")
        print(f"--- 🎯 Success rate: {success_rate:.1%}, Overall score: {overall_score:.3f} ---")
        
        return {
            "status": "success",
            "metrics": performance_metrics,
            "metrics_file": metrics_file,
            "summary": {
                "total_frames": total_frames,
                "success_rate": f"{success_rate:.1%}",
                "overall_score": f"{overall_score:.3f}",
                "deployment_ready": "✅" if deployment_ready else "❌"
            }
        }
        
    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        error_msg = f"Performance metrics calculation failed: {str(e)}"
        
        print(f"--- ❌ Tool: {error_msg} ---")
        
        return {
            "status": "error",
            "message": error_msg,
            "processing_time_ms": processing_time,
            "error_type": type(e).__name__
        }

def save_evaluation_report(report_data: Dict[str, Any], tool_context: ToolContext) -> Dict[str, Any]:
    """
    Save comprehensive evaluation report to local storage
    
    Args:
        report_data: Complete evaluation report data
        tool_context: ADK tool context
        
    Returns:
        Save status and file paths
    """
    print("--- 📋 Tool: save_evaluation_report called ---")
    
    start_time = time.time()
    
    try:
        session_id = tool_context.state.get('session_id', 'unknown')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_id = f"medical_eval_{session_id}_{timestamp}"
        
        # Prepare comprehensive report
        comprehensive_report = {
            "report_metadata": {
                "report_id": report_id,
                "session_id": session_id,
                "project_id": "mindfulfocus-470008",
                "model": "gemini-1.5-pro",
                "timestamp": datetime.now().isoformat(),
                "report_version": "1.0"
            },
            "evaluation_summary": {
                "total_duration_seconds": tool_context.state.get('evaluation_duration', 0),
                "total_frames_processed": tool_context.state.get('frames_analyzed', 0),
                "analysis_start_time": tool_context.state.get('session_created'),
                "analysis_end_time": datetime.now().isoformat()
            },
            "report_content": report_data,
            "session_state_snapshot": {
                "frames_analyzed": tool_context.state.get('frames_analyzed', 0),
                "last_frame_analyzed": tool_context.state.get('last_frame_analyzed'),
                "latest_performance_metrics": tool_context.state.get('latest_performance_metrics'),
                "evaluation_status": tool_context.state.get('evaluation_status', 'completed')
            }
        }
        
        # Create reports directory
        reports_dir = f"evaluation_data/reports/{session_id}"
        os.makedirs(reports_dir, exist_ok=True)
        
        # Save main report
        main_report_file = os.path.join(reports_dir, f"{report_id}.json")
        with open(main_report_file, 'w') as f:
            json.dump(comprehensive_report, f, indent=2, default=str)
        
        # Save summary report (condensed version)
        summary_report = {
            "report_id": report_id,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "frames_processed": tool_context.state.get('frames_analyzed', 0),
                "duration_seconds": tool_context.state.get('evaluation_duration', 0),
                "overall_score": tool_context.state.get('latest_performance_metrics', {}).get('overall_quality_score', 0),
                "deployment_ready": tool_context.state.get('latest_performance_metrics', {}).get('deployment_ready', False)
            },
            "key_metrics": tool_context.state.get('latest_performance_metrics', {})
        }
        
        summary_report_file = os.path.join(reports_dir, f"{report_id}_summary.json")
        with open(summary_report_file, 'w') as f:
            json.dump(summary_report, f, indent=2, default=str)
        
        # Generate human-readable text report
        text_report = generate_text_report(comprehensive_report)
        text_report_file = os.path.join(reports_dir, f"{report_id}_readable.txt")
        with open(text_report_file, 'w') as f:
            f.write(text_report)
        
        # Update session state
        tool_context.state['last_report_id'] = report_id
        tool_context.state['last_report_file'] = main_report_file
        tool_context.state['report_generation_time'] = datetime.now().isoformat()
        
        # Save to database if available
        if 'sqlite_storage' in tool_context.state:
            storage = tool_context.state['sqlite_storage']
            evaluation_run_id = tool_context.state.get('current_evaluation_run_id')
            
            if evaluation_run_id:
                storage.save_report(evaluation_run_id, "final", comprehensive_report, main_report_file)
        
        processing_time = (time.time() - start_time) * 1000
        
        print(f"--- ✅ Tool: Report saved successfully in {processing_time:.1f}ms ---")
        print(f"--- 📁 Files: {os.path.basename(main_report_file)} + summary + readable ---")
        
        return {
            "status": "success",
            "report_id": report_id,
            "files": {
                "main_report": main_report_file,
                "summary_report": summary_report_file,
                "text_report": text_report_file
            },
            "message": f"Report {report_id} saved successfully",
            "processing_time_ms": processing_time
        }
        
    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        error_msg = f"Failed to save evaluation report: {str(e)}"
        
        print(f"--- ❌ Tool: {error_msg} ---")
        
        return {
            "status": "error",
            "message": error_msg,
            "processing_time_ms": processing_time,
            "error_type": type(e).__name__
        }

def generate_medical_assessment(analysis_results: List[Dict[str, Any]], 
                                performance_metrics: Dict[str, Any],
                                tool_context: ToolContext) -> Dict[str, Any]:
    """
    Generate medical assessment and recommendations based on analysis results
    
    Args:
        analysis_results: List of frame analysis results
        performance_metrics: Performance metrics dictionary
        tool_context: ADK tool context
        
    Returns:
        Medical assessment with recommendations
    """
    print("--- 🏥 Tool: generate_medical_assessment called ---")
    
    try:
        successful_results = [r for r in analysis_results if r.get('status') == 'success']
        
        if not successful_results:
            return {
                "status": "insufficient_data",
                "message": "No successful frame analyses available for medical assessment"
            }
        
        # Analyze patterns for medical concerns
        high_asymmetry_frames = [r for r in successful_results if r.get('symmetry_score', 1) < 0.7]
        significant_ear_diff = [r for r in successful_results if r.get('ear_difference', 0) > 0.05]
        high_severity_frames = [r for r in successful_results if r.get('severity_score', 0) > 7.0]
        mouth_asymmetry_frames = [r for r in successful_results if r.get('mouth_asymmetry_mm', 0) > 3.0]
        
        total_frames = len(successful_results)
        
        # Calculate percentages
        asymmetry_percentage = len(high_asymmetry_frames) / total_frames * 100
        ear_diff_percentage = len(significant_ear_diff) / total_frames * 100
        severity_percentage = len(high_severity_frames) / total_frames * 100
        mouth_percentage = len(mouth_asymmetry_frames) / total_frames * 100
        
        # Generate medical flags and recommendations
        medical_flags = []
        recommendations = []
        severity_level = "normal"
        
        # Asymmetry assessment
        if asymmetry_percentage > 50:
            medical_flags.append("persistent_facial_asymmetry")
            recommendations.append("Significant facial asymmetry detected. Consider neurological evaluation.")
            severity_level = "high"
        elif asymmetry_percentage > 25:
            medical_flags.append("moderate_facial_asymmetry")
            recommendations.append("Moderate facial asymmetry observed. Monitor for progression.")
            if severity_level == "normal":
                severity_level = "moderate"
        
        # Eye movement assessment
        if ear_diff_percentage > 30:
            medical_flags.append("eye_movement_asymmetry")
            recommendations.append("Eye opening asymmetry detected. May indicate facial nerve involvement.")
            if severity_level in ["normal", "moderate"]:
                severity_level = "moderate"
        
        # Mouth assessment
        if mouth_percentage > 40:
            medical_flags.append("mouth_asymmetry")
            recommendations.append("Mouth corner asymmetry noted. Consider Bell's palsy evaluation.")
            if severity_level == "normal":
                severity_level = "moderate"
        
        # High severity assessment
        if severity_percentage > 20:
            medical_flags.append("high_severity_readings")
            recommendations.append("High severity scores detected consistently. Medical consultation recommended.")
            severity_level = "high"
        
        # Overall consistency assessment
        consistency_score = performance_metrics.get('overall_consistency', 0)
        if consistency_score < 0.6:
            medical_flags.append("inconsistent_measurements")
            recommendations.append("Measurement inconsistency detected. Repeat evaluation recommended.")
        
        # Generate clinical summary
        avg_symmetry = performance_metrics.get('average_symmetry_score', 0)
        avg_ear_diff = performance_metrics.get('average_ear_difference', 0)
        avg_severity = performance_metrics.get('average_severity_score', 0)
        
        clinical_summary = f"""
    Medical Analysis Summary:
    - Facial Symmetry Score: {avg_symmetry:.3f} (1.0 = perfect symmetry)
    - Average EAR Difference: {avg_ear_diff:.4f} (< 0.05 normal)
    - Average Severity Score: {avg_severity:.1f}/10.0
    - Consistency Score: {consistency_score:.3f}

    Pattern Analysis:
    - Asymmetry in {asymmetry_percentage:.1f}% of frames
    - Significant EAR difference in {ear_diff_percentage:.1f}% of frames  
    - High severity in {severity_percentage:.1f}% of frames
    - Mouth asymmetry in {mouth_percentage:.1f}% of frames
    """
        
        # Determine next steps
        if severity_level == "high":
            next_steps = [
                "Immediate medical consultation recommended",
                "Document symptoms and progression",
                "Consider neurological evaluation",
                "Monitor for changes in facial function"
            ]
        elif severity_level == "moderate":
            next_steps = [
                "Monitor facial symmetry over time",
                "Consider medical consultation if symptoms persist",
                "Document any functional difficulties",
                "Repeat evaluation in 1-2 weeks"
            ]
        else:
            next_steps = [
                "No immediate medical concerns identified",
                "Continue routine monitoring if desired",
                "Document baseline measurements for future reference"
            ]
        
        medical_assessment = {
            "assessment_timestamp": datetime.now().isoformat(),
            "severity_level": severity_level,
            "medical_flags": medical_flags,
            "clinical_summary": clinical_summary.strip(),
            "pattern_analysis": {
                "asymmetry_percentage": round(asymmetry_percentage, 1),
                "ear_difference_percentage": round(ear_diff_percentage, 1),
                "severity_percentage": round(severity_percentage, 1),
                "mouth_asymmetry_percentage": round(mouth_percentage, 1)
            },
            "key_metrics": {
                "average_symmetry_score": round(avg_symmetry, 4),
                "average_ear_difference": round(avg_ear_diff, 4),
                "average_severity_score": round(avg_severity, 2),
                "consistency_score": round(consistency_score, 3)
            },
            "recommendations": recommendations,
            "next_steps": next_steps,
            "disclaimer": "This assessment is for informational purposes only and does not constitute medical advice. Consult healthcare professionals for medical concerns."
        }
        
        # Store in session state
        tool_context.state['medical_assessment'] = medical_assessment
        
        print(f"--- ✅ Tool: Medical assessment generated - Severity: {severity_level} ---")
        print(f"--- 🏥 Flags: {len(medical_flags)}, Recommendations: {len(recommendations)} ---")
        
        return {
            "status": "success",
            "assessment": medical_assessment
        }
       
    except Exception as e:
        error_msg = f"Medical assessment generation failed: {str(e)}"
        print(f"--- ❌ Tool: {error_msg} ---")
        
        return {
            "status": "error",
            "message": error_msg,
            "error_type": type(e).__name__
        }

def calculate_severity_score(metrics) -> float:
    """Calculate severity score from medical metrics"""
    try:
        # Factors contributing to severity (0-10 scale)
        symmetry_penalty = (1.0 - metrics.symmetry_score) * 4  # 0-4 points
        mouth_penalty = min(3, metrics.mouth_corner_deviation_mm / 2)  # 0-3 points
        eyebrow_penalty = min(2, metrics.eyebrow_height_difference_mm / 3)  # 0-2 points
        ear_penalty = min(1, abs(metrics.ear_left - metrics.ear_right) * 20)  # 0-1 point
        
        total_severity = symmetry_penalty + mouth_penalty + eyebrow_penalty + ear_penalty
        return min(10.0, total_severity)
        
    except Exception as e:
        print(f"Warning: Could not calculate severity score: {e}")
        return 0.0

def generate_text_report(comprehensive_report: Dict[str, Any]) -> str:
    """Generate human-readable text version of the report"""
    
    metadata = comprehensive_report.get('report_metadata', {})
    summary = comprehensive_report.get('evaluation_summary', {})
    content = comprehensive_report.get('report_content', {})
    state = comprehensive_report.get('session_state_snapshot', {})
    
    performance_metrics = state.get('latest_performance_metrics', {})
    
    text_report = f"""
    MEDICAL PIPELINE EVALUATION REPORT
    ═══════════════════════════════════════════════════════════════

    Report ID: {metadata.get('report_id', 'Unknown')}
    Generated: {metadata.get('timestamp', 'Unknown')}
    Project: {metadata.get('project_id', 'Unknown')}
    Model: {metadata.get('model', 'Unknown')}

    EVALUATION SUMMARY
    ──────────────────────────────────────────────────────────────
    Duration: {summary.get('total_duration_seconds', 0)} seconds
    Frames Processed: {summary.get('total_frames_processed', 0)}
    Start Time: {summary.get('analysis_start_time', 'Unknown')}
    End Time: {summary.get('analysis_end_time', 'Unknown')}

    PERFORMANCE METRICS
    ──────────────────────────────────────────────────────────────
    Success Rate: {performance_metrics.get('success_rate', 0):.1%}
    Detection Rate: {performance_metrics.get('detection_rate', 0):.1%}
    Average Processing Time: {performance_metrics.get('average_processing_time_ms', 0):.1f} ms
    Estimated FPS: {performance_metrics.get('estimated_fps', 0):.1f}

    MEDICAL ACCURACY
    ──────────────────────────────────────────────────────────────
    Average Landmarks Detected: {performance_metrics.get('average_landmarks_detected', 0):.1f}
    Average Symmetry Score: {performance_metrics.get('average_symmetry_score', 0):.4f}
    Average EAR Difference: {performance_metrics.get('average_ear_difference', 0):.4f}
    Average Severity Score: {performance_metrics.get('average_severity_score', 0):.2f}/10.0

    CONSISTENCY ANALYSIS
    ──────────────────────────────────────────────────────────────
    Overall Consistency: {performance_metrics.get('overall_consistency', 0):.3f}
    Symmetry Consistency: {performance_metrics.get('symmetry_consistency', 0):.3f}
    EAR Consistency: {performance_metrics.get('ear_consistency', 0):.3f}

    DEPLOYMENT ASSESSMENT
    ──────────────────────────────────────────────────────────────
    Overall Quality Score: {performance_metrics.get('overall_quality_score', 0):.3f}/1.0
    Deployment Ready: {'✅ YES' if performance_metrics.get('deployment_ready', False) else '❌ NO'}

    CLINICAL FLAGS
    ──────────────────────────────────────────────────────────────
    """
    
    clinical_flags = performance_metrics.get('clinical_flags', {})
    for flag, count in clinical_flags.items():
        text_report += f"{flag.replace('_', ' ').title()}: {count} frames\n"
    
    text_report += f"""
    SUMMARY
    ──────────────────────────────────────────────────────────────
    This evaluation processed {summary.get('total_frames_processed', 0)} frames over {summary.get('total_duration_seconds', 0)} seconds.
    Overall pipeline performance: {'EXCELLENT' if performance_metrics.get('overall_quality_score', 0) > 0.8 else 'GOOD' if performance_metrics.get('overall_quality_score', 0) > 0.6 else 'NEEDS IMPROVEMENT'}

    Generated by MindfulFocus Medical Pipeline Evaluator
    Project ID: {metadata.get('project_id', 'mindfulfocus-470008')}
    ═══════════════════════════════════════════════════════════════
    """
    
    return text_report