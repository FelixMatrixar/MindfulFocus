# adk_medical_evaluation/tools/metrics_calculator.py
"""
Advanced metrics calculation tools for medical evaluation
Project: mindfulfocus-470008
"""

import numpy as np
import math
from typing import Dict, Any, List, Tuple, Optional
from google.adk.tools.tool_context import ToolContext

def calculate_frame_metrics(landmarks_data: Dict[str, Any], tool_context: ToolContext) -> Dict[str, Any]:
    """
    Calculate comprehensive metrics for a single frame
    
    Args:
        landmarks_data: Dictionary containing landmark positions and metadata
        tool_context: ADK tool context
        
    Returns:
        Dictionary with calculated metrics
    """
    print("--- 📏 Tool: calculate_frame_metrics called ---")
    
    try:
        # Extract landmark positions
        landmarks = landmarks_data.get('landmarks', [])
        image_width = landmarks_data.get('image_width', 640)
        image_height = landmarks_data.get('image_height', 480)
        
        if not landmarks or len(landmarks) < 468:
            return {
                "status": "error",
                "message": f"Insufficient landmarks provided: {len(landmarks)}/468"
            }
        
        # Calculate various metric categories
        symmetry_metrics = calculate_symmetry_analysis(landmarks, image_width, image_height, tool_context)
        ear_metrics = calculate_ear_analysis(landmarks, image_width, image_height, tool_context)
        clinical_scores = calculate_clinical_scores(landmarks, image_width, image_height, tool_context)
        
        # Combine all metrics
        comprehensive_metrics = {
            "status": "success",
            "landmark_count": len(landmarks),
            "image_dimensions": [image_width, image_height],
            "symmetry_analysis": symmetry_metrics,
            "ear_analysis": ear_metrics,
            "clinical_scores": clinical_scores,
            "processing_timestamp": tool_context.state.get('current_timestamp', 'unknown')
        }
        
        return comprehensive_metrics
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Metrics calculation failed: {str(e)}",
            "error_type": type(e).__name__
        }

def calculate_symmetry_analysis(landmarks: List, width: int, height: int, 
                               tool_context: ToolContext) -> Dict[str, Any]:
    """
    Calculate detailed facial symmetry analysis
    
    Args:
        landmarks: List of facial landmarks
        width: Image width
        height: Image height
        tool_context: ADK tool context
        
    Returns:
        Symmetry analysis results
    """
    print("--- 🔄 Tool: calculate_symmetry_analysis called ---")
    
    try:
        # Key landmark indices for symmetry analysis
        NOSE_TIP = 1
        LEFT_EYE_CORNER = 33
        RIGHT_EYE_CORNER = 133
        LEFT_MOUTH_CORNER = 61
        RIGHT_MOUTH_CORNER = 291
        LEFT_EYEBROW_INNER = 70
        RIGHT_EYEBROW_INNER = 300
        LEFT_CHEEK = 234
        RIGHT_CHEEK = 454
        CHIN = 152
        FOREHEAD_CENTER = 9
        
        # Get landmark coordinates
        def get_coords(idx):
            if idx < len(landmarks):
                lm = landmarks[idx]
                return (lm.x * width, lm.y * height)
            return (0, 0)
        
        # Calculate facial midline using nose tip
        nose_x, nose_y = get_coords(NOSE_TIP)
        
        # Define symmetry pairs
        symmetry_pairs = [
            (LEFT_EYE_CORNER, RIGHT_EYE_CORNER, "eyes"),
            (LEFT_MOUTH_CORNER, RIGHT_MOUTH_CORNER, "mouth"),
            (LEFT_EYEBROW_INNER, RIGHT_EYEBROW_INNER, "eyebrows"),
            (LEFT_CHEEK, RIGHT_CHEEK, "cheeks")
        ]
        
        # Calculate asymmetry for each pair
        asymmetry_scores = {}
        distance_deviations = {}
        
        for left_idx, right_idx, feature_name in symmetry_pairs:
            left_x, left_y =left_x, left_y = get_coords(left_idx)
            right_x, right_y = get_coords(right_idx)
            
            # Calculate distance from midline
            left_dist = abs(left_x - nose_x)
            right_dist = abs(right_x - nose_x)
            
            # Calculate vertical alignment
            vertical_diff = abs(left_y - right_y)
            
            # Asymmetry as relative difference
            if max(left_dist, right_dist) > 0:
                horizontal_asymmetry = abs(left_dist - right_dist) / max(left_dist, right_dist)
            else:
                horizontal_asymmetry = 0
            
            # Vertical asymmetry normalized by face height
            vertical_asymmetry = vertical_diff / height
            
            # Combined asymmetry score
            combined_asymmetry = (horizontal_asymmetry + vertical_asymmetry) / 2
            
            asymmetry_scores[feature_name] = {
                "horizontal_asymmetry": round(horizontal_asymmetry, 4),
                "vertical_asymmetry": round(vertical_asymmetry, 4),
                "combined_asymmetry": round(combined_asymmetry, 4),
                "left_distance_from_midline": round(left_dist, 2),
                "right_distance_from_midline": round(right_dist, 2),
                "vertical_difference": round(vertical_diff, 2)
            }
            
            distance_deviations[f"{feature_name}_deviation"] = abs(left_dist - right_dist)
        
        # Calculate overall symmetry score
        combined_asymmetries = [scores["combined_asymmetry"] for scores in asymmetry_scores.values()]
        overall_asymmetry = np.mean(combined_asymmetries) if combined_asymmetries else 0
        overall_symmetry_score = max(0, 1 - overall_asymmetry * 2)  # Scale and invert
        
        # Calculate facial axis deviation
        chin_x, chin_y = get_coords(CHIN)
        forehead_x, forehead_y = get_coords(FOREHEAD_CENTER)
        
        # Facial axis angle (should be vertical for perfect symmetry)
        if forehead_y != chin_y:
            facial_axis_angle = math.degrees(math.atan2(forehead_x - chin_x, chin_y - forehead_y))
        else:
            facial_axis_angle = 0
        
        # Midline deviation
        face_center_x = width / 2
        midline_deviation = abs(nose_x - face_center_x) / width
        
        symmetry_analysis = {
            "overall_symmetry_score": round(overall_symmetry_score, 4),
            "overall_asymmetry": round(overall_asymmetry, 4),
            "feature_asymmetries": asymmetry_scores,
            "facial_axis_angle": round(facial_axis_angle, 2),
            "midline_deviation_ratio": round(midline_deviation, 4),
            "midline_deviation_pixels": round(abs(nose_x - face_center_x), 2),
            "analysis_quality": "good" if len(landmarks) >= 468 else "limited"
        }
        
        return symmetry_analysis
        
    except Exception as e:
        return {
            "error": f"Symmetry analysis failed: {str(e)}",
            "overall_symmetry_score": 0.0
        }

def calculate_ear_analysis(landmarks: List, width: int, height: int,
                            tool_context: ToolContext) -> Dict[str, Any]:
    """
    Calculate Eye Aspect Ratio (EAR) analysis for both eyes
    
    Args:
        landmarks: List of facial landmarks
        width: Image width
        height: Image height
        tool_context: ADK tool context
        
    Returns:
        EAR analysis results
    """
    print("--- 👁️ Tool: calculate_ear_analysis called ---")
    
    try:
        # Eye landmark indices
        LEFT_EYE_POINTS = [362, 385, 387, 263, 373, 380]
        RIGHT_EYE_POINTS = [33, 160, 158, 133, 153, 144]
        
        def get_coords(idx):
            if idx < len(landmarks):
                lm = landmarks[idx]
                return (lm.x * width, lm.y * height)
            return (0, 0)
        
        def calculate_ear(eye_points):
            """Calculate EAR for given eye points"""
            coords = [get_coords(idx) for idx in eye_points]
            
            if len(coords) < 6:
                return 0.0
            
            # Calculate vertical distances
            A = math.hypot(coords[1][0] - coords[5][0], coords[1][1] - coords[5][1])
            B = math.hypot(coords[2][0] - coords[4][0], coords[2][1] - coords[4][1])
            
            # Calculate horizontal distance
            C = math.hypot(coords[0][0] - coords[3][0], coords[0][1] - coords[3][1])
            
            if C == 0:
                return 0.0
            
            return (A + B) / (2.0 * C)
        
        # Calculate EAR for both eyes
        left_ear = calculate_ear(LEFT_EYE_POINTS)
        right_ear = calculate_ear(RIGHT_EYE_POINTS)
        
        # EAR analysis metrics
        ear_difference = abs(left_ear - right_ear)
        average_ear = (left_ear + right_ear) / 2
        ear_ratio = left_ear / right_ear if right_ear > 0 else 1.0
        
        # Eye closure detection (typical EAR thresholds)
        EAR_CLOSED_THRESHOLD = 0.21
        EAR_DROWSY_THRESHOLD = 0.25
        
        left_eye_state = "closed" if left_ear < EAR_CLOSED_THRESHOLD else \
                        "drowsy" if left_ear < EAR_DROWSY_THRESHOLD else "open"
        right_eye_state = "closed" if right_ear < EAR_CLOSED_THRESHOLD else \
                            "drowsy" if right_ear < EAR_DROWSY_THRESHOLD else "open"
        
        # Asymmetry assessment
        asymmetry_severity = "normal"
        if ear_difference > 0.1:
            asymmetry_severity = "severe"
        elif ear_difference > 0.05:
            asymmetry_severity = "moderate"
        elif ear_difference > 0.02:
            asymmetry_severity = "mild"
        
        # Calculate eye opening percentages (normalized)
        max_ear = max(left_ear, right_ear, 0.35)  # 0.35 as typical maximum
        left_opening_percent = min(100, (left_ear / max_ear) * 100) if max_ear > 0 else 0
        right_opening_percent = min(100, (right_ear / max_ear) * 100) if max_ear > 0 else 0
        
        ear_analysis = {
            "left_ear": round(left_ear, 4),
            "right_ear": round(right_ear, 4),
            "ear_difference": round(ear_difference, 4),
            "average_ear": round(average_ear, 4),
            "ear_ratio": round(ear_ratio, 4),
            "left_eye_state": left_eye_state,
            "right_eye_state": right_eye_state,
            "left_opening_percent": round(left_opening_percent, 1),
            "right_opening_percent": round(right_opening_percent, 1),
            "asymmetry_severity": asymmetry_severity,
            "both_eyes_open": left_eye_state == "open" and right_eye_state == "open",
            "blink_detected": left_eye_state == "closed" or right_eye_state == "closed",
            "drowsiness_indicated": left_eye_state == "drowsy" or right_eye_state == "drowsy"
        }
        
        return ear_analysis
        
    except Exception as e:
        return {
            "error": f"EAR analysis failed: {str(e)}",
            "left_ear": 0.0,
            "right_ear": 0.0,
            "ear_difference": 0.0
        }

def calculate_clinical_scores(landmarks: List, width: int, height: int,
                                tool_context: ToolContext) -> Dict[str, Any]:
    """
    Calculate clinical assessment scores based on medical literature
    
    Args:
        landmarks: List of facial landmarks
        width: Image width 
        height: Image height
        tool_context: ADK tool context
        
    Returns:
        Clinical scoring results
    """
    print("--- 🏥 Tool: calculate_clinical_scores called ---")
    
    try:
        # Get coordinates helper
        def get_coords(idx):
            if idx < len(landmarks):
                lm = landmarks[idx]
                return (lm.x * width, lm.y * height)
            return (0, 0)
        
        # Key landmark indices
        NOSE_TIP = 1
        LEFT_MOUTH_CORNER = 61
        RIGHT_MOUTH_CORNER = 291
        LEFT_EYE_CORNER = 33
        RIGHT_EYE_CORNER = 133
        LEFT_EYEBROW_INNER = 70
        RIGHT_EYEBROW_INNER = 300
        UPPER_LIP = 13
        LOWER_LIP = 14
        
        # Calculate facial measurements
        nose_x, nose_y = get_coords(NOSE_TIP)
        left_mouth_x, left_mouth_y = get_coords(LEFT_MOUTH_CORNER)
        right_mouth_x, right_mouth_y = get_coords(RIGHT_MOUTH_CORNER)
        left_eye_x, left_eye_y = get_coords(LEFT_EYE_CORNER)
        right_eye_x, right_eye_y = get_coords(RIGHT_EYE_CORNER)
        left_brow_x, left_brow_y = get_coords(LEFT_EYEBROW_INNER)
        right_brow_x, right_brow_y = get_coords(RIGHT_EYEBROW_INNER)
        upper_lip_x, upper_lip_y = get_coords(UPPER_LIP)
        lower_lip_x, lower_lip_y = get_coords(LOWER_LIP)
        
        # 1. House-Brackmann Scale Components (for Bell's Palsy assessment)
        # Forehead movement (eyebrow height difference)
        eyebrow_height_diff = abs(left_brow_y - right_brow_y)
        eyebrow_asymmetry_score = min(4, eyebrow_height_diff / (height * 0.02)) # Normalized
        
        # Eye closure assessment (from EAR analysis)
        ear_analysis = calculate_ear_analysis(landmarks, width, height, tool_context)
        eye_closure_asymmetry = ear_analysis.get('ear_difference', 0) * 20  # Scale up
        
        # Mouth movement (corner asymmetry)
        mouth_height_diff = abs(left_mouth_y - right_mouth_y)
        mouth_asymmetry_score = min(4, mouth_height_diff / (height * 0.015))
        
        # Composite House-Brackmann-inspired score (1-6 scale, 1=normal, 6=severe)
        hb_components = [eyebrow_asymmetry_score, eye_closure_asymmetry, mouth_asymmetry_score]
        hb_average = np.mean(hb_components)
        
        if hb_average <= 0.5:
            hb_grade = 1  # Normal
            hb_description = "Normal facial function"
        elif hb_average <= 1.0:
            hb_grade = 2  # Mild dysfunction
            hb_description = "Mild facial weakness"
        elif hb_average <= 1.5:
            hb_grade = 3  # Moderate dysfunction  
            hb_description = "Moderate facial weakness"
        elif hb_average <= 2.0:
            hb_grade = 4  # Moderately severe
            hb_description = "Moderately severe facial weakness"
        elif hb_average <= 3.0:
            hb_grade = 5  # Severe
            hb_description = "Severe facial weakness"
        else:
            hb_grade = 6  # Total paralysis
            hb_description = "Complete facial paralysis"
        
        # 2. Facial Asymmetry Index (FAI)
        # Based on published medical literature
        face_width = abs(right_eye_x - left_eye_x)
        mouth_deviation = abs((left_mouth_x + right_mouth_x) / 2 - nose_x)
        
        if face_width > 0:
            fai_score = (mouth_deviation / face_width) * 100  # Percentage
        else:
            fai_score = 0
        
        # 3. Sunnybrook Facial Grading System components
        # Voluntary movement scores (estimated from static analysis)
        forehead_score = max(0, 5 - eyebrow_asymmetry_score)  # 0-5 scale
        eye_closure_score = max(0, 5 - eye_closure_asymmetry)
        mouth_movement_score = max(0, 5 - mouth_asymmetry_score)
        
        voluntary_movement_total = forehead_score + eye_closure_score + mouth_movement_score
        
        # Resting symmetry (inverted from asymmetry)
        symmetry_analysis = calculate_symmetry_analysis(landmarks, width, height, tool_context)
        resting_symmetry_score = symmetry_analysis.get('overall_symmetry_score', 0) * 20  # 0-20 scale
        
        # Sunnybrook composite score (higher is better, 0-120 scale)
        sunnybrook_score = voluntary_movement_total + resting_symmetry_score
        
        # 4. Clinical severity classification
        if hb_grade <= 2 and fai_score <= 5:
            severity_classification = "normal_to_mild"
            clinical_concern = "low"
        elif hb_grade <= 3 and fai_score <= 15:
            severity_classification = "mild_to_moderate"
            clinical_concern = "moderate"
        elif hb_grade <= 4 and fai_score <= 25:
            severity_classification = "moderate_to_severe"
            clinical_concern = "high"
        else:
            severity_classification = "severe"
            clinical_concern = "very_high"
        
        # 5. Functional assessment
        functional_impact = {
            "speech_affected": mouth_asymmetry_score > 1.5,
            "eating_affected": mouth_asymmetry_score > 2.0,
            "eye_protection_compromised": eye_closure_asymmetry > 2.0,
            "cosmetic_concern": hb_average > 1.0
        }
        
        affected_functions = sum(functional_impact.values())
        
        clinical_scores = {
            "house_brackmann_grade": hb_grade,
            "house_brackmann_description": hb_description,
            "house_brackmann_components": {
                "eyebrow_asymmetry": round(eyebrow_asymmetry_score, 2),
                "eye_closure_asymmetry": round(eye_closure_asymmetry, 2),
                "mouth_asymmetry": round(mouth_asymmetry_score, 2)
            },
            "facial_asymmetry_index": round(fai_score, 2),
            "sunnybrook_score": round(sunnybrook_score, 1),
            "sunnybrook_components": {
                "forehead_movement": round(forehead_score, 1),
                "eye_closure": round(eye_closure_score, 1),
                "mouth_movement": round(mouth_movement_score, 1),
                "resting_symmetry": round(resting_symmetry_score, 1)
            },
            "severity_classification": severity_classification,
            "clinical_concern_level": clinical_concern,
            "functional_impact": functional_impact,
            "affected_functions_count": affected_functions,
            "measurements": {
                "eyebrow_height_difference_px": round(eyebrow_height_diff, 2),
                "mouth_height_difference_px": round(mouth_height_diff, 2),
                "mouth_deviation_from_midline_px": round(mouth_deviation, 2),
                "face_width_px": round(face_width, 2)
            }
        }
        
        return clinical_scores
        
    except Exception as e:
        return {
            "error": f"Clinical scoring failed: {str(e)}",
            "house_brackmann_grade": 1,
            "severity_classification": "unknown"
        }

def calculate_temporal_analysis(frame_results: List[Dict[str, Any]], 
                                tool_context: ToolContext) -> Dict[str, Any]:
    """
    Analyze temporal patterns across multiple frames
    
    Args:
        frame_results: List of frame analysis results
        tool_context: ADK tool context
        
    Returns:
        Temporal analysis results
    """
    print("--- ⏱️ Tool: calculate_temporal_analysis called ---")
    
    try:
        if len(frame_results) < 2:
            return {
                "status": "insufficient_data",
                "message": "Need at least 2 frames for temporal analysis"
            }
        
        # Extract time series data
        timestamps = [r.get('timestamp', 0) for r in frame_results]
        symmetry_scores = [r.get('symmetry_score', 0) for r in frame_results]
        ear_differences = [r.get('ear_difference', 0) for r in frame_results]
        severity_scores = [r.get('severity_score', 0) for r in frame_results]
        
        # Calculate temporal statistics
        def calculate_trend(values):
            if len(values) < 2:
                return 0
            # Simple linear trend calculation
            x = np.arange(len(values))
            coeffs = np.polyfit(x, values, 1)
            return coeffs[0]  # Slope
        
        def calculate_stability(values):
            if len(values) < 2:
                return 1.0
            return 1.0 / (1.0 + np.std(values))  # Inverse of std dev
        
        # Analyze trends
        symmetry_trend = calculate_trend(symmetry_scores)
        ear_trend = calculate_trend(ear_differences)
        severity_trend = calculate_trend(severity_scores)
        
        # Analyze stability
        symmetry_stability = calculate_stability(symmetry_scores)
        ear_stability = calculate_stability(ear_differences)
        severity_stability = calculate_stability(severity_scores)
        
        # Detect significant changes
        symmetry_range = max(symmetry_scores) - min(symmetry_scores)
        ear_range = max(ear_differences) - min(ear_differences)
        severity_range = max(severity_scores) - min(severity_scores)
        
        # Classify temporal patterns
        pattern_classification = "stable"
        if abs(symmetry_trend) > 0.01 or abs(severity_trend) > 0.5:
            pattern_classification = "trending"
        if symmetry_range > 0.2 or ear_range > 0.1 or severity_range > 3:
            pattern_classification = "variable"
        
        # Detect anomalous frames
        symmetry_mean = np.mean(symmetry_scores)
        symmetry_std = np.std(symmetry_scores)
        
        anomalous_frames = []
        for i, (sym, ear, sev) in enumerate(zip(symmetry_scores, ear_differences, severity_scores)):
            anomaly_score = 0
            if abs(sym - symmetry_mean) > 2 * symmetry_std:
                anomaly_score += 1
            if ear > np.mean(ear_differences) + 2 * np.std(ear_differences):
                anomaly_score += 1
            if sev > np.mean(severity_scores) + 2 * np.std(severity_scores):
                anomaly_score += 1
            
            if anomaly_score >= 2:
                anomalous_frames.append(i)
        
        temporal_analysis = {
            "frames_analyzed": len(frame_results),
            "time_span_seconds": max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0,
            "trends": {
                "symmetry_trend": round(symmetry_trend, 6),
                "ear_trend": round(ear_trend, 6), 
                "severity_trend": round(severity_trend, 4)
            },
            "stability_scores": {
                "symmetry_stability": round(symmetry_stability, 4),
                "ear_stability": round(ear_stability, 4),
                "severity_stability": round(severity_stability, 4)
            },
            "value_ranges": {
                "symmetry_range": round(symmetry_range, 4),
                "ear_range": round(ear_range, 4),
                "severity_range": round(severity_range, 2)
            },
            "pattern_classification": pattern_classification,
            "anomalous_frames": anomalous_frames,
            "anomaly_rate": round(len(anomalous_frames) / len(frame_results), 3),
            "overall_temporal_quality": round((symmetry_stability + ear_stability + severity_stability) / 3, 3)
        }
        
        return temporal_analysis
        
    except Exception as e:
        return {
            "error": f"Temporal analysis failed: {str(e)}",
            "pattern_classification": "unknown"
        }