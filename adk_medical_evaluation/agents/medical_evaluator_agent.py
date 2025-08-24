# adk_medical_evaluation/agents/medical_evaluator_agent.py
"""
Main medical evaluation agent using ADK
Project: mindfulfocus-470008
Model: Gemini 2.5 Pro

Fixes:
- Each tool accepts `tool_context` as the first arg (required by ADK Runner).
- No defaults in function signatures (Gemini function schema rejects them).
- Defaults applied inside function body instead.
"""

import os
import sys
import inspect
from typing import Dict, Any, List

from google.adk.agents import Agent

# Add tools path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# ---- Import your existing implementations (UNTOUCHED) ----
from tools.pipeline_analyzer import (
    analyze_pipeline_frame as _impl_analyze_pipeline_frame,
    calculate_performance_metrics as _impl_calculate_performance_metrics,
    save_evaluation_report as _impl_save_evaluation_report,
    generate_medical_assessment as _impl_generate_medical_assessment,
)
from tools.data_processor import (
    process_camera_stream as _impl_process_camera_stream,
    batch_process_frames as _impl_batch_process_frames,
    aggregate_results as _impl_aggregate_results,
    export_results as _impl_export_results,
)
from tools.file_manager import (
    save_frame_locally as _impl_save_frame_locally,
    manage_session_files as _impl_manage_session_files,
    cleanup_old_files as _impl_cleanup_old_files,
)

def _call_impl(func, payload: Dict[str, Any]):
    """Call underlying tool impl with kwargs if possible, else single dict."""
    try:
        return func(**payload)  # kwargs style
    except TypeError:
        return func(payload)    # single dict style

# --------------------------- WRAPPER TOOLS (NO DEFAULTS) ---------------------------

def process_camera_stream(tool_context, duration_seconds: int) -> Dict[str, Any]:
    """Capture frames from a live camera and persist metadata."""
    payload = {
        "duration_seconds": int(duration_seconds),
        "camera_index": 0,
        "fps_target": 10,
        "width": 1280,
        "height": 720,
    }
    return _call_impl(_impl_process_camera_stream, payload)

def batch_process_frames(tool_context, batch_size: int) -> Dict[str, Any]:
    """Analyze captured frames in batches for landmarks and metrics."""
    payload = {
        "batch_size": int(batch_size),
        "max_workers": 2,
    }
    return _call_impl(_impl_batch_process_frames, payload)

def calculate_performance_metrics(tool_context, total_frames: int, processed_frames: int) -> Dict[str, Any]:
    """Compute aggregate performance stats from processed frames."""
    tf = int(total_frames)
    pf = int(processed_frames)
    payload = {
        "total_frames": tf,
        "processed_frames": pf,
        "avg_fps": 0.0,
        "avg_latency_ms": 0.0,
        "success_count": pf,
        "error_count": max(0, tf - pf),
        "notes": [],
    }
    return _call_impl(_impl_calculate_performance_metrics, payload)

def generate_medical_assessment(tool_context,
                                frames_considered: int,
                                severity_score_mean: float,
                                symmetry_score_mean: float,
                                ear_diff_mean: float) -> Dict[str, Any]:
    """Create clinical assessment from computed metrics."""
    payload = {
        "frames_considered": int(frames_considered),
        "severity_score_mean": float(severity_score_mean),
        "symmetry_score_mean": float(symmetry_score_mean),
        "ear_diff_mean": float(ear_diff_mean),
    }
    return _call_impl(_impl_generate_medical_assessment, payload)

def aggregate_results(tool_context, include_frames: bool, include_metrics: bool, include_assessment: bool) -> Dict[str, Any]:
    """Aggregate all intermediate results for export."""
    payload = {
        "include_frames": bool(include_frames),
        "include_metrics": bool(include_metrics),
        "include_assessment": bool(include_assessment),
    }
    return _call_impl(_impl_aggregate_results, payload)

def export_results(tool_context, formats: List[str]) -> Dict[str, Any]:
    """Export aggregated results into specific formats."""
    payload = {"formats": list(formats)}
    return _call_impl(_impl_export_results, payload)

def save_evaluation_report(tool_context, title: str, summary: str, include_recommendations: bool) -> Dict[str, Any]:
    """Generate and persist the final evaluation report."""
    payload = {
        "title": str(title),
        "summary": str(summary),
        "include_recommendations": bool(include_recommendations),
    }
    return _call_impl(_impl_save_evaluation_report, payload)

def manage_session_files(tool_context, keep_reports: bool, keep_latest_frames: int) -> Dict[str, Any]:
    """Organize/cleanup outputs for the current session."""
    payload = {
        "keep_reports": bool(keep_reports),
        "keep_latest_frames": int(keep_latest_frames),
    }
    return _call_impl(_impl_manage_session_files, payload)

def cleanup_old_files(tool_context, days: int, max_total_gb: float) -> Dict[str, Any]:
    """Cleanup aged files with retention policy."""
    payload = {
        "days": int(days),
        "max_total_gb": float(max_total_gb),
    }
    return _call_impl(_impl_cleanup_old_files, payload)

def save_frame_locally(tool_context, frame_id: int, image_path: str, dest_dir: str) -> Dict[str, Any]:
    """Persist a single frame to local storage."""
    payload = {
        "frame_id": int(frame_id),
        "image_path": str(image_path),
        "dest_dir": str(dest_dir) or "evaluation_data",
    }
    return _call_impl(_impl_save_frame_locally, payload)

def analyze_pipeline_frame(tool_context, frame_id: int, image_path: str) -> Dict[str, Any]:
    """Analyze a single image frame for landmarks and metrics."""
    payload = {
        "frame_id": int(frame_id),
        "image_path": str(image_path),
    }
    return _call_impl(_impl_analyze_pipeline_frame, payload)

# ----------------------------- AGENT FACTORY -----------------------------

def create_medical_evaluator_agent(model: str = "gemini-2.5-pro") -> Agent:
    agent = Agent(
        name="medical_pipeline_evaluator",
        model=model,
        description="Comprehensive medical facial analysis pipeline evaluator using advanced AI",
        instruction=(
            "You evaluate facial analysis pipelines end-to-end. "
            "Prefer tool calls when parameters are provided in the prompt. "
            "Return quantitative findings and practical recommendations."
        ),
        tools=[
            analyze_pipeline_frame,
            process_camera_stream,
            batch_process_frames,
            calculate_performance_metrics,
            generate_medical_assessment,
            aggregate_results,
            export_results,
            save_evaluation_report,
            save_frame_locally,
            manage_session_files,
            cleanup_old_files,
        ],
        output_key="last_evaluation_response",
    )

    # DEBUG: print tool signatures so you can confirm `tool_context` is present
    print(f"✅ Medical Evaluator Agent created with model: {model}")
    print(f"🔧 Available tools: {len(agent.tools)}")
    for fn in agent.tools:
        print(f"   • tool {fn.__name__} sig -> {inspect.signature(fn)}")

    return agent
