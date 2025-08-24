# adk_medical_evaluation/config/agent_config.py
"""
Agent configuration for Medical Pipeline Evaluator
Project ID: mindfulfocus-470008
Model: Gemini 2.5 Pro
"""

import os
from typing import Dict, Any

# Project Configuration
PROJECT_ID = "mindfulfocus-470008"
MODEL_NAME = "gemini-2.5-pro"
LOCATION = "us-central1"

# ADK Configuration
ADK_CONFIG = {
    "project_id": PROJECT_ID,
    "model": MODEL_NAME,
    "use_vertex_ai": False,  # Using direct API for local setup
    "api_key_env": "GOOGLE_API_KEY",
    "temperature": 0.3,
    "max_output_tokens": 8192,
    "top_p": 0.8,
    "top_k": 40
}

# Agent Configurations
AGENT_CONFIGS = {
    "medical_evaluator": {
        "name": "medical_pipeline_evaluator",
        "model": MODEL_NAME,
        "description": "Main medical pipeline evaluation agent",
        "max_turns": 50,
        "timeout_seconds": 300
    },
    
    "performance_analyzer": {
        "name": "performance_analyzer",
        "model": MODEL_NAME,
        "description": "Specialized performance analysis agent",
        "max_turns": 30,
        "timeout_seconds": 180
    },
    
    "safety_assessor": {
        "name": "safety_assessor",
        "model": MODEL_NAME,
        "description": "Medical safety and compliance assessment agent",
        "max_turns": 25,
        "timeout_seconds": 120
    },
    
    "report_generator": {
        "name": "report_generator",
        "model": MODEL_NAME,
        "description": "Comprehensive report generation agent",
        "max_turns": 20,
        "timeout_seconds": 240
    }
}

# Evaluation Settings
EVALUATION_CONFIG = {
    "default_duration_seconds": 300,
    "frames_per_analysis": 1,  # Analyze every frame
    "max_frames_per_session": 1000,
    "min_landmarks_threshold": 400,
    "symmetry_threshold": 0.7,
    "ear_difference_threshold": 0.05,
    "severity_threshold": 7.0
}

# Local Storage Settings
STORAGE_CONFIG = {
    "database_path": "local_database/medical_evaluation.db",
    "images_dir": "evaluation_data/images",
    "reports_dir": "evaluation_data/reports",
    "metrics_dir": "evaluation_data/metrics",
    "sessions_dir": "evaluation_data/sessions",
    "logs_dir": "evaluation_data/logs"
}

def get_model_config() -> Dict[str, Any]:
    """Get model configuration for Gemini"""
    return {
        "model": MODEL_NAME,
        "temperature": ADK_CONFIG["temperature"],
        "max_output_tokens": ADK_CONFIG["max_output_tokens"],
        "top_p": ADK_CONFIG["top_p"],
        "top_k": ADK_CONFIG["top_k"],
        "safety_settings": [
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            }
        ]
    }

def get_api_config() -> Dict[str, Any]:
    """Get API configuration"""
    return {
        "project_id": PROJECT_ID,
        "location": LOCATION,
        "api_key": os.getenv("GOOGLE_API_KEY", ""),
        "use_vertex_ai": "False"  # Corrected to boolean False
    }