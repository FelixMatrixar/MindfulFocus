"""
Agent Garden Configuration for Medical Pipeline Evaluator with Gemini
"""

AGENT_CONFIG = {
    "name": "MedicalPipelineEvaluator",
    "description": "Evaluates medical face analysis pipeline performance using Gemini",
    "tools": [
        "grounding_with_google_search",
        "function_calling", 
        "custom_apis",
        "bigquery",
        "cloud_sql_postgresql"
    ],
    "model": "gemini-2.5-pro",  # Using Gemini instead
    "vertex_ai_config": {
        "project_id": "mindfulfocus-470008",
        "location": "us-central1",
        "model": "gemini-2.5-pro"
    },
    "capabilities": [
        "vision_analysis",
        "medical_validation", 
        "performance_evaluation",
        "real_time_monitoring",
        "report_generation"
    ]
}